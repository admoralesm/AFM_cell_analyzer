"""
Four-point cell tracking from a compression video.

This is the tracking engine from the desktop Video Tracker, with the Tk
interface stripped out: everything here takes arrays and paths and gives
back arrays, so the same code can be driven by a Streamlit page, a script
or a batch job. app.py's "Cell motion" tab is the only caller today.

What it does, and why it is built this way:

  * Each of the four picked markers is tracked as a PATCH of a dozen
    sub-features rather than as one pixel. A single Lucas-Kanade point on a
    living cell is lost the moment its texture changes; a patch survives
    because the corner's position each frame is the MEDIAN of whichever
    sub-features are still good, each predicting the corner through the
    offset it started with.

  * Every frame the LK estimate is cross-checked against a normalised
    template match of the ORIGINAL first-frame patch. Small agreements are
    blended in, which removes the slow drift that otherwise accumulates
    over a few hundred frames, and a corner that is lost outright is
    recovered by searching a wider window for its template.

  * A marker that slides under the probe stops being tracked, but the cell
    it belongs to has not stopped moving. Rather than freezing it, the
    group reconciler works out how the markers that ARE tracked moved --
    one scale and one shift, fitted to all of them -- and moves the lost
    one the same way, so its displacement goes on counting as cell motion.
    The same transform pulls back any marker whose motion grossly
    disagrees with it (it stuck to the probe, or jumped to a decoy).

Nothing here imports streamlit, and nothing here writes a file.
"""

from __future__ import annotations

import math

import numpy as np

try:
    import cv2
except Exception as _exc:  # pragma: no cover - only on a broken deploy
    cv2 = None
    CV2_ERROR = str(_exc)
else:
    CV2_ERROR = ""


# --------------------------------------------------------------- settings ---
#
# The numbers the desktop app arrived at on C2C12 compression video. They
# are module-level rather than arguments because every one of them has a
# measured reason, and a caller that wants a different value should say so
# once, by name, rather than threading fifteen parameters through.

DEFAULT_FPS = 30.0

# Lucas-Kanade. A 41-pixel window is wide for LK, and deliberately: at 10x
# a marker on a beating cardiomyocyte can move several pixels between
# frames, and a narrow window loses it.
WINDOW_SIZE = 41
MAX_LEVEL = 5
CRITERIA_ITERS = 50
CRITERIA_EPS = 0.001
MIN_EIG_THRESH = 1e-4
# Forward-backward check: track forward, track the result back, and keep
# the point only if it lands where it started. This is what catches a
# marker that has quietly jumped to a different feature.
FB_THRESHOLD = 2.0

# Frame preparation. CLAHE because phase-contrast video is unevenly lit
# across the field, and a marker at the dim edge tracks badly without it.
USE_CLAHE = True
CLAHE_CLIP = 2.0
PREP_DENOISE_KSIZE = 3

# The patch: how many sub-features per corner, and how far out they may sit.
FEATURES_PER_CORNER = 12
PATCH_RADIUS_FRAC = 0.75          # of the LK window

# Drift anchoring against the first frame.
ANCHOR_ENABLED = True
ANCHOR_TEMPLATE_SIZE = 21         # odd
ANCHOR_SEARCH_PAD = 10            # px around the LK estimate
ANCHOR_CONF = 0.60                # minimum NCC to trust the anchor
ANCHOR_BLEND = 0.35               # weight of anchor vs LK when they agree
ANCHOR_MAX_CORR_PX = 4.0          # ignore the anchor if it disagrees more
ANCHOR_RECOVER_CONF = 0.50        # minimum NCC to re-lock a lost corner
ANCHOR_RECOVER_PAD = 40           # px search radius for recovery
# When Lucas-Kanade reports a position CONFIDENTLY and it is wrong, nothing
# above catches it: the corner is not NaN, so recovery never runs, and it is
# too far away to blend. That happens when something large enters the field
# -- the probe -- and corrupts the coarse level of the pyramid, which is
# shared by every marker. So when the template does not match near the LK
# estimate at all, the estimate is treated as suspect and the template is
# looked for over the wider window instead.
ANCHOR_SUSPECT_CONF = 0.35        # below this near the LK point, look wider

# Gentle mode: the anchor nudges instead of snapping, and the group
# reconciler tolerates more before it corrects a marker. Softer traces, a
# little less responsive.
GENTLE_ANCHOR_BLEND = 0.18
GENTLE_ANCHOR_MAX_CORR_PX = 2.5
GENTLE_POSITION_SMOOTH_WINDOW = 9
GENTLE_GROUP_MIN_DEV = 10.0
GENTLE_GROUP_DEV_FRAC = 0.9

# Group reconciliation tolerance, in pixels and as a fraction of the
# group's own speed, so real contraction survives but a marker that
# freezes while the cell slides is corrected.
GROUP_MIN_DEV = 6.0
GROUP_DEV_FRAC = 0.6

# Output smoothing (Savitzky-Golay). Position first, then the derivative,
# then the speed: each stage removes a different kind of noise.
POSITION_SMOOTH_WINDOW = 7
VELOCITY_SMOOTH_WINDOW = 9
SPEED_SMOOTH_WINDOW = 7
SMOOTH_POLYORDER = 2
INTERP_MAX_GAP = 30               # frames of NaN worth bridging

SUBPIX_WIN = 5

POINT_COLORS = ("#E6194B", "#3CB44B", "#4363D8", "#F58231")
POINT_LABELS = ("P1", "P2", "P3", "P4")


# ---------------------------------------------------------------- helpers ---

def clamp(value, low, high):
    """``value`` held inside [low, high]."""
    return max(low, min(high, value))


def ensure_odd(k):
    """The nearest odd integer at or above ``k``, never below 3."""
    k = max(3, int(k))
    return k if k % 2 == 1 else k + 1


def format_time(seconds):
    """Seconds as mm:ss.mmm, or hh:mm:ss.mmm past an hour."""
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    rest = seconds % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{rest:06.3f}"
    return f"{minutes:02d}:{rest:06.3f}"


def interpolate_nan_1d(values, max_gap=INTERP_MAX_GAP):
    """
    Bridge short runs of NaN by straight line; leave long ones alone.

    A marker lost for three frames is a marker the fit can carry across. One
    lost for a hundred is a marker that was not measured, and inventing a
    line through it would put a number in the export that no frame of the
    video supports. Returns (filled, mask of what was invented).
    """
    out = np.asarray(values, dtype=np.float64).copy()
    n = out.shape[0]
    invented = np.zeros(n, dtype=bool)
    if n == 0:
        return out, invented
    finite = np.isfinite(out)
    if not finite.any():
        return out, invented
    i = 0
    while i < n:
        if finite[i]:
            i += 1
            continue
        j = i
        while j < n and not finite[j]:
            j += 1
        left, right = i - 1, j
        if left >= 0 and right < n and (j - i) <= int(max_gap):
            y0, y1 = out[left], out[right]
            for k in range(i, j):
                t = (k - left) / float(right - left)
                out[k] = y0 + t * (y1 - y0)
                invented[k] = True
        i = j
    return out, invented


# ----------------------------------------------------- signal processing ---
#
# Savitzky-Golay by hand rather than through scipy: the coefficients are six
# lines, and this module then has no dependency beyond numpy and cv2, which
# is what lets it run wherever the video does.

def savgol_coeffs(window, polyorder, deriv=0, delta=1.0):
    """Savitzky-Golay filter coefficients for one derivative order."""
    window = int(window)
    if window % 2 == 0:
        raise ValueError("a Savitzky-Golay window must be odd")
    if polyorder >= window:
        raise ValueError("polyorder must be below the window length")
    half = window // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    design = np.vander(x, int(polyorder) + 1, increasing=True)
    return (np.linalg.pinv(design)[deriv]
            * (math.factorial(int(deriv)) / (float(delta) ** int(deriv))))


def _correlate(y, kernel):
    """``y`` against ``kernel``, reflected at both ends."""
    y = np.asarray(y, dtype=np.float64)
    if y.size <= 1:
        return y.copy()
    half = len(kernel) // 2
    padded = np.pad(y, (half, half), mode="reflect")
    out = np.zeros(y.size, dtype=np.float64)
    for j in range(len(kernel)):
        out += kernel[j] * padded[j:j + y.size]
    return out


def _segmental(y, function):
    """
    Apply ``function`` to each run of finite values, leaving NaN as NaN.

    Smoothing across a gap would carry the value from one side of a lost
    stretch to the other and call it data.
    """
    y = np.asarray(y, dtype=np.float64)
    out = np.full_like(y, np.nan)
    finite = np.isfinite(y)
    n = y.size
    i = 0
    while i < n:
        if not finite[i]:
            i += 1
            continue
        j = i
        while j < n and finite[j]:
            j += 1
        out[i:j] = function(y[i:j])
        i = j
    return out


def _fit_window(window, length):
    """The largest odd window at or below ``window`` that fits ``length``."""
    w = int(window)
    if w % 2 == 0:
        w += 1
    if w > length:
        w = length if length % 2 == 1 else length - 1
    return w


def smooth_series(y, window, polyorder=SMOOTH_POLYORDER):
    """Savitzky-Golay smoothing, run separately on each finite stretch."""
    if window is None or int(window) < 3:
        return np.asarray(y, dtype=np.float64).copy()

    def one(segment):
        w = _fit_window(window, segment.size)
        if w >= 3 and polyorder < w:
            return _correlate(segment, savgol_coeffs(w, polyorder))
        return segment.astype(np.float64)

    return _segmental(y, one)


def smooth_derivative(y, window, polyorder, delta):
    """d/dt by Savitzky-Golay, falling back to np.gradient on short runs."""

    def one(segment):
        if segment.size < 2:
            return np.full(segment.size, np.nan)
        w = _fit_window(window, segment.size)
        if w >= 3 and polyorder < w:
            return _correlate(
                segment, savgol_coeffs(w, polyorder, deriv=1, delta=delta))
        return np.gradient(segment, delta)

    return _segmental(y, one)


def plain_derivative(y, delta):
    """d/dt with no smoothing at all."""

    def one(segment):
        if segment.size < 2:
            return np.full(segment.size, np.nan)
        return np.gradient(segment, delta)

    return _segmental(y, one)


# ----------------------------------------------------- frame preparation ---

_CLAHE = None


def gray_for_lk(bgr, contrast=1.0, use_clahe=USE_CLAHE):
    """
    The grey frame the tracker actually sees.

    Denoised, contrast-equalised tile by tile, and optionally brightened.
    The same preparation is used on every frame, which matters more than
    which preparation it is: a tracker comparing an equalised frame with a
    raw one measures the preparation, not the cell.
    """
    global _CLAHE
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    k = int(PREP_DENOISE_KSIZE)
    if k >= 3:
        if k % 2 == 0:
            k += 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    if use_clahe:
        if _CLAHE is None:
            _CLAHE = cv2.createCLAHE(clipLimit=float(CLAHE_CLIP),
                                     tileGridSize=(8, 8))
        gray = _CLAHE.apply(gray)
    contrast = float(contrast)
    if abs(contrast - 1.0) > 1e-3:
        gray = cv2.convertScaleAbs(gray, alpha=contrast, beta=0.0)
    return gray


def refine_corners_subpix(gray, points_xy, win=SUBPIX_WIN):
    """
    Pull each picked point onto the nearest corner, to sub-pixel accuracy.

    A click is worth about two pixels. Everything downstream is measured in
    tenths of one, so the picks are snapped before the first frame is left.
    """
    points = np.asarray(points_xy, dtype=np.float32)
    if points.size == 0:
        return points
    shaped = points.reshape(-1, 1, 2).copy()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 40, 0.001)
    try:
        cv2.cornerSubPix(gray, shaped, (int(win), int(win)), (-1, -1),
                         criteria)
    except cv2.error:
        return points.reshape(-1, 2)
    return shaped.reshape(-1, 2)


# ------------------------------------------------------------ video I/O ---

def video_info(path):
    """{fps, frame_count, width, height, duration} for a video file."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("Could not open that video file.")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = float(fps) if fps and fps > 0 else DEFAULT_FPS
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if count <= 0 or width <= 0 or height <= 0:
            raise RuntimeError(
                "That file opened but reports no frames. It may be a format "
                "OpenCV cannot decode; converting it to .mp4 usually fixes it.")
        return {"fps": fps, "frame_count": count, "width": width,
                "height": height, "duration": count / fps}
    finally:
        capture.release()


def read_frame(path, index):
    """One BGR frame by index, or None."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, index)))
        ok, frame = capture.read()
        return frame if ok else None
    finally:
        capture.release()


# -------------------------------------------------------- the LK tracker ---

def _ncc_match(gray, template, cx, cy, pad):
    """
    Where ``template`` sits in ``gray`` near (cx, cy), to sub-pixel accuracy.

    Normalised cross-correlation, with a parabola fitted through the peak
    and its two neighbours in each direction. Returns (x, y, confidence),
    or None when the search window will not fit.
    """
    th, tw = template.shape[:2]
    height, width = gray.shape[:2]
    x0 = max(0, int(round(cx - tw / 2.0 - pad)))
    y0 = max(0, int(round(cy - th / 2.0 - pad)))
    x1 = min(width, x0 + tw + 2 * int(pad))
    y1 = min(height, y0 + th + 2 * int(pad))
    if x1 - x0 < tw + 2 or y1 - y0 < th + 2:
        return None
    region = gray[y0:y1, x0:x1]
    try:
        response = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
    except cv2.error:
        return None
    _lo, best, _loc_lo, loc = cv2.minMaxLoc(response)
    mx, my = int(loc[0]), int(loc[1])

    def peak(before, here, after):
        denominator = float(before) - 2.0 * float(here) + float(after)
        if abs(denominator) < 1e-9:
            return 0.0
        return clamp(0.5 * (float(before) - float(after)) / denominator,
                     -0.5, 0.5)

    dx = dy = 0.0
    if 0 < mx < response.shape[1] - 1:
        dx = peak(response[my, mx - 1], response[my, mx], response[my, mx + 1])
    if 0 < my < response.shape[0] - 1:
        dy = peak(response[my - 1, mx], response[my, mx], response[my + 1, mx])
    return (float(x0 + mx + dx + tw / 2.0),
            float(y0 + my + dy + th / 2.0),
            float(best))


def _seed_patches(gray, corners, patch_radius, per_corner):
    """
    A cloud of trackable sub-features around each picked corner.

    Returns (seeds, owners, offsets): where each sub-feature starts, which
    corner it belongs to, and the offset through which it predicts that
    corner. The corner itself is always the first seed of its own patch, so
    a corner with no texture around it still has one point to stand on.
    """
    height, width = gray.shape[:2]
    seeds, owners, offsets = [], [], []
    for index in range(corners.shape[0]):
        cx, cy = float(corners[index, 0]), float(corners[index, 1])
        local = [(0.0, 0.0)]
        x0 = max(0, int(cx - patch_radius))
        y0 = max(0, int(cy - patch_radius))
        x1 = min(width, int(cx + patch_radius))
        y1 = min(height, int(cy + patch_radius))
        if x1 - x0 >= 5 and y1 - y0 >= 5:
            try:
                found = cv2.goodFeaturesToTrack(
                    gray[y0:y1, x0:x1], maxCorners=int(per_corner) - 1,
                    qualityLevel=0.01, minDistance=3, blockSize=5)
            except cv2.error:
                found = None
            if found is not None:
                for point in found.reshape(-1, 2):
                    offsets_xy = ((x0 + float(point[0])) - cx,
                                  (y0 + float(point[1])) - cy)
                    local.append(offsets_xy)
        if len(local) < 3:
            # A featureless patch still gets a cross of four, so the corner
            # is carried by geometry rather than by one pixel.
            step = patch_radius * 0.5
            local += [(-step, 0.0), (step, 0.0), (0.0, -step), (0.0, step)]
        for (ox, oy) in local:
            sx, sy = cx + ox, cy + oy
            if 0 <= sx < width and 0 <= sy < height:
                seeds.append((sx, sy))
                owners.append(index)
                offsets.append((ox, oy))
    return (np.asarray(seeds, dtype=np.float32).reshape(-1, 1, 2),
            np.asarray(owners, dtype=int),
            np.asarray(offsets, dtype=np.float32))


def track_points(video_path, points_xy, start_frame=0, end_frame=None,
                 window_size=WINDOW_SIZE, fb_threshold=FB_THRESHOLD,
                 contrast=1.0, gentle=False, anchor=ANCHOR_ENABLED,
                 progress=None):
    """
    Track the picked markers from ``start_frame`` to ``end_frame``.

    Each marker is a patch of sub-features tracked by forward-backward
    Lucas-Kanade and reduced to one position per frame by the median of
    whichever sub-features survived; the result is then cross-checked
    against the marker's first-frame template to hold off drift.

    ``progress(done, total)`` is called as it goes and may return True to
    stop early. ``gentle`` softens the anchor and widens the smoothing.

    Returns a dict:
        frames      (N,)    the video frame index of each row
        points      (N,K,2) tracked position, NaN where the marker was lost
        status      (N,K)   1 where Lucas-Kanade itself followed the
                            marker. A 0 with a finite position beside it is
                            a marker the first-frame template re-found.
        fps, start_frame, window_size, gentle
    """
    if cv2 is None:
        raise RuntimeError(f"OpenCV is not available: {CV2_ERROR}")
    corners = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if corners.shape[0] < 1:
        raise RuntimeError("No markers were given to track.")

    blend = GENTLE_ANCHOR_BLEND if gentle else ANCHOR_BLEND
    max_correction = (GENTLE_ANCHOR_MAX_CORR_PX if gentle
                      else ANCHOR_MAX_CORR_PX)

    capture = RobustCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError("Could not open that video for tracking.")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = float(fps) if fps and fps > 0 else DEFAULT_FPS
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = int(clamp(start_frame, 0, max(0, total - 1)))
        if end_frame is None:
            end_frame = total - 1
        end_frame = int(clamp(end_frame, start_frame, max(0, total - 1)))

        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, first = capture.read()
        if not ok or first is None:
            raise RuntimeError(
                f"Could not read frame {start_frame} to start from.")
        gray0 = gray_for_lk(first, contrast)
        height, width = gray0.shape[:2]

        window_size = ensure_odd(window_size)
        patch_radius = max(10, int(round(window_size * PATCH_RADIUS_FRAC)))
        seeds, owners, offsets = _seed_patches(
            gray0, corners, patch_radius, FEATURES_PER_CORNER)
        if seeds.shape[0] == 0:
            raise RuntimeError(
                "None of the markers landed inside the frame.")

        lk = dict(
            winSize=(window_size, window_size), maxLevel=int(MAX_LEVEL),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                      int(CRITERIA_ITERS), float(CRITERIA_EPS)),
            flags=0, minEigThreshold=float(MIN_EIG_THRESH))

        # First-frame templates, for the drift anchor and for recovery.
        size = ensure_odd(ANCHOR_TEMPLATE_SIZE)
        half = size // 2
        templates = []
        for index in range(corners.shape[0]):
            cx = int(round(float(corners[index, 0])))
            cy = int(round(float(corners[index, 1])))
            x0, y0 = cx - half, cy - half
            if x0 >= 0 and y0 >= 0 and x0 + size <= width and y0 + size <= height:
                templates.append(gray0[y0:y0 + size, x0:x0 + size].copy())
            else:
                templates.append(None)

        count = corners.shape[0]
        last_good = corners.astype(np.float64).copy()

        def reduce_to_corners(sub_xy, alive):
            """Each corner, as the median of its surviving sub-features."""
            out = np.full((count, 2), np.nan, dtype=np.float32)
            for index in range(count):
                mine = (owners == index) & alive
                if np.any(mine):
                    predicted = sub_xy[mine] - offsets[mine]
                    out[index, 0] = float(np.median(predicted[:, 0]))
                    out[index, 1] = float(np.median(predicted[:, 1]))
            return out

        frames = [start_frame]
        first_row = reduce_to_corners(
            seeds.reshape(-1, 2), np.ones(seeds.shape[0], dtype=bool))
        tracked = [first_row]
        status = [(~np.isnan(first_row[:, 0])).astype(np.uint8)]

        previous_gray = gray0
        previous_sub = seeds.copy()
        span = max(1, end_frame - start_frame)

        for index in range(start_frame + 1, end_frame + 1):
            if progress is not None and progress(index - start_frame, span):
                break
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            gray = gray_for_lk(frame, contrast)

            forward, ok_f, _err = cv2.calcOpticalFlowPyrLK(
                previous_gray, gray, previous_sub, None, **lk)
            backward, ok_b, _err = cv2.calcOpticalFlowPyrLK(
                gray, previous_gray, forward, None, **lk)
            drift = np.linalg.norm(
                previous_sub.reshape(-1, 2) - backward.reshape(-1, 2), axis=1)
            good = ((ok_f.reshape(-1) == 1) & (ok_b.reshape(-1) == 1)
                    & (drift <= float(fb_threshold)))
            sub_xy = forward.reshape(-1, 2)
            corner_xy = reduce_to_corners(sub_xy, good)
            # Taken HERE, before the anchor is allowed to fill anything in:
            # this is what Lucas-Kanade itself managed to follow. A corner
            # the anchor later re-finds by template match still has a
            # position, and a good one, but calling that "measured" would
            # let a marker hidden under the probe report as tracked.
            measured = ~np.isnan(corner_xy[:, 0])

            snapped = np.zeros(count, dtype=bool)
            if anchor:
                for k in range(count):
                    template = templates[k]
                    if template is None:
                        continue
                    cx, cy = float(corner_xy[k, 0]), float(corner_xy[k, 1])
                    near = None
                    if np.isfinite(cx) and np.isfinite(cy):
                        near = _ncc_match(gray, template, cx, cy,
                                          ANCHOR_SEARCH_PAD)
                    if near is not None and near[2] >= ANCHOR_CONF:
                        apart = float(np.hypot(near[0] - cx, near[1] - cy))
                        if apart <= max_correction:
                            # They agree: blend, which is what removes the
                            # slow drift and the sub-pixel jitter.
                            corner_xy[k, 0] = ((1.0 - blend) * cx
                                               + blend * near[0])
                            corner_xy[k, 1] = ((1.0 - blend) * cy
                                               + blend * near[1])
                        continue
                    if (near is not None and near[2] >= ANCHOR_SUSPECT_CONF
                            and np.isfinite(cx)):
                        # A middling match: not agreement, not evidence the
                        # tracker is wrong either. Leave LK alone.
                        continue
                    # Either the marker was lost outright, or the template
                    # is nowhere near where LK says it is. Both mean the
                    # same thing: go looking for it, over a wide window
                    # around where it was last genuinely seen.
                    bx, by = last_good[k]
                    wide = _ncc_match(gray, template, float(bx), float(by),
                                      ANCHOR_RECOVER_PAD)
                    if wide is not None and wide[2] >= ANCHOR_RECOVER_CONF:
                        corner_xy[k, 0] = wide[0]
                        corner_xy[k, 1] = wide[1]
                        snapped[k] = True
                for k in range(count):
                    if np.isfinite(corner_xy[k, 0]):
                        last_good[k] = corner_xy[k]

            frames.append(index)
            tracked.append(corner_xy)
            status.append(measured.astype(np.uint8))

            # Re-seed the lost sub-features onto the corner's new position,
            # so a patch that recovers does not carry stale members.
            carried = previous_sub.reshape(-1, 2).copy()
            carried[good] = sub_xy[good]
            for k in range(count):
                if np.isfinite(corner_xy[k, 0]):
                    # A snapped corner takes its WHOLE patch with it. Moving
                    # only the sub-features LK gave up on would leave the
                    # ones it was confident about sitting on the wrong
                    # feature, and they would drag the corner back there on
                    # the next frame.
                    move = ((owners == k) if snapped[k]
                            else ((owners == k) & (~good)))
                    if np.any(move):
                        carried[move, 0] = corner_xy[k, 0] + offsets[move, 0]
                        carried[move, 1] = corner_xy[k, 1] + offsets[move, 1]
            previous_sub = carried.reshape(-1, 1, 2).astype(np.float32)
            previous_gray = gray

        return {
            "frames": np.asarray(frames, dtype=int),
            "points": np.asarray(tracked, dtype=np.float32),
            "status": np.asarray(status, dtype=np.uint8),
            "fps": float(fps),
            "start_frame": int(start_frame),
            "window_size": int(window_size),
            "gentle": bool(gentle),
        }
    finally:
        capture.release()


# ------------------------------------------- v1.9.4: reading past damage ---

# Frame rates re-measured from the frames' time stamps when the file header
# is wrong (path -> fps). RobustCapture reports these; the automatic
# analysis fills it in.
FPS_FIX = {}


class RobustCapture:
    """
    cv2.VideoCapture that does not stop at a damaged spot.

    OpenCV's read() returns False at the first frame it cannot decode and
    then every following read fails too, so anything that reads the video
    in order stopped there, often halfway. This wrapper finds the next
    readable frame by re-opening the file and seeking, and while it skips
    the unreadable frames it hands back the last good picture so frame
    numbers stay aligned. The skipped frame numbers are listed in
    ``bad_frames`` (and ``last_was_bad`` tells about the frame just read),
    so measurements on them can be left out. Everything else behaves like
    cv2.VideoCapture.
    """

    def __init__(self, path):
        self.path = str(path)
        self._cap = cv2.VideoCapture(self.path)
        try:
            self._n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        except Exception:
            self._n, self._fps = 0, 0.0
        if not (self._fps > 0):
            self._fps = float(DEFAULT_FPS)
        self._pos = 0
        self._last = None
        self._skip_to = None
        self._broken = False
        self._after_seek = True
        self.bad_frames = []
        self.unreadable_from = None
        self.last_was_bad = False

    def __getattr__(self, name):
        return getattr(self._cap, name)

    def _reopen(self, frame):
        try:
            self._cap.release()
        except Exception:
            pass
        self._cap = cv2.VideoCapture(self.path)
        if frame > 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        self._broken = False

    def _readable(self, k):
        c = cv2.VideoCapture(self.path)
        try:
            if k > 0:
                c.set(cv2.CAP_PROP_POS_FRAMES, int(k))
            ok, f = c.read()
            return bool(ok) and f is not None
        finally:
            c.release()

    def _find_next(self, bad):
        """First readable frame after ``bad`` (None if there is none)."""
        last = self._n - 1
        if bad >= last:
            return None
        lo, step, hi = bad, 1, None
        while hi is None:
            k = min(last, lo + step)
            if self._readable(k):
                hi = k
            elif k >= last:
                return None
            else:
                lo, step = k, step * 2
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self._readable(mid):
                hi = mid
            else:
                lo = mid
        return hi

    def release(self):
        try:
            self._cap.release()
        except Exception:
            pass

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._pos = max(0, int(round(value)))
            self._skip_to = None
            self._after_seek = True
            if self._broken:
                self._reopen(self._pos)
                return True
        return self._cap.set(prop, value)

    def get(self, prop):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self._pos)
        if prop == cv2.CAP_PROP_FPS and self.path in FPS_FIX:
            return float(FPS_FIX[self.path])
        if prop == cv2.CAP_PROP_POS_MSEC and self.last_was_bad:
            return float("nan")
        return self._cap.get(prop)

    def grab(self):
        ok, _f = self.read()
        return ok

    def read(self, *args):
        self.last_was_bad = False
        if self._skip_to is not None:
            if self._pos < self._skip_to:
                self.bad_frames.append(self._pos)
                self._pos += 1
                self.last_was_bad = True
                return True, self._last.copy()
            self._reopen(self._skip_to)
            self._skip_to = None
        ok, f = self._cap.read()
        if ok and f is not None:
            self._pos += 1
            self._last = f
            self._after_seek = False
            return True, f
        self._broken = True
        if self._after_seek or self._last is None or self._pos >= self._n - 1:
            return False, None
        k = self._find_next(self._pos)
        if k is None:
            self.unreadable_from = self._pos
            return False, None
        self._skip_to = k
        self.bad_frames.append(self._pos)
        self._pos += 1
        self.last_was_bad = True
        return True, self._last.copy()


# ------------------------------- v1.7: the reference-anchored tracker ---
#
# Each marker owns ONE circular area of radius ``radius`` px. On every frame
# that area is found again by matching it, to sub-pixel precision and
# independently of brightness, against how it looked on the FIRST tracked
# frame. Matching against the first frame instead of the previous one is
# what stops the slow creep off the picked spot, so every contraction is
# measured from the same place. The v1.6 patch tracker above is kept as
# ``track_points`` for comparison.

TRACK_AREA_RADIUS = 15            # px; the ring drawn on the preview
TRACK_AREA_RADIUS_MIN = 6
TRACK_AREA_RADIUS_MAX = 40
REF_MATCH_SIGMA = 1.0             # px Gaussian pre-blur for the matcher
REF_LOCK_CONF = 0.55              # min correlation to stay anchored
REF_RELOCK_CONF = 0.75            # min correlation to re-anchor after a loss
REF_STRONG_CONF = 0.80            # at/above this a match is taken as certain
REF_SEARCH_PAD = 10               # px searched around the predicted position
REF_RECOVER_PAD = 40              # px, widest search while un-anchored
REF_GROUP_TOL_PX = 4.0            # px a point may differ from the others
REF_REREF_SECONDS = 1.5           # un-anchored this long -> adopt a new look
REF_MIN_CONTRAST = 4.0            # grey levels; below = "nothing to track"
REF_MIN_EIG_RATIO = 0.12          # below = straight edge (can slide along it)

# Lock states, per frame and marker, in result["lock"].
LOST, ESTIMATED, ANCHORED, REANCHORED = 0, 1, 2, 3
LOCK_NAMES = {LOST: "lost", ESTIMATED: "estimated",
              ANCHORED: "anchored", REANCHORED: "re-anchored"}


class RefPatch:
    """
    One circular tracking area, remembered as it looked on the reference
    frame, plus a sub-pixel matcher for finding it again later.

    The matcher is the inverse-compositional Gauss-Newton scheme used in
    digital image correlation, with a zero-mean NORMALISED criterion, so a
    change of brightness or contrast (flicker, the cell darkening as it
    thickens) does not move the answer. Only pixels inside the circle take
    part (soft 2-px rim), so the circle drawn on the preview is the truth.
    """

    def __init__(self, img, cx, cy, radius, taper=2.0):
        self.r = int(max(2, radius))
        r = self.r
        ys, xs = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
        dist = np.hypot(xs, ys)
        w = np.clip((r + 0.5 - dist) / float(taper), 0.0, 1.0)
        self.w = w.astype(np.float64)
        self.wsum = float(self.w.sum())
        self.cx, self.cy = float(cx), float(cy)

        f = self._grab(img, self.cx, self.cy)
        fx = 0.5 * (self._grab(img, self.cx + 1.0, self.cy)
                    - self._grab(img, self.cx - 1.0, self.cy))
        fy = 0.5 * (self._grab(img, self.cx, self.cy + 1.0)
                    - self._grab(img, self.cx, self.cy - 1.0))
        self.square = f.astype(np.float32)
        fm = float((self.w * f).sum() / self.wsum)
        fz = f - fm
        fn = float(np.sqrt((self.w * fz * fz).sum()))
        self.contrast = fn / math.sqrt(self.wsum) if self.wsum > 0 else 0.0
        self.eig_ratio = 0.0
        self.Hinv = None
        self.ok = fn > 1e-6
        if not self.ok:
            return
        self.ft = fz / fn
        self.jx = fx / fn
        self.jy = fy / fn
        hxx = float((self.w * self.jx * self.jx).sum())
        hxy = float((self.w * self.jx * self.jy).sum())
        hyy = float((self.w * self.jy * self.jy).sum())
        H = np.array([[hxx, hxy], [hxy, hyy]], dtype=np.float64)
        ev = np.linalg.eigvalsh(H)
        self.eig_ratio = float(ev[0] / ev[1]) if ev[1] > 0 else 0.0
        self.ok = bool(ev[0] > 1e-9)
        if self.ok:
            self.Hinv = np.linalg.inv(H)

    def _grab(self, img, x, y):
        """
        The (2r+1)² patch centred on sub-pixel (x, y), by Catmull-Rom cubic
        interpolation done in NumPy. cv2.remap is deliberately not used: it
        rounds sampling positions to 1/32 px, which stops the sub-pixel
        iteration settling, and its a=-0.75 kernel adds a ripple that
        repeats every pixel of travel; both would end up in the speeds.
        """
        r = self.r
        S = 2 * r + 1
        if not (np.isfinite(x) and np.isfinite(y)):
            return np.zeros((S, S), dtype=np.float64)
        H, W = img.shape[:2]
        x = float(clamp(x, -4.0 * S, W + 4.0 * S))
        y = float(clamp(y, -4.0 * S, H + 4.0 * S))
        ix, iy = int(math.floor(x)), int(math.floor(y))
        tx, ty = x - ix, y - iy
        rows = np.clip(np.arange(iy - r - 1, iy + r + 3), 0, H - 1)
        cols = np.clip(np.arange(ix - r - 1, ix + r + 3), 0, W - 1)
        blk = img[np.ix_(rows, cols)].astype(np.float64)

        def taps(t):
            t2, t3 = t * t, t * t * t
            return (0.5 * (-t3 + 2.0 * t2 - t),
                    0.5 * (3.0 * t3 - 5.0 * t2 + 2.0),
                    0.5 * (-3.0 * t3 + 4.0 * t2 + t),
                    0.5 * (t3 - t2))

        a0, a1, a2, a3 = taps(tx)
        h = (a0 * blk[:, 0:S] + a1 * blk[:, 1:S + 1]
             + a2 * blk[:, 2:S + 2] + a3 * blk[:, 3:S + 3])
        b0, b1, b2, b3 = taps(ty)
        return (b0 * h[0:S] + b1 * h[1:S + 1]
                + b2 * h[2:S + 2] + b3 * h[3:S + 3])

    def score(self, img, x, y):
        """Normalised correlation (-1..1) of the area at (x, y)."""
        if not self.ok:
            return -1.0
        g = self._grab(img, x, y)
        gz = g - (self.w * g).sum() / self.wsum
        gn = math.sqrt(float((self.w * gz * gz).sum()))
        if gn < 1e-6:
            return -1.0
        return float((self.w * self.ft * (gz / gn)).sum())

    def refine(self, img, x, y, iters=30, eps=0.003, max_move=None):
        """Sub-pixel refinement from (x, y) -> (x, y, corr, converged)."""
        if not self.ok:
            return (float(x), float(y), -1.0, False)
        x0, y0 = float(x), float(y)
        x, y = x0, y0
        limit = float(max_move if max_move is not None else self.r)
        w = self.w
        step = float("inf")
        for _ in range(int(iters)):
            g = self._grab(img, x, y)
            gz = g - (w * g).sum() / self.wsum
            gn = math.sqrt(float((w * gz * gz).sum()))
            if gn < 1e-6:
                return (x, y, -1.0, False)
            e = self.ft - gz / gn
            b = np.array([(w * self.jx * e).sum(), (w * self.jy * e).sum()])
            dp = self.Hinv @ b
            step = float(np.hypot(dp[0], dp[1]))
            if step > 2.0:
                dp = dp * (2.0 / step)
            x += float(dp[0])
            y += float(dp[1])
            if np.hypot(x - x0, y - y0) > limit:
                return (x, y, self.score(img, x, y), False)
            if step < eps:
                break
        return (x, y, self.score(img, x, y), bool(step < 0.05))

    def search(self, img, cx, cy, pad):
        """Whole-pixel search within ±pad of (cx, cy) -> (x, y, corr)|None."""
        r = self.r
        H, W = img.shape[:2]
        pad = int(max(1, pad))
        ix, iy = int(round(cx)), int(round(cy))
        x0, y0 = max(0, ix - r - pad), max(0, iy - r - pad)
        x1, y1 = min(W, ix + r + pad + 1), min(H, iy + r + pad + 1)
        if x1 - x0 < 2 * r + 2 or y1 - y0 < 2 * r + 2:
            return None
        try:
            res = cv2.matchTemplate(img[y0:y1, x0:x1], self.square,
                                    cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            return None
        res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
        _, maxv, _, maxloc = cv2.minMaxLoc(res)
        return (float(x0 + maxloc[0] + r), float(y0 + maxloc[1] + r),
                float(maxv))


def group_predict(src, dst, p, rigid=False):
    """
    If the points ``src`` moved to ``dst``, where is ``p`` now?

    rigid=False: the cell stretched, shortened, turned or shifted as one
    piece (affine); needs three points not on one line. rigid=True: shift,
    turn and uniform scale only; needs two. Returns None when it cannot be
    determined.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    n = src.shape[0]
    if n < 2 or not (np.all(np.isfinite(src)) and np.all(np.isfinite(dst))):
        return None
    c = src.mean(axis=0)
    X = src - c
    if rigid:
        d = dst.mean(axis=0)
        a = X[:, 0] + 1j * X[:, 1]
        b = (dst[:, 0] - d[0]) + 1j * (dst[:, 1] - d[1])
        den = float(np.sum(np.abs(a) ** 2))
        if den < 1e-9:
            return None
        z = np.sum(np.conj(a) * b) / den
        q = z * complex(float(p[0]) - c[0], float(p[1]) - c[1])
        return np.array([d[0] + q.real, d[1] + q.imag])
    if n < 3:
        return None
    sv = np.linalg.svd(X, compute_uv=False)
    if sv[0] < 1e-6 or sv[-1] / sv[0] < 0.08:
        return None
    A = np.hstack([X, np.ones((n, 1))])
    M, *_ = np.linalg.lstsq(A, dst, rcond=None)
    return np.array([float(p[0]) - c[0], float(p[1]) - c[1], 1.0]) @ M


def track_points_ref(video_path, points_xy, start_frame=0, end_frame=None,
                     radius=TRACK_AREA_RADIUS, fb_threshold=FB_THRESHOLD,
                     contrast=1.0, progress=None):
    """
    The v1.7 four-point tracker: every marker anchored to its own area.

    While a marker is anchored (state 2) it is always compared with the
    reference, so errors do not add up frame to frame: after each
    contraction it is back on the spot that was picked. When its own area
    cannot be found (under the probe, badly blurred) it is carried by the
    stretch and shift of the other markers if three are anchored, else by
    frame-to-frame optical flow (state 1, "estimated"). When it is seen
    again it re-anchors, and the error collected meanwhile is spread evenly
    over the gap, so no jump and no speed spike appears. If the area has
    not been recognisable for REF_REREF_SECONDS while only flow carried it,
    its present look becomes the new reference (state 3): stable again from
    there, but any error built up until then stays.

    Returns the same dict as ``track_points`` plus:
        method      "ref"
        lock        (N,K) 2 anchored, 3 re-anchored, 1 estimated, 0 lost
        zncc        (N,K) the match score
        radius      the area radius used
        rerefs      [(frame, marker)] where a new reference was taken
        contrast, eig_ratio   how trackable each picked spot was
        bad_frames  frames the file would not give (last good one reused)
    """
    if cv2 is None:
        raise RuntimeError(f"OpenCV is not available: {CV2_ERROR}")
    P0 = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if P0.shape[0] < 1:
        raise RuntimeError("No markers were given to track.")

    cap = RobustCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open that video for tracking.")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps) if fps and fps > 0 else DEFAULT_FPS
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = int(clamp(start_frame, 0, max(0, total - 1)))
        if end_frame is None:
            end_frame = total - 1
        end_frame = int(clamp(end_frame, start_frame, max(0, total - 1)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, frame0 = cap.read()
        if not ok or frame0 is None:
            raise RuntimeError(
                f"Could not read frame {start_frame} to start from.")

        r = int(clamp(int(radius), 3, 200))
        sigma = float(REF_MATCH_SIGMA)
        lock_thr, relock_thr = float(REF_LOCK_CONF), float(REF_RELOCK_CONF)
        strong_thr = float(REF_STRONG_CONF)
        pad, recover_pad = int(REF_SEARCH_PAD), int(REF_RECOVER_PAD)
        group_tol = float(REF_GROUP_TOL_PX)
        reref_after = max(10, int(round(float(REF_REREF_SECONDS) * fps)))
        ANCH, ESTM, GONE = ANCHORED, ESTIMATED, LOST

        def match_image(bgr):
            # Plain grey, no CLAHE: the matcher is already independent of
            # brightness and contrast. A light blur tames sensor noise and
            # compression blocks.
            g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if sigma > 0:
                g = cv2.GaussianBlur(g, (0, 0), sigma)
            return g

        gray8 = gray_for_lk(frame0, contrast)
        gm = match_image(frame0)
        K = P0.shape[0]
        refs = [RefPatch(gm, P0[k, 0], P0[k, 1], r) for k in range(K)]
        first_refs = list(refs)

        lk_win = ensure_odd(2 * r + 1)
        lk_params = dict(
            winSize=(lk_win, lk_win), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                      30, 0.01))

        est = P0.copy()
        base = P0.copy()
        state = np.full(K, ANCH, dtype=np.uint8)
        gen = np.zeros(K, dtype=int)
        n_un = np.zeros(K, dtype=int)
        n_free = np.zeros(K, dtype=int)
        span_a = [None] * K
        delta = [None] * K
        pts = [P0.copy()]
        locks = [state.copy()]
        scores = [np.ones(K, dtype=np.float32)]
        frames = [start_frame]
        rerefs = []
        prev8 = gray8
        span = max(1, end_frame - start_frame)

        def try_lock(k, centre, spad, thr):
            ref = refs[k]
            best = None
            res = ref.refine(gm, centre[0], centre[1], max_move=spad + 1.5)
            if res[3]:
                best = res
            if best is None or best[2] < strong_thr:
                c = ref.search(gm, centre[0], centre[1], spad)
                if c is not None and c[2] >= 0.3:
                    r2 = ref.refine(gm, c[0], c[1], max_move=4.0)
                    if (r2[3] and (best is None or r2[2] > best[2])
                            and np.hypot(r2[0] - centre[0],
                                         r2[1] - centre[1]) <= spad + 1.5):
                        best = r2
            z = best[2] if best is not None else float("nan")
            if best is not None and best[2] >= thr:
                return np.array(best[:2], dtype=np.float64), z
            return None, z

        for frame_idx in range(start_frame + 1, end_frame + 1):
            if progress is not None and progress(frame_idx - start_frame,
                                                 span):
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            gray8 = gray_for_lk(frame, contrast)
            gm = match_image(frame)

            # Frame-to-frame flow: only a first guess of where to look, and
            # the fallback for a marker with nothing better.
            p_prev = est.astype(np.float32).reshape(-1, 1, 2)
            nxt, st_f, _ = cv2.calcOpticalFlowPyrLK(
                prev8, gray8, p_prev, None, **lk_params)
            bck, st_b, _ = cv2.calcOpticalFlowPyrLK(
                gray8, prev8, nxt, None, **lk_params)
            fb = np.linalg.norm(
                p_prev.reshape(-1, 2) - bck.reshape(-1, 2), axis=1)
            flow_ok = ((st_f.reshape(-1) == 1) & (st_b.reshape(-1) == 1)
                       & (fb <= float(fb_threshold)))
            nxt = nxt.reshape(-1, 2).astype(np.float64)

            out = np.full((K, 2), np.nan, dtype=np.float64)
            st = np.zeros(K, dtype=np.uint8)
            zz = np.full(K, np.nan, dtype=np.float32)
            idx = len(pts)
            prev = pts[idx - 1]

            def group_tolerance(A):
                d = out[A] - base[A]
                d = d - d.mean(axis=0)
                return max(group_tol,
                           0.5 * float(np.hypot(d[:, 0], d[:, 1]).max()))

            def expected_from(A, k):
                if len(A) >= 3:
                    p = group_predict(base[A], out[A], base[k])
                    if p is not None:
                        return p, group_tolerance(A)
                if len(A) >= 2:
                    p = group_predict(base[A], out[A], base[k], rigid=True)
                    if p is not None:
                        return p, max(2.0 * group_tol, group_tolerance(A))
                return None, 0.0

            def unanchored(k, anchored_mask):
                A = [j for j in range(K) if j != k and anchored_mask[j]]
                pred = (group_predict(base[A], out[A], base[k])
                        if len(A) >= 3 else None)
                if pred is not None:
                    if delta[k] is None:
                        d = np.zeros(2)
                        if (np.isfinite(prev[k, 0])
                                and np.all(np.isfinite(prev[A]))):
                            q = group_predict(base[A], prev[A], base[k])
                            if q is not None:
                                d = prev[k] - q
                        delta[k] = d
                    return pred + delta[k], False
                delta[k] = None
                if not flow_ok[k]:
                    return None, False
                return nxt[k].copy(), True

            # 1) anchored markers: is my own area still there?
            for k in range(K):
                if state[k] == ANCH:
                    p, zz[k] = try_lock(
                        k, nxt[k] if flow_ok[k] else est[k], pad, lock_thr)
                    if p is not None:
                        out[k] = p
                        st[k] = ANCH

            # 2) un-anchored markers: look for my own area again
            first_pass = st == ANCH
            regained = {}
            for k in range(K):
                if state[k] == ANCH:
                    continue
                u, _ = unanchored(k, first_pass)
                centre = u if u is not None else est[k]
                p, zz[k] = try_lock(
                    k, centre, min(recover_pad, pad + 2 * int(n_un[k])),
                    relock_thr)
                if p is not None:
                    out[k] = p
                    st[k] = ANCH
                    regained[k] = centre

            # 3) plausibility: a marker just re-found must sit where the
            #    markers that never lost their area say the cell is; if not
            #    it is a look-alike (the rim of the probe) and stays lost.
            steady = [j for j in range(K)
                      if st[j] == ANCH and j not in regained]
            for k in sorted(list(regained), key=lambda j: zz[j]):
                pred, tol = expected_from(steady, k)
                if pred is None or np.hypot(*(out[k] - pred)) <= tol:
                    continue
                p, z = try_lock(k, pred, group_tol, relock_thr)
                if p is not None:
                    out[k] = p
                    zz[k] = z
                else:
                    st[k] = GONE
                    out[k] = np.nan
                    regained.pop(k, None)
            #    An established match is never overridden by the group.
            #    Only if its weakest member disagrees with the rest AND a
            #    better match exists where they point, that one is taken.
            C = [int(j) for j in np.where(st == ANCH)[0]]
            if len(C) >= 4:
                wk = min(C, key=lambda j: zz[j])
                A = [j for j in C if j != wk]
                pred = group_predict(base[A], out[A], base[wk])
                if (pred is not None and np.hypot(
                        *(out[wk] - pred)) > group_tolerance(A)):
                    p, z = try_lock(wk, pred, group_tol,
                                    max(relock_thr, float(zz[wk]) + 0.05))
                    if p is not None:
                        out[wk] = p
                        zz[wk] = z

            # 4) settle every marker for this frame
            anchored_now = st == ANCH
            for k in range(K):
                if st[k] == ANCH:
                    if k in regained and span_a[k] is not None:
                        a = span_a[k]
                        closure = out[k] - regained[k]
                        for j in range(a + 1, idx):
                            if np.isfinite(pts[j][k, 0]):
                                pts[j][k] += ((j - a) / float(idx - a)) \
                                    * closure
                    n_un[k] = 0
                    n_free[k] = 0
                    span_a[k] = None
                    delta[k] = None
                    continue
                u, flow_only = unanchored(k, anchored_now)
                if state[k] == ANCH:
                    span_a[k] = idx - 1
                n_un[k] += 1
                if u is None:
                    st[k] = GONE
                    out[k] = np.nan
                    n_free[k] = 0
                    continue
                out[k] = u
                st[k] = ESTM
                if flow_only:
                    A = [j for j in range(K) if j != k and anchored_now[j]]
                    pred, tol = expected_from(A, k)
                    if pred is not None and np.hypot(*(u - pred)) > tol:
                        flow_only = False
                n_free[k] = n_free[k] + 1 if flow_only else 0

            # 5) the area looks different for good and nothing contradicts
            #    the flow: adopt its present look as the new reference.
            for k in range(K):
                if st[k] == ESTM and n_free[k] >= reref_after:
                    fresh = RefPatch(gm, out[k, 0], out[k, 1], r)
                    if fresh.ok:
                        refs[k] = fresh
                        gen[k] += 1
                        st[k] = ANCH
                        n_un[k] = 0
                        n_free[k] = 0
                        span_a[k] = None
                        delta[k] = None
                        rerefs.append((int(frame_idx), int(k)))
                        base = np.where(np.isfinite(out), out, est)

            for k in range(K):
                if np.isfinite(out[k, 0]):
                    est[k] = out[k]
                state[k] = st[k]
            frames.append(frame_idx)
            pts.append(out)
            rec = st.copy()
            rec[(st == ANCH) & (gen > 0)] = REANCHORED
            locks.append(rec)
            scores.append(zz)
            prev8 = gray8
        bad = list(cap.bad_frames)
    finally:
        cap.release()

    lock_arr = np.array(locks, dtype=np.uint8)
    return {
        "frames": np.asarray(frames, dtype=int),
        "points": np.array(pts, dtype=np.float32),
        # 1 where the marker was found on its own area this frame
        "status": ((lock_arr == ANCHORED) | (lock_arr == REANCHORED))
        .astype(np.uint8),
        "fps": float(fps),
        "start_frame": int(start_frame),
        "window_size": int(2 * r + 1),
        "gentle": False,
        "method": "ref",
        "lock": lock_arr,
        "zncc": np.array(scores, dtype=np.float32),
        "radius": int(r),
        "rerefs": rerefs,
        "contrast": [float(rf.contrast) for rf in first_refs],
        "eig_ratio": [float(rf.eig_ratio) for rf in first_refs],
        "bad_frames": bad,
    }


def track_points_corrected(video_path, anchors, end_frame=None,
                           radius=TRACK_AREA_RADIUS,
                           fb_threshold=FB_THRESHOLD, contrast=1.0,
                           progress=None):
    """
    The area-anchored tracker, restarted at every hand correction.

    ``anchors`` maps frame -> the (K, 2) positions someone placed there: the
    first is where tracking starts, every later one is a correction. Each
    stretch between two anchors is tracked by ``track_points_ref`` starting
    from the earlier anchor (so each marker's reference look is taken from
    that frame), and whatever the tracker is off by when it reaches the next
    anchor is spread evenly over the stretch, as the desktop app's anchored
    re-track does. The anchor frames themselves sit exactly where they were
    placed. After the last anchor, tracking simply runs on to ``end_frame``.

    Returns the same dict as ``track_points_ref`` plus ``anchors``, the
    anchor frame numbers.
    """
    if not anchors:
        raise RuntimeError("No starting positions were given.")
    keys = sorted(int(f) for f in anchors)
    placed = {int(f): np.asarray(anchors[f], dtype=np.float64).reshape(-1, 2)
              for f in anchors}
    first = keys[0]
    if end_frame is None:
        end_frame = keys[-1]
    end_frame = int(max(end_frame, keys[-1]))
    stops = [f for f in keys if f <= end_frame]
    bounds = list(zip(stops, stops[1:] + [end_frame]))
    if bounds and bounds[-1][0] == bounds[-1][1] and len(bounds) > 1:
        bounds = bounds[:-1]
    total = max(1, end_frame - first)
    parts = []
    for a, b in bounds:
        def tick(done, span, _a=a):
            if progress is None:
                return False
            return progress(int(_a - first + done), total)

        part = track_points_ref(video_path, placed[a], start_frame=a,
                                end_frame=b, radius=radius,
                                fb_threshold=fb_threshold,
                                contrast=contrast, progress=tick)
        pts = part["points"].astype(np.float64)
        frames = part["frames"]
        if b in placed and int(frames[-1]) == b and len(frames) > 1:
            # Close onto the correction, the error growing linearly from
            # nothing at the earlier anchor to all of it at this one.
            miss = placed[b] - pts[-1]
            weight = (frames - a) / float(b - a)
            for k in range(pts.shape[1]):
                if np.all(np.isfinite(miss[k])):
                    pts[:, k, :] += weight[:, None] * miss[k][None, :]
                else:
                    pts[-1, k, :] = placed[b][k]
            part["lock"][-1, :] = ANCHORED
        part["points"] = pts.astype(np.float32)
        parts.append(part)
        if int(frames[-1]) < b:
            break                   # the file stopped giving frames

    def joined(name):
        pieces = [np.asarray(p[name]) for p in parts]
        return np.concatenate([pieces[0]] + [x[1:] for x in pieces[1:]],
                              axis=0)

    out = dict(parts[0])
    for name in ("frames", "points", "status", "lock", "zncc"):
        out[name] = joined(name)
    out["status"] = ((out["lock"] == ANCHORED)
                     | (out["lock"] == REANCHORED)).astype(np.uint8)
    out["rerefs"] = [r for p in parts for r in p["rerefs"]]
    out["bad_frames"] = sorted({b for p in parts for b in p["bad_frames"]})
    out["start_frame"] = int(first)
    out["anchors"] = [f for f in stops]
    return out


def bridge_lost_points(points):
    """
    v1.7: fill ONLY the frames where a marker has no position.

    A lost marker is carried with the median frame-to-frame motion of the
    markers still tracked, and when it comes back the gap is closed onto
    the re-found position by spreading the mismatch evenly over the gap, so
    there is no jump on return and no speed spike. Tracked positions are
    never touched (the v1.6 reconciler also snapped tracked markers to the
    group, which flattened fast contractions). Returns (filled, mask).
    """
    src = np.asarray(points, dtype=np.float64)
    N, K, _ = src.shape
    out = src.copy()
    filled = np.zeros((N, K), dtype=bool)
    fin = np.isfinite(src[:, :, 0]) & np.isfinite(src[:, :, 1])
    step = np.zeros((N, 2), dtype=np.float64)
    for i in range(1, N):
        m = fin[i] & fin[i - 1]
        if m.any():
            step[i] = np.median(src[i, m] - src[i - 1, m], axis=0)
    cum = np.cumsum(step, axis=0)
    for k in range(K):
        idx = np.where(fin[:, k])[0]
        if idx.size == 0:
            continue
        first, last = int(idx[0]), int(idx[-1])
        if first > 0:
            out[:first, k] = src[first, k]
            filled[:first, k] = True
        for a, b in zip(idx[:-1], idx[1:]):
            a, b = int(a), int(b)
            if b - a <= 1:
                continue
            dead = src[a, k] + (cum[a + 1:b + 1] - cum[a])
            closure = src[b, k] - dead[-1]
            for n, j in enumerate(range(a + 1, b)):
                out[j, k] = dead[n] + ((j - a) / float(b - a)) * closure
                filled[j, k] = True
        if last < N - 1:
            out[last + 1:, k] = src[last, k] + (cum[last + 1:] - cum[last])
            filled[last + 1:, k] = True
    return out.astype(np.float32), filled


def quality_report(result):
    """
    v1.7 honesty report for the reference-anchored tracker.

    Returns (warnings, summary): how much of the time each marker was
    locked on its own area, and anything about the picked spots that makes
    them hard to track. Both empty for a v1.6 result.
    """
    if not result or result.get("method") != "ref":
        return [], ""
    lock = np.asarray(result.get("lock"))
    if lock.ndim != 2 or lock.shape[0] == 0:
        return [], ""
    K = lock.shape[1]
    rad = int(result.get("radius", 0))
    labels = list(POINT_LABELS) + [f"P{i + 1}" for i in range(len(POINT_LABELS), K)]
    pct = [100.0 * float(np.mean(lock[:, k] == ANCHORED)) for k in range(K)]
    if min(pct) >= 99.95:
        summary = (f"All {K} markers locked on their own {rad}-px areas on "
                   "every frame.")
    else:
        summary = ("Locked on own area: " + " · ".join(
            f"{labels[k]} {pct[k]:.0f}%" for k in range(K))
            + f" (r = {rad} px).")
    warn = []
    con = list(result.get("contrast") or [])
    eig = list(result.get("eig_ratio") or [])
    for k in range(K):
        if k < len(con) and con[k] < float(REF_MIN_CONTRAST):
            warn.append(f"{labels[k]}: almost nothing to see inside its "
                        "ring. Move it onto an edge or corner, or enlarge "
                        "the ring.")
        elif k < len(eig) and eig[k] < float(REF_MIN_EIG_RATIO):
            warn.append(f"{labels[k]} sits on a straight edge and can slide "
                        "along it. A corner or spot is safer.")
    rr = result.get("rerefs") or []
    if rr:
        first = {}
        for f, k in rr:
            first.setdefault(int(k), int(f))
        warn.append(
            "Area no longer recognisable, new reference taken: "
            + ", ".join(f"{labels[k]} from frame {first[k]}"
                        for k in sorted(first))
            + ". Positions after that may be offset; check them.")
    bad = result.get("bad_frames") or []
    if bad:
        warn.append(f"{len(bad)} frame(s) could not be decoded and repeat "
                    "the frame before them.")
    return warn, summary


# ------------------------------------------------- keeping the cell whole ---

def hold_invalid_corners(points):
    """A lost marker stays where it was last seen. Returns (held, mask)."""
    points = np.asarray(points, dtype=np.float64)
    n, k = points.shape[0], points.shape[1]
    held = points.copy()
    was_held = np.zeros((n, k), dtype=bool)
    last = [None] * k
    for i in range(n):
        for j in range(k):
            x, y = held[i, j]
            if np.isfinite(x) and np.isfinite(y):
                last[j] = (float(x), float(y))
            elif last[j] is not None:
                held[i, j] = last[j]
                was_held[i, j] = True
    return held.astype(np.float32), was_held


def _frame_transform(before, after):
    """
    How the tracked markers moved between two frames, as scale + shift.

    Fits ``q = s·p + t`` by least squares over the markers finite in both:
    one uniform scale and one translation, which is what a cell that drifts
    across the field while contracting about its own centre actually does.

    The median motion, which is the obvious thing to use instead, is not
    that. With four markers round a contracting cell, two move one way and
    two the other, and the median of three of them is whichever side has
    two -- so a marker filled in from the median inherits half the
    contraction as if it were drift. Returns (s, tx, ty, how many markers).
    """
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    usable = (np.isfinite(before[:, 0]) & np.isfinite(before[:, 1])
              & np.isfinite(after[:, 0]) & np.isfinite(after[:, 1]))
    n = int(usable.sum())
    if n == 0:
        return 1.0, 0.0, 0.0, 0
    p, q = before[usable], after[usable]
    if n < 3:
        # Too few to tell a scale from a shift. Take the shift alone: the
        # mean is the better estimate of it, and with one or two markers
        # there is nothing to be robust against anyway.
        shift = np.mean(q - p, axis=0)
        return 1.0, float(shift[0]), float(shift[1]), n
    p_mean, q_mean = p.mean(axis=0), q.mean(axis=0)
    pc, qc = p - p_mean, q - q_mean
    spread = float(np.sum(pc * pc))
    scale = float(np.sum(pc * qc) / spread) if spread > 1e-9 else 1.0
    # A frame cannot really rescale the cell by a quarter. A fit that says
    # so is a fit driven by a marker that jumped, and clamping it keeps the
    # damage to that marker instead of spreading it to the others.
    scale = float(clamp(scale, 0.8, 1.25))
    shift = q_mean - scale * p_mean
    return scale, float(shift[0]), float(shift[1]), n


def reconcile_group_motion(points, min_dev=GROUP_MIN_DEV,
                           dev_frac=GROUP_DEV_FRAC):
    """
    Keep the markers moving as one cell. Returns (fixed, mask).

    Each frame, work out how the markers that WERE tracked moved -- one
    scale and one shift, fitted to all of them -- and then

      (a) move any lost marker by that same transform, so its displacement
          goes on counting as cell motion rather than freezing, and

      (b) pull back any marker whose own motion grossly disagrees with it,
          which is what a marker stuck to the probe or jumped to a decoy
          looks like.

    The tolerance scales with how far the transform says the marker should
    have moved, so real contraction is left alone and only a marker that
    has come off the cell is corrected.
    """
    points = np.asarray(points, dtype=np.float64)
    n, k = points.shape[0], points.shape[1]
    out = points.copy()
    fixed = np.zeros((n, k), dtype=bool)
    last = (1.0, 0.0, 0.0)
    for i in range(1, n):
        previous, current = out[i - 1], out[i]
        scale, tx, ty, seen = _frame_transform(previous, current)
        if seen == 0:
            scale, tx, ty = last
        else:
            last = (scale, tx, ty)
        for j in range(k):
            if not (np.isfinite(previous[j, 0]) and np.isfinite(previous[j, 1])):
                continue
            px = scale * previous[j, 0] + tx
            py = scale * previous[j, 1] + ty
            moved = float(np.hypot(px - previous[j, 0], py - previous[j, 1]))
            tolerance = max(float(min_dev), float(dev_frac) * moved)
            if not (np.isfinite(current[j, 0]) and np.isfinite(current[j, 1])):
                out[i, j, 0], out[i, j, 1] = px, py
                fixed[i, j] = True
            elif np.hypot(current[j, 0] - px, current[j, 1] - py) > tolerance:
                out[i, j, 0], out[i, j, 1] = px, py
                fixed[i, j] = True
    return out.astype(np.float32), fixed


def settle(result, follow_under_probe=True, gentle=None):
    """
    The tracked positions, made into one moving cell.

    ``follow_under_probe`` reconciles the markers against the group's own
    motion; without it a lost marker simply holds its last position, which
    reads as the cell stopping when what stopped was the tracking.
    """
    points = result["points"]
    if gentle is None:
        gentle = bool(result.get("gentle"))
    if follow_under_probe and result.get("method") == "ref":
        # v1.7: only frames with NO position are filled. Tracked positions
        # are never snapped to the group: that flattened fast contractions.
        return bridge_lost_points(points)
    if follow_under_probe:
        if gentle:
            return reconcile_group_motion(
                points, min_dev=GENTLE_GROUP_MIN_DEV,
                dev_frac=GENTLE_GROUP_DEV_FRAC)
        return reconcile_group_motion(points)
    return hold_invalid_corners(points)


# -------------------------------------------------------------- readings ---

def speeds(points, fps, smooth=True, gentle=False):
    """
    Speed of every marker, in pixels per second. Shape (N, K).

    Position is bridged across short gaps and smoothed, differentiated by
    Savitzky-Golay rather than by difference (a difference of noisy
    positions is mostly noise), and the resulting speed smoothed again.
    """
    points = np.asarray(points, dtype=np.float64)
    n, k = points.shape[0], points.shape[1]
    out = np.full((n, k), np.nan, dtype=np.float64)
    if n < 2:
        return np.nan_to_num(out)
    dt = 1.0 / (float(fps) if fps else DEFAULT_FPS)
    position_window = (GENTLE_POSITION_SMOOTH_WINDOW if gentle
                       else POSITION_SMOOTH_WINDOW)
    for j in range(k):
        x, _ = interpolate_nan_1d(points[:, j, 0])
        y, _ = interpolate_nan_1d(points[:, j, 1])
        if smooth:
            if position_window >= 3:
                x = smooth_series(x, position_window)
                y = smooth_series(y, position_window)
            dx = smooth_derivative(x, VELOCITY_SMOOTH_WINDOW,
                                   SMOOTH_POLYORDER, dt)
            dy = smooth_derivative(y, VELOCITY_SMOOTH_WINDOW,
                                   SMOOTH_POLYORDER, dt)
            speed = np.hypot(dx, dy)
            if SPEED_SMOOTH_WINDOW >= 3:
                speed = smooth_series(speed, SPEED_SMOOTH_WINDOW)
            speed = np.clip(speed, 0.0, None)
        else:
            speed = np.hypot(plain_derivative(x, dt), plain_derivative(y, dt))
        out[:, j] = speed
    return out


def centroid(points):
    """The cell's centre, as the mean of whichever markers are finite."""
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    out = np.full((n, 2), np.nan, dtype=np.float64)
    for i in range(n):
        row = points[i]
        finite = np.isfinite(row[:, 0]) & np.isfinite(row[:, 1])
        if finite.any():
            out[i, 0] = float(np.mean(row[finite, 0]))
            out[i, 1] = float(np.mean(row[finite, 1]))
    return out


def displacement(series):
    """Distance from where it started, for a (N,2) track."""
    series = np.asarray(series, dtype=np.float64)
    if series.shape[0] == 0:
        return np.zeros(0)
    return np.hypot(series[:, 0] - series[0, 0], series[:, 1] - series[0, 1])


def quad_area(points):
    """
    The area of the quadrilateral the four markers make, in px².

    The shoelace formula in the order they were picked, so it is only a
    cell area if they were picked round the cell rather than across it. NaN
    on any frame where a marker is missing.
    """
    points = np.asarray(points, dtype=np.float64)
    n, k = points.shape[0], points.shape[1]
    out = np.full(n, np.nan, dtype=np.float64)
    if k < 3:
        return out
    for i in range(n):
        row = points[i]
        if not np.all(np.isfinite(row)):
            continue
        total = 0.0
        for j in range(k):
            nxt = (j + 1) % k
            total += row[j, 0] * row[nxt, 1] - row[nxt, 0] * row[j, 1]
        out[i] = abs(total) * 0.5
    return out


# ------------------------------------------------------------- clipping ---
#
# The desktop app's clipper: cut the stretch of the video that matters out
# of the recording, by frame or by time, as its own file. FFmpeg does it
# properly -- frame-accurate, sound kept, H.264 that every browser plays --
# and OpenCV does it when FFmpeg is not there, video only, so the button
# never simply fails.

import os as _os
import re as _re
import shutil as _shutil
import subprocess as _subprocess


def ffmpeg_path():
    """Where FFmpeg is, or None. The system one first, then imageio's."""
    found = _shutil.which("ffmpeg")
    if found:
        return found
    try:  # pragma: no cover - only where imageio-ffmpeg is installed
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def parse_time(text):
    """
    '1:02.5', '00:01:02.500' or '62.5' as seconds; None when it is not one.

    The same three forms the desktop clipper takes, so a time copied out of
    a video player's position box can be pasted straight in.
    """
    if text is None:
        return None
    value = str(text).strip()
    if not value:
        return None
    if ":" not in value:
        try:
            number = float(value)
        except ValueError:
            return None
        return number if number >= 0 else None
    parts = value.split(":")
    if len(parts) not in (2, 3):
        return None
    try:
        numbers = [float(p) for p in parts]
    except ValueError:
        return None
    if any(n < 0 for n in numbers):
        return None
    if len(numbers) == 2:
        return numbers[0] * 60.0 + numbers[1]
    return numbers[0] * 3600.0 + numbers[1] * 60.0 + numbers[2]


def clip_name(video_path, t0, t1, ext=".mp4"):
    """video_clip_00m01s250_to_00m04s000.mp4 -- the desktop app's naming."""
    stem = _os.path.splitext(_os.path.basename(str(video_path)))[0]
    stem = _re.sub(r"^afm_video_", "", stem)

    def tag(seconds):
        seconds = max(0.0, float(seconds))
        minutes = int(seconds // 60)
        whole = int(seconds % 60)
        millis = int(round((seconds - int(seconds)) * 1000)) % 1000
        return f"{minutes:02d}m{whole:02d}s{millis:03d}"

    return f"{stem}_clip_{tag(t0)}_to_{tag(t1)}{ext}"


def clip_video(video_path, out_path, start_s, end_s, accurate=True,
               progress=None):
    """
    Write ``start_s``..``end_s`` of the video to ``out_path``.

    With FFmpeg: ``accurate`` re-encodes (H.264, sound kept, starts exactly
    on the frame asked for); otherwise the streams are copied, which is
    instant and lossless but can only start on a keyframe, so the clip may
    begin a little early. Without FFmpeg the frames are copied through
    OpenCV: frame-exact, no sound.

    Returns {"path", "method", "frames" (OpenCV only), "note"}.
    """
    start_s = max(0.0, float(start_s))
    end_s = max(start_s, float(end_s))
    duration = end_s - start_s
    if duration <= 0:
        raise ValueError("The clip ends where it starts.")

    exe = ffmpeg_path()
    if exe:
        if accurate:
            command = [exe, "-y", "-i", str(video_path),
                       "-ss", f"{start_s:.6f}", "-t", f"{duration:.6f}",
                       "-map", "0:v:0?", "-map", "0:a:0?",
                       "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                       "-pix_fmt", "yuv420p",
                       "-c:a", "aac", "-b:a", "192k",
                       "-movflags", "+faststart", str(out_path)]
        else:
            command = [exe, "-y", "-ss", f"{start_s:.6f}",
                       "-t", f"{duration:.6f}", "-i", str(video_path),
                       "-map", "0:v:0?", "-map", "0:a:0?", "-c", "copy",
                       str(out_path)]
        done = _subprocess.run(command, capture_output=True, text=True,
                               timeout=900)
        if done.returncode == 0 and _os.path.exists(out_path) \
                and _os.path.getsize(out_path) > 0:
            return {"path": str(out_path),
                    "method": "ffmpeg, re-encoded" if accurate
                              else "ffmpeg, streams copied",
                    "note": ("Frame-accurate, H.264, sound kept."
                             if accurate else
                             "Lossless and instant; starts on the nearest "
                             "keyframe, so it may begin a little early.")}
        # FFmpeg is there but refused this file. Say so, then fall through
        # to OpenCV rather than leave the person with nothing.
        failure = (done.stderr or "").strip().splitlines()[-1:] or [""]
        fallback_note = f"FFmpeg could not cut it ({failure[0][:160]}). "
    else:
        fallback_note = ""

    if cv2 is None:
        raise RuntimeError(f"Neither FFmpeg nor OpenCV is available: "
                           f"{CV2_ERROR}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Could not open that video to clip it.")
    writer = None
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = float(fps) if fps and fps > 0 else DEFAULT_FPS
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        first = int(clamp(round(start_s * fps), 0, max(0, total - 1)))
        # The end is exclusive, as it is for FFmpeg's -t: 1.0 to 2.0 s at
        # 30 fps is frames 30 to 59, thirty of them, whichever engine cuts it.
        last = int(clamp(round(end_s * fps) - 1, first, max(0, total - 1)))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(out_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                 (width, height))
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not open a video file to write.")
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        written = 0
        span = max(1, last - first + 1)
        for index in range(first, last + 1):
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
            written += 1
            if progress is not None and progress(written, span):
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if written == 0:
        raise RuntimeError("No frames could be read in that range.")
    return {"path": str(out_path), "method": "OpenCV", "frames": written,
            "note": fallback_note + "Frame-exact, video only (no sound), "
                    "MPEG-4 Part 2. It downloads and opens in VLC or "
                    "Fiji; some browsers will not preview it. Installing "
                    "FFmpeg gives H.264 with sound."}
