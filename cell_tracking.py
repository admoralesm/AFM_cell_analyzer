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

    capture = cv2.VideoCapture(str(video_path))
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
