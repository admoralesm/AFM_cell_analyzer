"""
Compression-video analysis: find the cell, measure it, line it up with the
force curve.

The job here is narrow. During an AFM whole-cell compression the camera sees
the cell squashed between the substrate and the cantilever. Two things are
worth extracting:

1. A picture of the cell at a chosen point on the force curve, so a modulus
   can be read next to the thing it was measured on.
2. The cell's height in pixels frame by frame, which gives a relative
   deformation derived from the video alone. Comparing that against the
   deformation axis of the force curve is an independent check on the contact
   point and the cell height, the two inputs the moduli are most sensitive to.

OpenCV is imported lazily so the rest of the app runs without it.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2

    CV2_ERROR = None
except Exception as exc:  # pragma: no cover - depends on local install
    cv2 = None
    CV2_ERROR = str(exc)


def available():
    """True when OpenCV is importable."""
    return cv2 is not None


def _require_cv2():
    if cv2 is None:
        raise RuntimeError(
            f"OpenCV is not available ({CV2_ERROR}). Add 'opencv-python-headless' "
            f"to requirements.txt."
        )


# ------------------------------------------------------------------ probing


def probe(path):
    """Frame count, fps, size and duration of a video file."""
    _require_cv2()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    # Some containers report a bogus frame count; fall back to counting.
    if n_frames <= 0:
        n_frames = _count_frames(path)

    return {
        "fps": fps,
        "n_frames": n_frames,
        "width": width,
        "height": height,
        "duration_s": n_frames / fps if fps > 0 else float("nan"),
    }


def _count_frames(path):
    cap = cv2.VideoCapture(str(path))
    count = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        count += 1
    cap.release()
    return count


def read_frame(path, index):
    """Return one frame as an RGB array, or None if it cannot be read."""
    _require_cv2()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(index)))
        ok, frame = cap.read()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


# ---------------------------------------------------------------- detection


def _strip_horizontal(mask, view_w, min_run_frac=0.55):
    """
    Delete pixels belonging to horizontal runs longer than the cell can be.

    The substrate line and the cantilever span most of the frame; a cell does
    not. Opening with a long, one-pixel-tall kernel keeps only those long runs,
    and subtracting them leaves the cell standing alone.
    """
    length = max(9, int(view_w * min_run_frac))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return cv2.subtract(mask, lines)


def _candidate_masks(gray, sensitivity, view_w, view_h, strip_lines):
    """
    Several segmentations of the same frame.

    No single rule works across brightfield, phase and DIC, so the cheap
    approach is to generate a few plausible masks and let the scoring below
    decide. Both intensity polarities are tried because a cell can read darker
    or brighter than its background.
    """
    masks = []
    k = max(3, (min(view_h, view_w) // 60) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # 1-2: Otsu, both polarities.
    for flag in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
        _, mask = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)
        masks.append(mask)

    # 3: local background subtraction. A cell only slightly darker than the
    # field is invisible to a global threshold but stands out clearly against
    # a heavily blurred copy of the same frame, which is what illumination
    # without the cell would look like.
    blurred = cv2.GaussianBlur(gray, (0, 0), max(view_w, view_h) / 12.0)
    for diff in (cv2.subtract(blurred, gray), cv2.subtract(gray, blurred)):
        if diff.max() > 3:
            _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            masks.append(mask)

    # 4: edge magnitude, which catches cells that match the background level
    # and are visible only by their outline.
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(sobel_x, sobel_y)
    if magnitude.max() > 0:
        magnitude = (255 * magnitude / magnitude.max()).astype(np.uint8)
        cutoff = float(np.clip(np.percentile(magnitude, 100 - 12 * sensitivity), 1, 254))
        _, edge_mask = cv2.threshold(magnitude, cutoff, 255, cv2.THRESH_BINARY)
        masks.append(edge_mask)

    processed = []
    for mask in masks:
        if strip_lines:
            mask = _strip_horizontal(mask, view_w)
        # Close the outline into a solid blob, drop leftover speckle, then
        # flood the interior so contourArea reflects the whole cell.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        filled = mask.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        processed.append(filled)
    return processed


def detect_cell(
    frame_rgb,
    roi=None,
    min_area_frac=0.01,
    max_area_frac=0.60,
    sensitivity=1.0,
    strip_lines=True,
    probe=None,
    cell_side="anywhere",
    reject_dark=True,
    dark_margin=0.35,
):
    """
    Find the cell in one frame.

    Candidates come from edge density rather than plain thresholding, because
    in brightfield the cell is often the same average brightness as its
    background and only its outline stands out. Each candidate is scored on
    size, circularity and how central it is, and the best one wins. The
    cantilever tends to lose on circularity, the image border on size.

    Parameters
    ----------
    frame_rgb : array
        Frame in RGB.
    roi : (x0, y0, x1, y1), optional
        Fractional crop (0-1) to search inside. Use it to exclude the
        cantilever or the edges of the field of view.
    min_area_frac, max_area_frac : float
        Acceptable blob area as a fraction of the searched region.
    sensitivity : float
        Scales the edge threshold. Raise it if the cell is faint, lower it if
        texture in the background is being picked up.
    strip_lines : bool
        Remove long horizontal structures before looking for blobs. The
        substrate and the cantilever are exactly that, and without this they
        merge with the cell into one wide shapeless region.
    probe : dict, optional
        Result of :func:`detect_probe`. When given together with ``cell_side``,
        candidates on the wrong side of the cantilever are discarded.
    cell_side : {"anywhere", "right", "left", "above", "below"}
        Where the cell sits relative to the probe.
    reject_dark : bool
        Discard candidates darker than ``dark_margin`` times the frame's median
        brightness. The cantilever is nearly black, so this alone stops it
        being mistaken for the cell even when it happens to look round. The
        default margin is deliberately low: a cell is often legitimately darker
        than its background, and a stricter cut throws the cell away too.

    Returns
    -------
    dict
        ``found``; when found also ``bbox`` (x, y, w, h in full-frame pixels),
        ``center``, ``height_px``, ``width_px``, ``area_px``, ``circularity``,
        ``score``, ``ellipse`` and ``contour``.
    """
    _require_cv2()
    frame = np.asarray(frame_rgb)
    H, W = frame.shape[:2]

    if probe is None and (reject_dark or cell_side != "anywhere"):
        found_probe = detect_probe(frame)
        probe = found_probe if found_probe.get("found") else None

    if roi:
        x0 = int(np.clip(roi[0], 0, 1) * W)
        y0 = int(np.clip(roi[1], 0, 1) * H)
        x1 = int(np.clip(roi[2], 0, 1) * W)
        y1 = int(np.clip(roi[3], 0, 1) * H)
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        if x1 - x0 < 10 or y1 - y0 < 10:
            x0, y0, x1, y1 = 0, 0, W, H
    else:
        x0, y0, x1, y1 = 0, 0, W, H

    view = frame[y0:y1, x0:x1]
    if view.size == 0:
        return {"found": False, "reason": "empty region"}

    gray = cv2.cvtColor(view, cv2.COLOR_RGB2GRAY)

    # Paint the cantilever out before segmenting anything. Rejecting it after
    # the fact is too late: it is near black against a bright field, so it
    # dominates the Otsu split and sets the scale the gradient threshold is
    # measured against, and a faint cell then falls below both. Replacing it
    # with the background level removes it from the statistics entirely.
    if probe and probe.get("found"):
        px, py, pw, ph = probe["bbox"]
        px, py = px - x0, py - y0
        sx0, sy0 = max(0, px), max(0, py)
        sx1, sy1 = min(gray.shape[1], px + pw), min(gray.shape[0], py + ph)
        if sx1 > sx0 and sy1 > sy0:
            outside = np.ones(gray.shape, dtype=bool)
            outside[sy0:sy1, sx0:sx1] = False
            fill = float(np.median(gray[outside])) if outside.any() else 255.0
            gray = gray.copy()
            gray[sy0:sy1, sx0:sx1] = fill

    # Median first: it removes speckle without softening the cell outline,
    # which is what the edge detector below is looking for.
    gray = cv2.medianBlur(gray, 5)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    view_h, view_w = view.shape[:2]
    view_area = float(view_h * view_w)
    cx_view, cy_view = view_w / 2.0, view_h / 2.0
    diag = float(np.hypot(view_w, view_h))

    masks = _candidate_masks(gray, sensitivity, view_w, view_h, strip_lines)
    contours = []
    for mask in masks:
        found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(found)
    if not contours:
        return {"found": False, "reason": "no contours"}

    background = float(np.median(gray))
    probe_box = probe.get("bbox") if (probe and probe.get("found")) else None

    def on_expected_side(bx, by, bw, bh):
        """Reject a candidate that sits on the wrong side of the cantilever."""
        if not probe_box or cell_side == "anywhere":
            return True
        px, py, pw, ph = probe_box
        cx, cy = bx + bw / 2 + x0, by + bh / 2 + y0
        if cell_side == "right":
            return cx > px + pw * 0.5
        if cell_side == "left":
            return cx < px + pw * 0.5
        if cell_side == "below":
            return cy > py + ph * 0.5
        if cell_side == "above":
            return cy < py + ph * 0.5
        return True

    # An edge-derived outline is often a broken ring: filling it does nothing,
    # and it scores terribly on solidity even though its convex hull is exactly
    # the cell. So a ring-like contour is also judged as its hull. Only
    # ring-like ones: taking the hull of an already-solid blob can bridge the
    # cell to whatever else the mask caught and produce a region far taller
    # than the cell really is.
    candidates = []
    for contour in contours:
        candidates.append(contour)
        if len(contour) < 5:
            continue
        hull = cv2.convexHull(contour)
        area = float(cv2.contourArea(contour))
        hull_area = float(cv2.contourArea(hull))
        if hull_area <= 0:
            continue
        if area / hull_area < 0.5:  # genuinely ring-like or broken
            candidates.append(hull)

    best, best_score = None, -np.inf
    rejected_dark = 0
    rejected_side = 0
    for contour in candidates:
        area = float(cv2.contourArea(contour))
        if area < min_area_frac * view_area or area > max_area_frac * view_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue

        bx, by, bw, bh = cv2.boundingRect(contour)
        # A blob spanning almost the whole field is the frame itself, not a cell.
        if bw > 0.95 * view_w and bh > 0.95 * view_h:
            continue

        # Cells are convex. Ragged noise clusters are not, and this separates
        # them far more reliably than area alone.
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = area / hull_area if hull_area > 0 else 0.0
        if solidity < 0.5:
            continue

        if not on_expected_side(bx, by, bw, bh):
            rejected_side += 1
            continue

        if reject_dark:
            # The cantilever is close to black. Anything much darker than the
            # frame's own median is the probe or its shadow, not a cell.
            blob = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(blob, [contour], -1, 255, thickness=cv2.FILLED)
            mean_intensity = float(cv2.mean(gray, mask=blob)[0])
            if mean_intensity < dark_margin * background:
                rejected_dark += 1
                continue

        circularity = float(np.clip(4.0 * np.pi * area / perimeter ** 2, 0.0, 1.0))

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        centrality = 1.0 - float(np.hypot(cx - cx_view, cy - cy_view)) / (diag / 2.0)

        # A blob running off the edge of the searched region is almost always
        # the cell merged with something else, and its height is then wrong in
        # a way nothing downstream can detect.
        touches_edge = (
            bx <= 1 or by <= 1 or bx + bw >= view_w - 1 or by + bh >= view_h - 1
        )

        # Roundness and solidity identify a cell; size breaks ties against
        # small artefacts; centrality prefers the object being compressed.
        score = (
            1.6 * circularity
            + 1.2 * solidity
            + 1.0 * centrality
            + 1.5 * np.sqrt(area / view_area)
            - (1.5 if touches_edge else 0.0)
        )
        if score > best_score:
            best_score, best = score, {
                "contour": contour,
                "area_px": area,
                "circularity": circularity,
                "solidity": solidity,
                "centrality": centrality,
                "score": float(score),
                "centroid": (cx, cy),
            }

    if best is None:
        reason = "no candidate of a plausible size and shape"
        if rejected_dark:
            reason += f"; {rejected_dark} rejected as too dark (probe)"
        if rejected_side:
            reason += f"; {rejected_side} rejected on the wrong side of the probe"
        return {"found": False, "reason": reason}

    contour = best["contour"]
    x, y, w, h = cv2.boundingRect(contour)
    ellipse = None
    if len(contour) >= 5:
        (ex, ey), (maj, minor), angle = cv2.fitEllipse(contour)
        ellipse = {
            "center": (float(ex + x0), float(ey + y0)),
            "axes": (float(maj), float(minor)),
            "angle": float(angle),
        }

    # Height comes from the bounding box. An ellipse fit was tried here, but
    # cv2.fitEllipse's axis ordering and angle convention make "which axis is
    # vertical" ambiguous, and getting it backwards silently corrupts every
    # deformation derived from the video. The box is cruder and honest.
    height_px, width_px = float(h), float(w)

    return {
        "found": True,
        "bbox": (int(x + x0), int(y + y0), int(w), int(h)),
        "center": (float(best["centroid"][0] + x0), float(best["centroid"][1] + y0)),
        "height_px": height_px,
        "width_px": width_px,
        "area_px": best["area_px"],
        "circularity": best["circularity"],
        "solidity": best["solidity"],
        "score": best["score"],
        "rejected_dark": rejected_dark,
        "rejected_side": rejected_side,
        "ellipse": ellipse,
        "contour": contour + np.array([[x0, y0]]),
        "roi_px": (x0, y0, x1, y1),
    }


def annotate(
    frame_rgb,
    detection,
    label=None,
    color=(255, 60, 60),
    thickness=None,
    nucleus=None,
    probe=None,
):
    """
    Draw what was found onto a copy of the frame.

    Cell in red with its outline in green, nucleus in purple, cantilever in
    grey. Drawing the probe as well as the cell is worth the clutter: when the
    detector grabs the wrong object, seeing which box landed where says
    immediately whether the problem is the side constraint or the contrast.
    """
    _require_cv2()
    out = np.array(frame_rgb).copy()
    t = thickness or max(1, int(round(min(out.shape[:2]) / 250)))

    if probe and probe.get("found"):
        px, py, pw, ph = probe["bbox"]
        cv2.rectangle(out, (px, py), (px + pw, py + ph), (120, 120, 120), t)
        cv2.putText(
            out, "probe", (px, max(14, py - 6)), cv2.FONT_HERSHEY_SIMPLEX,
            max(0.35, min(out.shape[:2]) / 900.0), (120, 120, 120), t, cv2.LINE_AA,
        )

    if not detection or not detection.get("found"):
        return out

    x, y, w, h = detection["bbox"]
    cv2.rectangle(out, (x, y), (x + w, y + h), color, t)
    if detection.get("contour") is not None:
        cv2.drawContours(out, [detection["contour"]], -1, (60, 220, 90), t)

    # A vertical bar marking the measured height, which is the quantity that
    # actually feeds the deformation estimate.
    cx = x + w // 2
    cv2.arrowedLine(out, (cx, y), (cx, y + h), (255, 220, 40), t, tipLength=0.08)
    cv2.arrowedLine(out, (cx, y + h), (cx, y), (255, 220, 40), t, tipLength=0.08)

    if nucleus and nucleus.get("found"):
        nx, ny, nw, nh = nucleus["bbox"]
        cv2.rectangle(out, (nx, ny), (nx + nw, ny + nh), (170, 90, 200), t)
        if nucleus.get("contour") is not None:
            cv2.drawContours(out, [nucleus["contour"]], -1, (170, 90, 200), t)
        cv2.putText(
            out, "nucleus", (nx, min(out.shape[0] - 4, ny + nh + 16)),
            cv2.FONT_HERSHEY_SIMPLEX, max(0.35, min(out.shape[:2]) / 900.0),
            (170, 90, 200), t, cv2.LINE_AA,
        )

    if label:
        scale = max(0.4, min(out.shape[:2]) / 700.0)
        cv2.putText(
            out, label, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, scale,
            (0, 0, 0), t + 2, cv2.LINE_AA,
        )
        cv2.putText(
            out, label, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, scale,
            (255, 255, 255), t, cv2.LINE_AA,
        )
    return out


def crop(frame_rgb, detection, pad_frac=0.35):
    """Crop tightly around the detection, with padding, for a side panel."""
    frame = np.asarray(frame_rgb)
    if not detection or not detection.get("found"):
        return frame
    H, W = frame.shape[:2]
    x, y, w, h = detection["bbox"]
    px, py = int(w * pad_frac), int(h * pad_frac)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(W, x + w + px), min(H, y + h + py)
    return frame[y0:y1, x0:x1]


# ----------------------------------------------------------------- tracking


def track_cell(path, n_samples=60, roi=None, sensitivity=1.0, start=0, end=None,
               enhance=None, cell_side="anywhere", reject_dark=True, track_nucleus=False):
    """
    Measure the cell across the video.

    Returns
    -------
    dict
        ``frames`` (sampled indices), ``height_px`` (NaN where detection
        failed), ``width_px``, ``found`` and ``detections``.
    """
    _require_cv2()
    info = probe(path)
    total = info["n_frames"]
    end = total - 1 if end is None else min(int(end), total - 1)
    start = max(0, int(start))
    if end <= start:
        end = total - 1

    indices = np.unique(
        np.linspace(start, end, max(2, int(n_samples))).astype(int)
    )

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    heights, widths, found, detections = [], [], [], []
    try:
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                heights.append(np.nan)
                widths.append(np.nan)
                found.append(False)
                detections.append(None)
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if enhance:
                rgb = enhance_frame(rgb, **enhance)
            probe_here = detect_probe(rgb) if cell_side != "anywhere" else None
            det = detect_cell(
                rgb, roi=roi, sensitivity=sensitivity,
                probe=probe_here, cell_side=cell_side, reject_dark=reject_dark,
            )
            if track_nucleus and det.get("found"):
                det["nucleus"] = detect_nucleus(rgb, det)
            detections.append(det)
            if det.get("found"):
                heights.append(det["height_px"])
                widths.append(det["width_px"])
                found.append(True)
            else:
                heights.append(np.nan)
                widths.append(np.nan)
                found.append(False)
    finally:
        cap.release()

    return {
        "frames": indices,
        "height_px": np.array(heights, dtype=float),
        "width_px": np.array(widths, dtype=float),
        "found": np.array(found, dtype=bool),
        "detections": detections,
        "info": info,
    }


def deformation_from_track(track, reference="first"):
    """
    Turn tracked cell heights into a relative deformation.

        eps_video = 1 - h(frame) / h_reference

    ``reference`` is either ``"first"`` (the first successful detection) or
    ``"max"`` (the largest height seen, more robust when the first frames are
    already slightly compressed).
    """
    heights = np.asarray(track["height_px"], dtype=float)
    valid = np.isfinite(heights)
    if not valid.any():
        return np.full(heights.shape, np.nan), float("nan")

    h_ref = float(np.nanmax(heights)) if reference == "max" else float(heights[valid][0])
    if h_ref <= 0:
        return np.full(heights.shape, np.nan), float("nan")
    return 1.0 - heights / h_ref, h_ref


# ------------------------------------------------- curve <-> video mapping


def frame_for_epsilon(epsilon, contact_frame, end_frame, epsilon_at_end):
    """
    Map a point on the force curve to a frame index.

    Assumes the piezo ramps at constant speed between the contact frame and
    the end frame, so deformation grows linearly with frame number. That holds
    for a standard constant-velocity approach.
    """
    if epsilon_at_end <= 0:
        return int(contact_frame)
    span = float(end_frame) - float(contact_frame)
    frame = float(contact_frame) + span * (float(epsilon) / float(epsilon_at_end))
    lo, hi = sorted((float(contact_frame), float(end_frame)))
    return int(round(float(np.clip(frame, lo, hi))))


def epsilon_for_frame(frame, contact_frame, end_frame, epsilon_at_end):
    """Inverse of :func:`frame_for_epsilon`."""
    span = float(end_frame) - float(contact_frame)
    if span == 0:
        return 0.0
    return float(epsilon_at_end) * (float(frame) - float(contact_frame)) / span


def align_scale(epsilon_video, epsilon_curve):
    """
    Least-squares scale factor between video-derived and curve deformation.

    A factor near 1 means the cell height and contact point used for the force
    curve agree with what the video shows. A factor of, say, 1.3 means the
    curve's deformation axis is 30 % smaller than the cell actually deformed,
    which usually points at the cell height being wrong by that factor.
    """
    a = np.asarray(epsilon_video, dtype=float)
    b = np.asarray(epsilon_curve, dtype=float)
    good = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-9)
    if good.sum() < 3:
        return float("nan"), float("nan")
    scale = float(np.sum(a[good] * b[good]) / np.sum(b[good] ** 2))
    residual = a[good] - scale * b[good]
    ss_tot = float(np.sum((a[good] - a[good].mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    return scale, r2


# ------------------------------------------------------- Google Drive fetch


def drive_file_id(url):
    """Pull the file id out of the usual Google Drive link shapes."""
    import re

    if not url:
        return None
    for pattern in (r"/file/d/([A-Za-z0-9_-]{10,})", r"[?&]id=([A-Za-z0-9_-]{10,})"):
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    # A bare id.
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", url.strip()):
        return url.strip()
    return None


def download_drive_video(url, dest_path, max_bytes=512 * 1024 * 1024):
    """
    Best-effort download of a publicly shared Drive video.

    Only works for files shared with "anyone with the link". Anything else
    (restricted files, sign-in walls, quota pages) raises, and the caller
    should ask for a direct upload instead.
    """
    import urllib.request
    import urllib.parse
    import http.cookiejar

    file_id = drive_file_id(url)
    if not file_id:
        raise ValueError("That does not look like a Google Drive file link.")

    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]
    base = "https://drive.google.com/uc?export=download"

    response = opener.open(f"{base}&id={file_id}", timeout=60)
    payload = response.read(65536)

    # Large files return an interstitial page holding a confirm token.
    if b"<html" in payload[:2048].lower():
        import re

        text = payload.decode("utf-8", errors="ignore")
        token = None
        match = re.search(r"confirm=([0-9A-Za-z_-]+)", text)
        if match:
            token = match.group(1)
        if token is None:
            raise RuntimeError(
                "Google Drive returned a web page instead of the file. The video is "
                "probably not shared with 'anyone with the link'. Upload it directly."
            )
        response = opener.open(f"{base}&confirm={token}&id={file_id}", timeout=60)
        payload = response.read(65536)
        if b"<html" in payload[:2048].lower():
            raise RuntimeError(
                "Google Drive would not serve the file. Upload it directly instead."
            )

    written = 0
    with open(dest_path, "wb") as handle:
        handle.write(payload)
        written += len(payload)
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                raise RuntimeError(
                    f"Video is larger than the {max_bytes / 1e6:.0f} MB limit; "
                    f"trim it or upload a shorter clip."
                )
            handle.write(chunk)

    if written < 10_000:
        raise RuntimeError("Downloaded file is too small to be a video.")
    return dest_path

# --------------------------------------------------------------- enhancement


def enhance_frame(frame_rgb, clahe_clip=0.0, gamma=1.0, brightness=0, contrast=1.0):
    """
    Adjust a frame before detection or display.

    Cells in brightfield are often barely above the background, and the
    detector can only find what is visible. CLAHE equalises local contrast,
    which brings out a faint cell outline without blowing out the bright
    field; gamma lifts the dark end where a cell sitting in the probe's shadow
    lives.
    """
    _require_cv2()
    out = np.asarray(frame_rgb).astype(np.float32)

    if contrast != 1.0 or brightness:
        out = out * float(contrast) + float(brightness)
    out = np.clip(out, 0, 255).astype(np.uint8)

    if gamma and gamma != 1.0:
        table = (np.linspace(0, 1, 256) ** (1.0 / float(gamma)) * 255).astype(np.uint8)
        out = cv2.LUT(out, table)

    if clahe_clip and clahe_clip > 0:
        lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return out


# ---------------------------------------------------------------- the probe


def detect_probe(frame_rgb, darkness_percentile=8.0, min_area_frac=0.01):
    """
    Find the cantilever, which is the large very dark object in the frame.

    Worth locating for two reasons: it is the thing the cell detector keeps
    grabbing by mistake, and once its position is known the search for the
    cell can be restricted to the correct side of it.
    """
    _require_cv2()
    frame = np.asarray(frame_rgb)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)

    cutoff = float(np.percentile(gray, darkness_percentile))
    mask = (gray <= cutoff).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_total = float(gray.size)
    best, best_area = None, 0.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area_frac * area_total or area <= best_area:
            continue
        best, best_area = contour, area

    if best is None:
        return {"found": False, "reason": "no large dark object"}

    x, y, w, h = cv2.boundingRect(best)
    return {
        "found": True,
        "bbox": (int(x), int(y), int(w), int(h)),
        "area_px": best_area,
        "mean_intensity": float(gray[y : y + h, x : x + w].mean()),
        "contour": best,
    }


# --------------------------------------------------------------- the nucleus


def detect_nucleus(frame_rgb, cell, min_area_frac=0.04, max_area_frac=0.75):
    """
    Find the nucleus inside an already-detected cell.

    The nucleus is denser than the cytoplasm around it, so within the cell's
    own bounding box it shows as a darker, roundish region. Searching only
    inside the cell is what makes this tractable: the same threshold applied
    to the whole frame would return the probe.
    """
    _require_cv2()
    if not cell or not cell.get("found"):
        return {"found": False, "reason": "no cell to look inside"}

    frame = np.asarray(frame_rgb)
    x, y, w, h = cell["bbox"]
    pad_x, pad_y = int(w * 0.05), int(h * 0.05)
    x0, y0 = max(0, x + pad_x), max(0, y + pad_y)
    x1, y1 = min(frame.shape[1], x + w - pad_x), min(frame.shape[0], y + h - pad_y)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return {"found": False, "reason": "cell too small to look inside"}

    view = frame[y0:y1, x0:x1]
    gray = cv2.medianBlur(cv2.cvtColor(view, cv2.COLOR_RGB2GRAY), 5)

    # Otsu inside the cell, not a fixed percentile: the fraction of the box
    # the nucleus occupies varies with how squashed the cell is, and a fixed
    # cut either swallows the whole cytoplasm or misses the nucleus entirely.
    k = max(3, (min(view.shape[:2]) // 20) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    candidates_masks = []
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    candidates_masks.append(otsu)
    for percentile in (20.0, 30.0):
        cutoff = float(np.percentile(gray, percentile))
        candidates_masks.append((gray <= cutoff).astype(np.uint8) * 255)

    contours = []
    for mask in candidates_masks:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(found)
    view_area = float(view.shape[0] * view.shape[1])

    best, best_score = None, -np.inf
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area_frac * view_area or area > max_area_frac * view_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        circularity = float(np.clip(4 * np.pi * area / perimeter ** 2, 0, 1))
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        centrality = 1.0 - float(
            np.hypot(cx - view.shape[1] / 2, cy - view.shape[0] / 2)
        ) / (np.hypot(view.shape[1], view.shape[0]) / 2)
        score = 1.5 * circularity + 1.0 * centrality + np.sqrt(area / view_area)
        if score > best_score:
            best_score, best = score, (contour, area, circularity, cx, cy)

    if best is None:
        return {"found": False, "reason": "no nucleus-like region inside the cell"}

    contour, area, circularity, cx, cy = best
    nx, ny, nw, nh = cv2.boundingRect(contour)
    return {
        "found": True,
        "bbox": (int(nx + x0), int(ny + y0), int(nw), int(nh)),
        "center": (float(cx + x0), float(cy + y0)),
        "height_px": float(nh),
        "width_px": float(nw),
        "area_px": area,
        "circularity": circularity,
        "contour": contour + np.array([[x0, y0]]),
        "area_fraction_of_cell": float(area / max(w * h, 1)),
    }


# ------------------------------------------------------------- Box fetch


def box_shared_id(url):
    """Pull the shared-link id out of a Box URL."""
    import re

    if not url:
        return None
    match = re.search(r"box\.com/s/([A-Za-z0-9]+)", url)
    return match.group(1) if match else None


def download_box_video(url, dest_path, access_token=None, max_bytes=512 * 1024 * 1024):
    """
    Download a video from Box.

    With an ``access_token`` this goes through the Box API, which is the
    reliable path: the shared link is resolved to a file id and the content
    endpoint streams the file. Without a token it falls back to the public
    download URL, which only works for links shared as "people with the link"
    and no password.

    Box tokens are short lived. Put a developer token in
    ``.streamlit/secrets.toml`` as ``[box] access_token = "..."`` for occasional
    use, or set up a service app if this needs to keep working unattended.
    """
    import json
    import urllib.error
    import urllib.request

    if not url or "box.com" not in url:
        raise ValueError("That does not look like a Box link.")

    def fetch(request):
        return urllib.request.urlopen(request, timeout=90)

    response = None
    if access_token:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "BoxApi": f"shared_link={url}",
        }
        try:
            info = fetch(
                urllib.request.Request(
                    "https://api.box.com/2.0/shared_items", headers=headers
                )
            )
            item = json.loads(info.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"Box rejected the shared link ({exc.code}). Check that the token "
                f"is current and that the link is shared with your account."
            ) from exc

        if item.get("type") != "file":
            raise RuntimeError(
                f"That Box link points to a {item.get('type', 'thing')}, not a file."
            )
        file_id = item["id"]
        response = fetch(
            urllib.request.Request(
                f"https://api.box.com/2.0/files/{file_id}/content", headers=headers
            )
        )
    else:
        shared = box_shared_id(url)
        if not shared:
            raise ValueError(
                "Could not read a shared-link id from that URL. It should look "
                "like https://app.box.com/s/xxxxxxxx"
            )
        request = urllib.request.Request(
            f"https://app.box.com/public/static/{shared}",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        try:
            response = fetch(request)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"Box refused the public download ({exc.code}). Either the link is "
                f"not shared publicly, or it needs an access token. Add one under "
                f"[box] in secrets, or download the file and upload it here."
            ) from exc

    written = 0
    with open(dest_path, "wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            if written == 0 and chunk[:512].lstrip().lower().startswith(b"<!doctype html"):
                raise RuntimeError(
                    "Box returned a web page rather than the file, which means the "
                    "link is not publicly downloadable. Use an access token or "
                    "upload the file directly."
                )
            written += len(chunk)
            if written > max_bytes:
                raise RuntimeError(
                    f"Video is larger than the {max_bytes / 1e6:.0f} MB limit; trim "
                    f"it or upload a shorter clip."
                )
            handle.write(chunk)

    if written < 10_000:
        raise RuntimeError("Downloaded file is too small to be a video.")
    return dest_path


def fetch_video(url, dest_path, box_token=None):
    """Download from whichever service the URL points at."""
    if not url:
        raise ValueError("No link given.")
    if "box.com" in url:
        return download_box_video(url, dest_path, access_token=box_token)
    if "drive.google.com" in url or "docs.google.com" in url:
        return download_drive_video(url, dest_path)
    raise ValueError(
        "Only Google Drive and Box links are recognised. Upload the file directly "
        "for anything else."
    )
