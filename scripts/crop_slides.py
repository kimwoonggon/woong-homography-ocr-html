#!/usr/bin/env python3
"""Crop photographed presentation slides with homography."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DETECT_MAX_DIM = 1200


@dataclass
class Candidate:
    points: np.ndarray
    method: str
    mask_name: str
    contour_area: float
    score: float = 0.0


@dataclass
class Detection:
    candidate: Optional[Candidate]
    mask: np.ndarray
    message: str


def image_paths(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"unsupported image extension: {input_path}")
        return [input_path]

    if input_path.is_dir():
        return sorted(
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    raise FileNotFoundError(f"input does not exist: {input_path}")


def safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._") or "image"


def resize_for_detection(image: np.ndarray) -> Tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    largest = max(height, width)
    if largest <= DETECT_MAX_DIM:
        return image.copy(), 1.0

    scale = DETECT_MAX_DIM / float(largest)
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def order_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)

    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(4)
    rect[0] = pts[np.argmin(sums)]
    rect[2] = pts[np.argmax(sums)]
    rect[1] = pts[np.argmin(diffs)]
    rect[3] = pts[np.argmax(diffs)]

    if len({tuple(point) for point in rect}) == 4:
        return rect

    y_sorted = pts[np.argsort(pts[:, 1])]
    top = y_sorted[:2][np.argsort(y_sorted[:2, 0])]
    bottom = y_sorted[2:][np.argsort(y_sorted[2:, 0])]
    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)


def side_lengths(points: np.ndarray) -> Tuple[float, float, float, float]:
    tl, tr, br, bl = order_points(points)
    width_top = float(np.linalg.norm(tr - tl))
    width_bottom = float(np.linalg.norm(br - bl))
    height_right = float(np.linalg.norm(br - tr))
    height_left = float(np.linalg.norm(bl - tl))
    return width_top, width_bottom, height_right, height_left


def polygon_mask(shape: Tuple[int, int], points: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(order_points(points)).astype(np.int32), 255)
    return mask


def normalized_aspect_score(aspect: float) -> float:
    if aspect <= 0:
        return 0.0
    targets = (16.0 / 9.0, 4.0 / 3.0, 16.0 / 10.0)
    best = min(abs(math.log(aspect / target)) for target in targets)
    return max(0.0, 1.0 - best / math.log(2.0))


def area_score(area_ratio: float) -> float:
    if area_ratio <= 0:
        return 0.0
    if 0.18 <= area_ratio <= 0.90:
        return 1.0
    if area_ratio < 0.18:
        return max(0.0, area_ratio / 0.18)
    return max(0.0, (0.98 - area_ratio) / 0.08)


def score_candidate(
    candidate: Candidate,
    gray: np.ndarray,
    edges: np.ndarray,
    text_mask: np.ndarray,
) -> float:
    image_area = float(gray.shape[0] * gray.shape[1])
    points = order_points(candidate.points)
    polygon_area = abs(float(cv2.contourArea(points)))
    if polygon_area <= image_area * 0.015:
        return 0.0

    width_top, width_bottom, height_right, height_left = side_lengths(points)
    out_width = max(width_top, width_bottom)
    out_height = max(height_right, height_left)
    if out_width < 40 or out_height < 40:
        return 0.0

    aspect = out_width / max(out_height, 1.0)
    if aspect < 0.8:
        aspect = 1.0 / aspect

    rect = cv2.minAreaRect(points)
    rect_area = max(float(rect[1][0] * rect[1][1]), 1.0)
    rectangularity = min(1.0, polygon_area / rect_area)

    mask = polygon_mask(gray.shape, points)
    inside_area = max(float(cv2.countNonZero(mask)), 1.0)
    edge_density = float(cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=mask))) / inside_area
    text_density = float(cv2.countNonZero(cv2.bitwise_and(text_mask, text_mask, mask=mask))) / inside_area

    moments = cv2.moments(points)
    if abs(moments["m00"]) > 1e-6:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
    else:
        cx, cy = points.mean(axis=0)
    center_distance = math.hypot(
        (cx - gray.shape[1] / 2.0) / max(gray.shape[1], 1),
        (cy - gray.shape[0] / 2.0) / max(gray.shape[0], 1),
    )
    center_score = max(0.0, 1.0 - center_distance * 2.4)

    mean_value = float(cv2.mean(gray, mask=mask)[0]) / 255.0
    brightness_score = max(0.0, 1.0 - abs(mean_value - 0.72) / 0.72)

    edge_score = min(1.0, edge_density * 14.0)
    text_score = min(1.0, text_density * 6.0)

    score = float(
        0.28 * area_score(polygon_area / image_area)
        + 0.23 * rectangularity
        + 0.19 * normalized_aspect_score(aspect)
        + 0.12 * edge_score
        + 0.10 * text_score
        + 0.05 * center_score
        + 0.03 * brightness_score
    )
    if "min-area-rect" in candidate.method:
        score -= 0.07
    elif "hull-quad" in candidate.method:
        score += 0.045
    elif "contour-quad" in candidate.method:
        score += 0.035
    if candidate.mask_name == "bright":
        score += 0.04
    elif candidate.mask_name == "edges_closed":
        score += 0.03
    elif candidate.mask_name == "combined":
        score += 0.015
    elif candidate.mask_name == "text":
        score -= 0.08
    return float(min(1.0, max(0.0, score)))


def build_masks(image: np.ndarray) -> Dict[str, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edge_low = max(20, int(np.percentile(blurred, 45) * 0.55))
    edge_high = max(edge_low + 30, int(np.percentile(blurred, 85) * 1.15))
    edges = cv2.Canny(blurred, edge_low, edge_high)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    bright_threshold = max(105, int(np.percentile(value, 58)))
    bright = np.where((value >= bright_threshold) & (saturation <= 185), 255, 0).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    text = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )
    text = cv2.morphologyEx(text, cv2.MORPH_CLOSE, np.ones((5, 3), np.uint8), iterations=1)
    text_dilated = cv2.dilate(text, np.ones((7, 7), np.uint8), iterations=2)

    combined = cv2.bitwise_or(edges_closed, bright)
    combined = cv2.bitwise_or(combined, text_dilated)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8), iterations=2)

    return {
        "edges": edges,
        "edges_closed": edges_closed,
        "bright": bright,
        "text": text,
        "combined": combined,
    }


def add_candidate(
    candidates: List[Candidate],
    points: np.ndarray,
    method: str,
    mask_name: str,
    contour_area: float,
    shape: Tuple[int, int],
) -> None:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    if not cv2.isContourConvex(np.round(order_points(points)).astype(np.int32)):
        return

    image_area = float(shape[0] * shape[1])
    area = abs(float(cv2.contourArea(order_points(points))))
    if area < image_area * 0.015 or area > image_area * 0.985:
        return

    width_top, width_bottom, height_right, height_left = side_lengths(points)
    if min(width_top, width_bottom, height_right, height_left) < 25:
        return

    candidates.append(Candidate(points=order_points(points), method=method, mask_name=mask_name, contour_area=contour_area))


def find_candidates(mask: np.ndarray, mask_name: str) -> List[Candidate]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    candidates: List[Candidate] = []
    image_area = float(mask.shape[0] * mask.shape[1])

    for contour in contours[:35]:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < image_area * 0.015:
            continue

        perimeter = cv2.arcLength(contour, True)
        for epsilon in (0.015, 0.025, 0.04, 0.06, 0.085):
            approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)
            if len(approx) == 4:
                add_candidate(
                    candidates,
                    approx.reshape(4, 2),
                    f"{mask_name}:contour-quad",
                    mask_name,
                    contour_area,
                    mask.shape,
                )
                break

        hull = cv2.convexHull(contour)
        hull_perimeter = cv2.arcLength(hull, True)
        for epsilon in (0.005, 0.01, 0.015, 0.025, 0.04, 0.06, 0.085):
            approx = cv2.approxPolyDP(hull, epsilon * hull_perimeter, True)
            if len(approx) == 4:
                add_candidate(
                    candidates,
                    approx.reshape(4, 2),
                    f"{mask_name}:hull-quad",
                    mask_name,
                    contour_area,
                    mask.shape,
                )
                break

        if contour_area >= image_area * 0.03:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            add_candidate(
                candidates,
                box,
                f"{mask_name}:min-area-rect",
                mask_name,
                contour_area,
                mask.shape,
            )

    return candidates


def dedupe_candidates(candidates: Sequence[Candidate]) -> List[Candidate]:
    deduped: List[Candidate] = []
    seen: set = set()
    for candidate in candidates:
        key = tuple(np.round(order_points(candidate.points) / 8.0).astype(int).reshape(-1).tolist())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def detect_slide(image: np.ndarray) -> Detection:
    resized, scale = resize_for_detection(image)
    masks = build_masks(resized)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    candidates: List[Candidate] = []
    for name in ("combined", "edges_closed", "bright", "text"):
        candidates.extend(find_candidates(masks[name], name))

    candidates = dedupe_candidates(candidates)
    for candidate in candidates:
        candidate.score = score_candidate(candidate, gray, masks["edges"], masks["text"])

    scored = [candidate for candidate in candidates if candidate.score > 0.0]
    if not scored:
        return Detection(candidate=None, mask=masks["combined"], message="no rectangular slide candidate found")

    best = max(scored, key=lambda candidate: candidate.score)
    best.points = order_points(best.points / scale)
    return Detection(candidate=best, mask=masks["combined"], message="ok")


def warp_crop(
    image: np.ndarray,
    points: np.ndarray,
    max_output_width: int,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    ordered = order_points(points)
    width_top, width_bottom, height_right, height_left = side_lengths(ordered)
    output_width = int(round(max(width_top, width_bottom)))
    output_height = int(round(max(height_right, height_left)))

    if output_width <= 0 or output_height <= 0:
        raise ValueError("candidate has invalid output size")

    if max_output_width > 0 and output_width > max_output_width:
        scale = max_output_width / float(output_width)
        output_width = max(1, int(round(output_width * scale)))
        output_height = max(1, int(round(output_height * scale)))

    destination = np.array(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )
    transform, _ = cv2.findHomography(ordered.astype(np.float32), destination)
    if transform is None:
        transform = cv2.getPerspectiveTransform(ordered.astype(np.float32), destination)
    crop = cv2.warpPerspective(image, transform, (output_width, output_height))
    return crop, (output_width, output_height)


def debug_overlay(image: np.ndarray, candidate: Optional[Candidate], message: str) -> np.ndarray:
    overlay = image.copy()
    if candidate is not None:
        points = np.round(order_points(candidate.points)).astype(np.int32)
        cv2.polylines(overlay, [points], True, (0, 255, 0), 4, cv2.LINE_AA)
        label = f"{candidate.score:.3f} {candidate.method}"
    else:
        label = message

    cv2.putText(
        overlay,
        label[:120],
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def write_manifest(output_dir: Path, manifest: Dict[str, object]) -> None:
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def process_image(
    source: Path,
    output_dir: Path,
    debug: bool,
    max_output_width: int,
) -> Dict[str, object]:
    record: Dict[str, object] = {
        "source": str(source),
        "crop_path": None,
        "status": "miss",
        "score": None,
        "quad": None,
        "output_size": None,
        "method": None,
        "message": "",
    }

    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        record.update(status="error", message="failed to read image")
        return record

    detection = detect_slide(image)
    candidate = detection.candidate
    if candidate is None:
        record.update(message=detection.message)
        if debug:
            write_debug_outputs(source, output_dir, image, detection.mask, None, detection.message)
        return record

    try:
        crop, output_size = warp_crop(image, candidate.points, max_output_width)
    except Exception as exc:
        record.update(status="error", message=str(exc))
        if debug:
            write_debug_outputs(source, output_dir, image, detection.mask, candidate, str(exc))
        return record

    crop_path = output_dir / f"{safe_stem(source)}_crop.png"
    cv2.imwrite(str(crop_path), crop)

    record.update(
        crop_path=str(crop_path),
        status="ok",
        score=round(float(candidate.score), 6),
        quad=[[round(float(x), 3), round(float(y), 3)] for x, y in order_points(candidate.points)],
        output_size=[int(output_size[0]), int(output_size[1])],
        method=candidate.method,
        message="ok",
    )

    if debug:
        write_debug_outputs(source, output_dir, image, detection.mask, candidate, "ok")

    return record


def write_debug_outputs(
    source: Path,
    output_dir: Path,
    image: np.ndarray,
    mask: np.ndarray,
    candidate: Optional[Candidate],
    message: str,
) -> None:
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_stem(source)
    overlay = debug_overlay(image, candidate, message)
    cv2.imwrite(str(debug_dir / f"{stem}_overlay.png"), overlay)
    cv2.imwrite(str(debug_dir / f"{stem}_mask.png"), mask)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="image file or directory of images")
    parser.add_argument("--output", type=Path, default=Path("crops"), help="output directory")
    parser.add_argument("--debug", action="store_true", help="write debug masks and overlays")
    parser.add_argument(
        "--max-output-width",
        type=int,
        default=1800,
        help="cap crop width while preserving aspect ratio; use 0 for no cap",
    )
    parser.add_argument("--fail-on-miss", action="store_true", help="exit non-zero if any image is missed or errors")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.max_output_width < 0:
        parser.error("--max-output-width must be >= 0")

    try:
        sources = image_paths(args.input)
    except Exception as exc:
        parser.error(str(exc))

    args.output.mkdir(parents=True, exist_ok=True)
    items = [process_image(source, args.output, args.debug, args.max_output_width) for source in sources]
    manifest: Dict[str, object] = {
        "input": str(args.input),
        "output_dir": str(args.output),
        "count": len(items),
        "ok_count": sum(1 for item in items if item["status"] == "ok"),
        "items": items,
    }
    write_manifest(args.output, manifest)

    failed = [item for item in items if item["status"] != "ok"]
    if failed and args.fail_on_miss:
        print(f"{len(failed)} image(s) missed or failed; see {args.output / 'manifest.json'}", file=sys.stderr)
        return 2

    print(f"wrote {manifest['ok_count']}/{manifest['count']} crop(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
