import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "crop_slides.py"


def make_slide(width=760, height=430):
    slide = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.rectangle(slide, (0, 0), (width - 1, height - 1), (30, 30, 30), 6)
    cv2.putText(slide, "Homography OCR", (60, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 4)
    for index, text in enumerate(("Detect rectangle", "Warp perspective", "Create HTML")):
        y = 175 + index * 65
        cv2.circle(slide, (80, y - 12), 8, (35, 35, 35), -1)
        cv2.putText(slide, text, (110, y), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (30, 30, 30), 3)
    return slide


def place_perspective_slide(tmp_path, name="slide.jpg", noisy=True):
    canvas = np.full((780, 1100, 3), (76, 92, 101), dtype=np.uint8)
    if noisy:
        rng = np.random.default_rng(7)
        noise = rng.normal(0, 8, canvas.shape).astype(np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    slide = make_slide()
    h, w = slide.shape[:2]
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    dst = np.array([[180, 120], [930, 75], [1000, 610], [105, 655]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(slide, matrix, (canvas.shape[1], canvas.shape[0]))
    mask = cv2.warpPerspective(np.full((h, w), 255, dtype=np.uint8), matrix, (canvas.shape[1], canvas.shape[0]))
    canvas[mask > 0] = warped[mask > 0]

    path = tmp_path / name
    cv2.imwrite(str(path), canvas)
    return path


def run_crop(input_path, output_dir, *args):
    command = [sys.executable, str(SCRIPT), str(input_path), "--output", str(output_dir), *args]
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True)


def load_manifest(output_dir):
    return json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))


def test_crop_perspective_slide_has_expected_aspect_and_content(tmp_path):
    source = place_perspective_slide(tmp_path)
    output_dir = tmp_path / "crops"

    result = run_crop(source, output_dir, "--max-output-width", "900", "--fail-on-miss")

    assert result.returncode == 0, result.stderr
    manifest = load_manifest(output_dir)
    item = manifest["items"][0]
    assert item["status"] == "ok"
    crop = cv2.imread(item["crop_path"])
    assert crop is not None

    height, width = crop.shape[:2]
    assert 1.45 <= width / height <= 1.95
    assert width <= 900
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    assert gray.mean() > 175
    assert cv2.Canny(gray, 70, 160).mean() > 2.0


def test_noisy_background_selects_main_rectangle(tmp_path):
    source = place_perspective_slide(tmp_path, "noisy.jpg", noisy=True)
    output_dir = tmp_path / "out"

    result = run_crop(source, output_dir, "--debug", "--fail-on-miss")

    assert result.returncode == 0, result.stderr
    manifest = load_manifest(output_dir)
    item = manifest["items"][0]
    assert item["status"] == "ok"
    assert item["score"] > 0.55
    assert (output_dir / "debug" / "noisy_overlay.png").exists()
    assert (output_dir / "debug" / "noisy_mask.png").exists()


def test_missing_rectangle_is_reported_and_fail_on_miss_exits_nonzero(tmp_path):
    blank = np.full((360, 540, 3), 88, dtype=np.uint8)
    source = tmp_path / "blank.jpg"
    cv2.imwrite(str(source), blank)
    output_dir = tmp_path / "miss"

    result = run_crop(source, output_dir, "--fail-on-miss")

    assert result.returncode != 0
    manifest = load_manifest(output_dir)
    item = manifest["items"][0]
    assert item["status"] == "miss"
    assert item["crop_path"] is None
    assert "candidate" in item["message"]


def test_directory_cli_writes_crops_and_manifest_in_sorted_order(tmp_path):
    first = place_perspective_slide(tmp_path, "b.jpg", noisy=False)
    second = place_perspective_slide(tmp_path, "a.jpg", noisy=False)
    output_dir = tmp_path / "dir-crops"

    result = run_crop(tmp_path, output_dir, "--max-output-width", "700", "--fail-on-miss")

    assert result.returncode == 0, result.stderr
    manifest = load_manifest(output_dir)
    assert manifest["count"] == 2
    assert manifest["ok_count"] == 2
    assert [Path(item["source"]).name for item in manifest["items"]] == [second.name, first.name]
    for item in manifest["items"]:
        assert Path(item["crop_path"]).exists()
        assert item["quad"] and len(item["quad"]) == 4
        assert item["output_size"][0] <= 700
        assert math.isfinite(item["score"])
