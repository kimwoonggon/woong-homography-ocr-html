---
name: homography-ocr-html
description: Crop photographed presentation slides into rectified images with OpenCV homography, then use Codex vision OCR to turn the slide crops into one local HTML document. Use when working from phone or camera photos of presentation screens, boards, projector slides, or other text-heavy rectangular slide regions.
---

# Homography OCR HTML

## Workflow

1. Confirm the input images are local and should remain uncommitted. Do not stage or commit source photos, crop outputs, debug overlays, manifests, or generated HTML.
2. Install runtime dependencies when needed:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Crop a file or directory of slide photos. The cropper follows the same shape as "straightening a tilted business card": detect the four slide corners in top-left, top-right, bottom-right, bottom-left order, map those points to a flat rectangle, then warp the image with homography.

   ```bash
   python scripts/crop_slides.py INPUT --output crops --debug --max-output-width 1800 --fail-on-miss
   ```

4. Inspect `crops/manifest.json`. Re-run without `--fail-on-miss` if partial output is useful, or inspect `crops/debug/*_overlay.png` for misses.
5. Use Codex vision on the generated `*_crop.png` files, in manifest order, to transcribe slide content.
6. Write a single local HTML document from the OCR result. Keep the HTML local unless the user explicitly asks for a different destination and it is safe to publish.

## Crop Tool

Use the bundled skill script `scripts/crop_slides.py` for deterministic preprocessing before OCR. This repository root is the skill folder, so `scripts/crop_slides.py` is part of the skill and should be invoked by Codex when this skill triggers. `INPUT` can be an image file or a directory. Directories are processed by sorted image filename using these extensions: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tif`, `.tiff`.

Bundled resource:

- `scripts/crop_slides.py`: OpenCV/Numpy CLI that detects a text-heavy rectangular slide region, orders its four corners, computes a homography, warps it to a flat crop, and writes a manifest for OCR.

The tool writes:

- `*_crop.png` for each successful crop
- `manifest.json` with `source`, `crop_path`, `status`, `score`, `quad`, `output_size`, `method`, and `message`
- `debug/*_overlay.png` and `debug/*_mask.png` when `--debug` is enabled

The detector intentionally combines multiple signals: resized image processing, edge masks, brightness masks, text-density masks, contour quadrilateral approximation, convex-hull quadrilateral approximation, rotated-rectangle fallback candidates, candidate scoring, and `cv2.findHomography` / `cv2.warpPerspective`. Treat `min-area-rect` results as a fallback; prefer `contour-quad` or `hull-quad` methods because they preserve real perspective corners before warping.

## OCR And HTML

Use the crop images as the visual source of truth. Preserve slide order from `manifest.json`. For each slide:

- Extract visible headings, bullets, labels, table text, and meaningful diagrams.
- Mark unreadable regions as `[unreadable]` instead of guessing.
- Prefer semantic HTML (`section`, `h1`-`h3`, `ul`, `ol`, `table`, `figure`) over a visual clone.
- Keep generated HTML out of Git by default.

## Git Hygiene

Before committing, run:

```bash
git status --short
git check-ignore -v images/*.jpg crops/*.png crops/debug/*.png 2>/dev/null || true
```

Only commit skill code, tests, dependency files, and `.gitignore`. Do not commit local photos, processed crop images, debug images, OCR manifests, or generated HTML.
