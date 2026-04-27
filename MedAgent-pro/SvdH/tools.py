import json
import os

from PIL import Image, ImageStat


def _write_json(save_dir, save_name, payload):
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, save_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
    return output_path


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_patch_metadata(image_path, save_dir, save_name):
    """Extract patch metadata from a local SvdH joint patch image path and write JSON."""
    with Image.open(image_path) as image:
        width, height = image.size
        mode = image.mode

    patch_name = os.path.basename(image_path)
    site = os.path.splitext(patch_name)[0]
    folder_name = os.path.basename(os.path.dirname(image_path))
    image_name = f"{folder_name}.bmp"

    payload = {
        "image_name": image_name,
        "site": site,
        "patch_path": image_path,
        "width": width,
        "height": height,
        "mode": mode,
    }
    return _write_json(save_dir, save_name, payload)


def compute_patch_image_statistics(image_path, save_dir, save_name):
    """Compute simple image intensity statistics for a local SvdH patch and write JSON."""
    with Image.open(image_path) as image:
        gray = image.convert("L")
        stat = ImageStat.Stat(gray)
        width, height = gray.size
        pixels = sorted(gray.getdata())

    def percentile(q):
        if not pixels:
            return None
        idx = min(len(pixels) - 1, max(0, int(round(q * (len(pixels) - 1)))))
        return int(pixels[idx])

    payload = {
        "width": width,
        "height": height,
        "mean_intensity": float(stat.mean[0]),
        "std_intensity": float(stat.stddev[0]),
        "min_intensity": int(pixels[0]) if pixels else None,
        "max_intensity": int(pixels[-1]) if pixels else None,
        "p05_intensity": percentile(0.05),
        "p50_intensity": percentile(0.50),
        "p95_intensity": percentile(0.95),
    }
    return _write_json(save_dir, save_name, payload)


def build_svdh_scoring_context(inputs, save_dir, save_name):
    """Build SvdH bone erosion scoring context from prior tool JSON files and write JSON."""
    metadata = _read_json(inputs[0]) if inputs else {}
    statistics = _read_json(inputs[1]) if len(inputs) > 1 else {}
    site = metadata.get("site", "unknown")

    payload = {
        "site": site,
        "metadata": metadata,
        "image_statistics": statistics,
        "scoring_rule": (
            "Score bone erosion only for this single cropped anatomical site under the "
            "Sharp/van der Heijde SvdH BE system. Use 0 for no visible erosion, 1 for minimal erosion, "
            "2 for mild erosion, 3 for moderate erosion, 4 for marked erosion, and 5 for severe erosion. "
            "Do not score joint space narrowing, osteophytes, soft tissue swelling, or global disease activity."
        ),
        "required_output_schema": {"score": 0, "reason": "short visual reason"},
    }
    return _write_json(save_dir, save_name, payload)
