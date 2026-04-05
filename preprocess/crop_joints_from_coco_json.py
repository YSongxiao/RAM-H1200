import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop ROI patches from a prediction COCO JSON and save them as BMP files."
    )
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory containing the source images.")
    parser.add_argument("--coco-json", type=Path, required=True, help="Joints Detection COCO JSON file.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/combined_joints/rois"))
    return parser.parse_args()


def clamp_bbox_to_image(bbox: List[float], width: int, height: int) -> List[int]:
    x, y, w, h = bbox
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(width, int(round(x + w)))
    y2 = min(height, int(round(y + h)))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return [x1, y1, x2, y2]


def get_image_folder_name(image_meta: Dict) -> str:
    extra = image_meta.get("extra", {})
    original_name = extra.get("name")
    if original_name:
        return Path(original_name).stem
    return Path(image_meta["file_name"]).stem


def main() -> None:
    args = parse_args()
    coco = json.loads(args.coco_json.read_text(encoding="utf-8"))

    categories = {item["id"]: item["name"] for item in coco["categories"]}
    images = {item["id"]: item for item in coco["images"]}
    ann_by_image: Dict[int, List[Dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_image[ann["image_id"]].append(ann)

    for image_id, image_meta in images.items():
        image_path = args.image_dir / image_meta["file_name"]
        if not image_path.exists():
            print(f"Image file not found, skipped: {image_path}")
            continue

        image = Image.open(image_path).convert("L")
        image_width, image_height = image.size

        image_folder = args.output_dir / get_image_folder_name(image_meta)
        image_folder.mkdir(parents=True, exist_ok=True)

        name_counts: Dict[str, int] = defaultdict(int)
        for ann in ann_by_image.get(image_id, []):
            category_name = categories[ann["category_id"]]
            x1, y1, x2, y2 = clamp_bbox_to_image(ann["bbox"], image_width, image_height)
            roi = image.crop((x1, y1, x2, y2))

            name_counts[category_name] += 1
            suffix = "" if name_counts[category_name] == 1 else f"_{name_counts[category_name]}"
            output_path = image_folder / f"{category_name}{suffix}.bmp"
            roi.save(output_path)

    print(f"Saved ROI crops to: {args.output_dir}")


if __name__ == "__main__":
    main()
