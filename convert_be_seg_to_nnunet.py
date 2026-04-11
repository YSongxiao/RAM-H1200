#!/usr/bin/env python3
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


SOURCE_DATA_ROOT = Path("/home/yafei/data/RAM-H1200/Segmentation")
NNUNET_ROOT = Path("/home/yafei/code/RAM-H1200/models/nnUNet")
DATASET_ID = 120
DATASET_NAME = "RAMH1200BESeg"
OVERWRITE = False
LABEL_MAP = {
    "SvdH-BE-90": 1,
    "SvdH-BE-50": 2,
    "Non-SvdH-BE": 3,
}


def dataset_folder_name(dataset_id: int, dataset_name: str) -> str:
    return f"Dataset{dataset_id:03d}_{dataset_name}"


def require_existing_dir(path: Path):
    if not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")


def prepare_output_dir(path: Path, overwrite: bool):
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output dataset already exists: {path}. Use --overwrite to replace it.")
        shutil.rmtree(path)
    (path / "imagesTr").mkdir(parents=True, exist_ok=True)
    (path / "labelsTr").mkdir(parents=True, exist_ok=True)
    (path / "imagesTs").mkdir(parents=True, exist_ok=True)
    (path / "tmp_labelsTs").mkdir(parents=True, exist_ok=True)


def load_grayscale_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def save_png_image(image: np.ndarray, output_path: Path):
    if image.dtype == np.uint8:
        Image.fromarray(image, mode="L").save(output_path)
        return
    if image.dtype == np.uint16:
        Image.fromarray(image, mode="I;16").save(output_path)
        return

    image = image.astype(np.float32)
    vmin = float(image.min())
    vmax = float(image.max())
    if vmax > vmin:
        image = ((image - vmin) / (vmax - vmin) * 255.0).round().astype(np.uint8)
    else:
        image = np.zeros_like(image, dtype=np.uint8)
    Image.fromarray(image, mode="L").save(output_path)


def decode_annotation_mask(annotation: dict, height: int, width: int) -> np.ndarray:
    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
        decoded = mask_utils.decode(rle)
        if decoded.ndim == 3:
            decoded = np.any(decoded, axis=2)
        return decoded.astype(bool)

    if isinstance(segmentation, list):
        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in segmentation:
            if len(polygon) < 6:
                continue
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            points = np.round(points).astype(np.int32)
            cv2.fillPoly(mask, [points], 1)
        return mask.astype(bool)

    raise ValueError(f"Unsupported segmentation type: {type(segmentation).__name__}")


def load_coco(coco_path: Path) -> dict:
    with coco_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_label_maps(coco: dict, split_root: Path, label_map: dict[str, int]):
    categories = {int(cat["id"]): cat["name"] for cat in coco.get("categories", [])}
    target_cat_ids = {cat_id: label_map[name] for cat_id, name in categories.items() if name in label_map}
    if not target_cat_ids:
        raise ValueError(f"No target BE categories found. Available categories: {categories}")

    annotations_by_image_id = {}
    for annotation in coco.get("annotations", []):
        cat_id = int(annotation["category_id"])
        if cat_id in target_cat_ids:
            annotations_by_image_id.setdefault(int(annotation["image_id"]), []).append(annotation)

    label_maps = {}
    image_infos = []
    for image_info in coco.get("images", []):
        image_id = int(image_info["id"])
        filename = image_info["file_name"]
        height = int(image_info["height"])
        width = int(image_info["width"])
        image_path = split_root / filename
        if not image_path.is_file():
            raise FileNotFoundError(f"Image listed in COCO but missing on disk: {image_path}")

        label = np.zeros((height, width), dtype=np.uint8)
        # Lower priority is written first; higher priority can overwrite overlap.
        annotations = sorted(
            annotations_by_image_id.get(image_id, []),
            key=lambda ann: target_cat_ids[int(ann["category_id"])],
            reverse=True,
        )
        for annotation in annotations:
            class_value = target_cat_ids[int(annotation["category_id"])]
            mask = decode_annotation_mask(annotation, height, width)
            label[mask] = class_value

        case_id = Path(filename).stem
        label_maps[case_id] = label
        image_infos.append((case_id, image_path))

    return image_infos, label_maps


def convert_split(split_name: str, split_root: Path, output_images: Path, output_labels: Path | None):
    coco_path = split_root / "_annotations_be_rle.coco.json"
    coco = load_coco(coco_path)
    image_infos, label_maps = build_label_maps(coco, split_root, LABEL_MAP)

    case_ids = []
    mapping_rows = []
    for case_id, image_path in image_infos:
        case_ids.append(case_id)
        nnunet_image_name = f"{case_id}_0000.png"
        nnunet_label_name = f"{case_id}.png" if output_labels is not None else ""
        image = load_grayscale_image(image_path)
        save_png_image(image, output_images / nnunet_image_name)
        if output_labels is not None:
            label = label_maps[case_id]
            Image.fromarray(label, mode="L").save(output_labels / nnunet_label_name)
        mapping_rows.append(
            {
                "split": split_name,
                "case_id": case_id,
                "source_image_path": str(image_path),
                "source_image_name": image_path.name,
                "nnunet_image_name": nnunet_image_name,
                "nnunet_label_name": nnunet_label_name,
            }
        )
    return case_ids, mapping_rows


def write_dataset_json(dataset_dir: Path, num_training: int):
    payload = {
        "channel_names": {"0": "CR"},
        "labels": {
            "background": 0,
            "SvdH-BE-90": 1,
            "SvdH-BE-50": 2,
            "Non-SvdH-BE": 3,
        },
        "numTraining": int(num_training),
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }
    with (dataset_dir / "dataset.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def write_split_files(nnunet_root: Path, dataset_name: str, train_cases: list[str], val_cases: list[str]):
    splits = [{"train": sorted(train_cases), "val": sorted(val_cases)}]
    raw_split_path = nnunet_root / "DATASET" / "nnUNet_raw" / dataset_name / "splits_final.json"
    with raw_split_path.open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=4)

    preprocessed_dataset_dir = nnunet_root / "DATASET" / "nnUNet_preprocessed" / dataset_name
    preprocessed_dataset_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_split_path = preprocessed_dataset_dir / "splits_final.json"
    with preprocessed_split_path.open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=4)
    return raw_split_path, preprocessed_split_path


def write_case_mapping(dataset_dir: Path, mapping_rows: list[dict]):
    mapping_json_path = dataset_dir / "case_mapping.json"
    mapping_csv_path = dataset_dir / "case_mapping.csv"

    with mapping_json_path.open("w", encoding="utf-8") as f:
        json.dump(mapping_rows, f, indent=4, ensure_ascii=False)

    columns = [
        "split",
        "case_id",
        "source_image_path",
        "source_image_name",
        "nnunet_image_name",
        "nnunet_label_name",
    ]
    with mapping_csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for row in mapping_rows:
            f.write(",".join(str(row.get(column, "")).replace(",", ";") for column in columns) + "\n")
    return mapping_json_path, mapping_csv_path


def validate_unique_cases(train_cases: list[str], val_cases: list[str], test_cases: list[str]):
    train_set = set(train_cases)
    val_set = set(val_cases)
    test_set = set(test_cases)
    if len(train_set) != len(train_cases):
        raise ValueError("Duplicate case ids found in train split.")
    if len(val_set) != len(val_cases):
        raise ValueError("Duplicate case ids found in val split.")
    overlap = train_set & val_set
    if overlap:
        raise ValueError(f"Train/val case id overlap: {sorted(overlap)[:10]}")
    test_overlap = (train_set | val_set) & test_set
    if test_overlap:
        print(f"[WARN] Test case ids overlap with train/val: {sorted(test_overlap)[:10]}")


def print_next_commands(nnunet_root: Path, dataset_id: int, dataset_name: str):
    data_root = nnunet_root / "DATASET"
    print("\nConversion finished.")
    print(f"Dataset: {dataset_name}")
    print("\nUse these environment variables before running nnU-Net:")
    print(f'export nnUNet_raw="{data_root / "nnUNet_raw"}"')
    print(f'export nnUNet_preprocessed="{data_root / "nnUNet_preprocessed"}"')
    print(f'export nnUNet_results="{data_root / "nnUNet_trained_models"}"')
    print("\nThen run:")
    print(f"nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity")
    print(f"nnUNetv2_train {dataset_id} 2d 0 -tr nnUNetTrainerBE")
    print(
        "After preprocessing, if nnU-Net overwrites splits, copy the generated raw "
        f"splits_final.json into {data_root / 'nnUNet_preprocessed' / dataset_name / 'splits_final.json'}."
    )


def main():
    nnunet_root = NNUNET_ROOT.resolve()
    source_root = SOURCE_DATA_ROOT.resolve()
    require_existing_dir(nnunet_root)
    require_existing_dir(source_root)
    for split in ("train", "val", "test"):
        require_existing_dir(source_root / split)

    dataset_name = dataset_folder_name(DATASET_ID, DATASET_NAME)
    dataset_dir = nnunet_root / "DATASET" / "nnUNet_raw" / dataset_name
    prepare_output_dir(dataset_dir, overwrite=OVERWRITE)
    (nnunet_root / "DATASET" / "nnUNet_preprocessed").mkdir(parents=True, exist_ok=True)
    (nnunet_root / "DATASET" / "nnUNet_trained_models").mkdir(parents=True, exist_ok=True)

    train_cases, train_mapping = convert_split("train", source_root / "train", dataset_dir / "imagesTr", dataset_dir / "labelsTr")
    val_cases, val_mapping = convert_split("val", source_root / "val", dataset_dir / "imagesTr", dataset_dir / "labelsTr")
    test_cases, test_mapping = convert_split("test", source_root / "test", dataset_dir / "imagesTs", dataset_dir / "tmp_labelsTs")
    validate_unique_cases(train_cases, val_cases, test_cases)

    write_dataset_json(dataset_dir, num_training=len(train_cases) + len(val_cases))
    raw_split_path, preprocessed_split_path = write_split_files(nnunet_root, dataset_name, train_cases, val_cases)
    mapping_json_path, mapping_csv_path = write_case_mapping(dataset_dir, train_mapping + val_mapping + test_mapping)

    print(f"Train cases: {len(train_cases)}")
    print(f"Val cases: {len(val_cases)}")
    print(f"Test cases: {len(test_cases)}")
    print(f"Raw dataset dir: {dataset_dir}")
    print(f"Raw split file: {raw_split_path}")
    print(f"Preprocessed split file: {preprocessed_split_path}")
    print(f"Case mapping json: {mapping_json_path}")
    print(f"Case mapping csv: {mapping_csv_path}")
    print_next_commands(nnunet_root, DATASET_ID, dataset_name)


if __name__ == "__main__":
    main()
