import json
import os
import re

from utils import (
    balanced_accuracy_percent,
    positive_negative_accuracy_percent,
    positive_negative_sensitivity_percent,
    quadratic_weighted_kappa,
)


def load_svdh_jsn_data(svdh_root: str, split: str = "test", sites: list[str] | None = None):
    svdh_root = os.path.expanduser(svdh_root)
    split_dir = os.path.join(svdh_root, split)
    annotation_path = os.path.join(split_dir, "_annotation_jsn_scores.json")
    if not os.path.isfile(annotation_path):
        raise FileNotFoundError(f"SvdH annotation file not found: {annotation_path}")

    with open(annotation_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    requested_sites = set(sites or [])
    samples = []
    for image_name in sorted(annotations):
        patch_dir = os.path.join(split_dir, os.path.splitext(image_name)[0])
        if not os.path.isdir(patch_dir):
            raise FileNotFoundError(f"Patch folder not found: {patch_dir}")

        for score_key, label in sorted(annotations[image_name].items()):
            if not str(score_key).startswith("JSN_"):
                continue

            site = score_key.replace("JSN_", "", 1)
            if requested_sites and site not in requested_sites:
                continue

            patch_path = os.path.join(patch_dir, f"{site}.bmp")
            if not os.path.isfile(patch_path):
                raise FileNotFoundError(f"Patch file not found for {image_name} {site}: {patch_path}")

            samples.append({
                "split": split,
                "image_name": image_name,
                "patch_dir": patch_dir,
                "site": site,
                "score_key": score_key,
                "patch_path": patch_path,
                "label": int(label),
            })

    return samples


def create_svdh_jsn_prompt(sample):
    return (
        "Task: Sharp/van der Heijde (SvdH) joint space narrowing (JSN) scoring for rheumatoid arthritis "
        "hand/wrist X-ray patches.\n"
        f"Image: {sample['image_name']}; local anatomical site: {sample['site']}.\n"
        "Score this single cropped patch for joint space narrowing only. Do not score bone erosion, "
        "osteophytes, soft tissue swelling, or global disease activity.\n"
        "The score must be an integer from 0 to 4.\n"
        "Return the final answer as JSON in this exact schema: {\"score\":0,\"reason\":\"short visual reason\"}.\n"
        "Keep the reason extremely short. Do not provide long analysis."
    )


def extract_svdh_jsn_score(text):
    match = re.search(r'"score"\s*:\s*"?([0-4])"?', text or "")
    if match:
        return int(match.group(1))
    match = re.search(r'\b(?:score|answer)\s*[:=]?\s*([0-4])\b', text or "", re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def compute_svdh_jsn_metrics(results):
    scored = [result for result in results if result.get("prediction") is not None]
    if not scored:
        return {
            "QWK": None,
            "MAE": None,
            "BACC (%)": None,
            "ACC (%)": None,
            "W1-ACC (%)": None,
            "P/N-SEN (%)": None,
            "P/N-ACC (%)": None,
            "Params (M)": None,
            "Time (ms)": None,
        }

    labels = [int(result["label"]) for result in scored]
    predictions = [int(result["prediction"]) for result in scored]
    total = len(scored)
    correct = sum(1 for label, prediction in zip(labels, predictions) if label == prediction)
    within_one = sum(1 for label, prediction in zip(labels, predictions) if abs(label - prediction) <= 1)
    mae = sum(abs(label - prediction) for label, prediction in zip(labels, predictions)) / total
    times = [result.get("time_ms") for result in scored if result.get("time_ms") is not None]

    return {
        "QWK": quadratic_weighted_kappa(labels, predictions, min_rating=0, max_rating=4),
        "MAE": mae,
        "BACC (%)": balanced_accuracy_percent(labels, predictions),
        "ACC (%)": correct / total * 100,
        "W1-ACC (%)": within_one / total * 100,
        "P/N-SEN (%)": positive_negative_sensitivity_percent(labels, predictions),
        "P/N-ACC (%)": positive_negative_accuracy_percent(labels, predictions),
        "Params (M)": None,
        "Time (ms)": sum(times) / len(times) if times else None,
    }
