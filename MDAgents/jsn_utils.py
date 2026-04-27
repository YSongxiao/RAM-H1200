import json
import os
import re
import time

from utils import (
    balanced_accuracy_percent,
    determine_difficulty,
    positive_negative_accuracy_percent,
    positive_negative_sensitivity_percent,
    process_advanced_query,
    process_basic_query,
    process_intermediate_query,
    quadratic_weighted_kappa,
)


def load_svdh_jsn_data(svdh_root, split="test", sites=None):
    svdh_root = os.path.expanduser(svdh_root)
    split_dir = os.path.join(svdh_root, split)
    annotation_path = os.path.join(split_dir, "_annotation_jsn_scores.json")
    if not os.path.isfile(annotation_path):
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    with open(annotation_path, "r", encoding="utf-8") as file:
        annotations = json.load(file)

    requested_sites = set(sites) if sites else None
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


def create_svdh_jsn_question(sample):
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


def extract_svdh_jsn_score(response):
    match = re.search(r'"score"\s*:\s*"?([0-4])"?', response or "")
    if match:
        return int(match.group(1))
    match = re.search(r'\b(?:score|answer)\s*[:=]?\s*([0-4])\b', response or "", re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def response_to_text(response):
    if isinstance(response, dict):
        return "\n".join(response_to_text(value) for value in response.values())
    if isinstance(response, (list, tuple)):
        return "\n".join(response_to_text(value) for value in response)
    return str(response)


def process_svdh_jsn_mdagents_query(sample, model, difficulty="adaptive"):
    class SvdHJSNArgs:
        dataset = "svdh_jsn"
        # Reuse the existing image-scoring branches in shared utils without touching BE code.
        task = "svdh_be"

    question = create_svdh_jsn_question(sample)
    start_time = time.perf_counter()
    selected_difficulty, difficulty_trace = determine_difficulty(question, difficulty, return_trace=True)

    if selected_difficulty == "basic":
        response, workflow_trace = process_basic_query(
            question, [], model, SvdHJSNArgs(), img_path=sample["patch_path"], return_trace=True
        )
    elif selected_difficulty == "intermediate":
        response, workflow_trace = process_intermediate_query(
            question, [], model, SvdHJSNArgs(), img_path=sample["patch_path"], return_trace=True
        )
    elif selected_difficulty == "advanced":
        response = process_advanced_query(question, model, SvdHJSNArgs(), img_path=sample["patch_path"])
        workflow_trace = {
            "mode": "advanced",
            "note": "Detailed trace capture is currently implemented for basic and intermediate workflows.",
        }
    else:
        raise ValueError(f"Unsupported SvdH JSN difficulty: {selected_difficulty}")

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    response_text = response_to_text(response)
    return {
        "difficulty": selected_difficulty,
        "question": question,
        "prediction": extract_svdh_jsn_score(response_text),
        "response": response,
        "time_ms": elapsed_ms,
        "trace": {
            "difficulty_selection": difficulty_trace,
            "workflow": workflow_trace,
        },
    }


def compute_svdh_jsn_metrics(results):
    scored = [result for result in results if result.get("prediction") is not None]
    if not scored:
        return {
            "num_scored": 0,
            "qwk": None,
            "mae": None,
            "bacc_percent": 0,
            "acc_percent": 0,
            "w1_acc_percent": 0,
            "p_n_sen_percent": 0,
            "p_n_acc_percent": 0,
            "params_m": None,
            "time_ms": None,
            "exact_accuracy": 0,
            "within_1_accuracy": 0,
            "per_site": {},
        }

    exact = sum(1 for result in scored if result["prediction"] == result["label"])
    abs_errors = [abs(result["prediction"] - result["label"]) for result in scored]
    labels = [result["label"] for result in scored]
    predictions = [result["prediction"] for result in scored]
    time_values = [result["time_ms"] for result in scored if result.get("time_ms") is not None]
    acc = exact / len(scored)
    within_1_acc = sum(1 for error in abs_errors if error <= 1) / len(scored)
    per_site = {}
    for result in scored:
        site = result["site"]
        per_site.setdefault(site, {"count": 0, "exact": 0, "abs_error": 0})
        per_site[site]["count"] += 1
        per_site[site]["exact"] += int(result["prediction"] == result["label"])
        per_site[site]["abs_error"] += abs(result["prediction"] - result["label"])

    for site, values in per_site.items():
        values["exact_accuracy"] = values["exact"] / values["count"]
        values["mae"] = values["abs_error"] / values["count"]

    return {
        "num_scored": len(scored),
        "qwk": quadratic_weighted_kappa(labels, predictions, min_rating=0, max_rating=4),
        "mae": sum(abs_errors) / len(scored),
        "bacc_percent": balanced_accuracy_percent(labels, predictions, classes=range(5)),
        "acc_percent": acc * 100,
        "w1_acc_percent": within_1_acc * 100,
        "p_n_sen_percent": positive_negative_sensitivity_percent(labels, predictions),
        "p_n_acc_percent": positive_negative_accuracy_percent(labels, predictions),
        "params_m": None,
        "time_ms": sum(time_values) / len(time_values) if time_values else None,
        "exact_accuracy": acc,
        "within_1_accuracy": within_1_acc,
        "per_site": per_site,
    }
