# utils.py
import os
import re
import json
from importlib import import_module, reload
from collections import Counter

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp")

CONCISE_SYSTEM_RULE = (
    "Keep responses concise. Use the minimum text needed. "
    "Do not provide long step-by-step analysis unless explicitly required."
)

CONCISE_USER_RULE = (
    "Keep the answer brief. No long analysis. Use the minimum text needed."
)

SHORT_REASON_RULE = (
    "If a reason is requested, keep it to one short sentence or short phrase."
)


def concise_system(prompt: str) -> str:
    return f"{prompt} {CONCISE_SYSTEM_RULE}"


def concise_user(prompt: str, extra: str | None = None) -> str:
    suffix = CONCISE_USER_RULE if extra is None else f"{CONCISE_USER_RULE} {extra}"
    return f"{prompt}\n\n{suffix}"


# ---------- package / registry helpers ----------

def ensure_pkg_inited(data_root: str):
    """Ensure {data_root}/ and {data_root}/tools/ are importable packages."""
    for pkg_dir in [data_root, os.path.join(data_root, "tools")]:
        os.makedirs(pkg_dir, exist_ok=True)
        init_py = os.path.join(pkg_dir, "__init__.py")
        if not os.path.exists(init_py):
            with open(init_py, "a", encoding="utf-8"):
                pass  # touch


def register_generated_function(data_root: str, registry: dict, fn_name: str):
    """
    Reload {data_root}.tools.GenCode and register the new function object into registry.
    Example: register_generated_function("Glaucoma", TOOL_FN_REGISTRY, "compute_cdr_6")
    """
    module_name = f"{data_root}.tools.GenCode"
    mod = import_module(module_name) if module_name not in globals() else import_module(module_name)
    mod = reload(mod)
    if hasattr(mod, fn_name):
        registry[fn_name] = getattr(mod, fn_name)
        print(f"[registry] registered {fn_name} -> TOOL_FN_REGISTRY")
    else:
        print(f"[warn] {fn_name} not found in {module_name} after reload")


def command_to_fn_name(command: str) -> str:
    """Extract function name from a string like 'segment_optic_disc()' -> 'segment_optic_disc'."""
    if not command:
        return ""
    s = command.strip()
    return s.split("(", 1)[0].strip() if "(" in s else s


# ---------- naming / prompt helpers ----------

def snake(s: str, fallback: str = "generated_fn") -> str:
    """Convert arbitrary text to snake_case."""
    s = re.sub(r"[^0-9a-zA-Z]+", " ", str(s or "")).strip().lower()
    s = "_".join(w for w in s.split() if w)
    return s or fallback


def inputs_desc(step: dict, plan_by_id: dict, tool_by_id: dict, task_input_desc: str):
    """
    Resolve step['input_type'] into a LIST of human-readable descriptors.
    Priority: previous step's tool.output -> fallback previous step's output_type -> task input.
    """
    deps = step.get("input_type", []) or []
    descs = []
    for dep in deps:
        try:
            dep = int(dep)
        except Exception:
            continue

        if dep == 0:
            descs.append(str(task_input_desc).strip())
            continue

        prev = plan_by_id.get(dep)
        if not prev:
            descs.append(f"[missing step {dep}]")
            continue

        tids = prev.get("tool", []) or []
        if not isinstance(tids, list):
            tids = [tids]
        outs = [str(tool_by_id.get(int(tid), {}).get("output", "")).strip() for tid in tids]
        fallback = str(prev.get("output_type", "")).strip()
        descs.append(" / ".join([o for o in outs if o]) or fallback)

    return [d for d in descs if d]


# ---------- I/O helpers for qualitative steps ----------

def json_to_text(value, max_chars: int = 1200) -> str:
    """Convert any JSON value (str/num/bool/dict/list) to compact text for prompts."""
    if isinstance(value, str):
        s = value.strip()
    else:
        s = json.dumps(value, ensure_ascii=False)
    return s if len(s) <= max_chars else (s[:max_chars] + " …[truncated]")


def read_prev_output(save_dir: str, filename: str, dep_id: int):
    """
    Return (text, image_path). JSON -> prefer data['step_<dep_id>'] else whole file.
    Image -> (None, path). Text file -> (text, None). Missing -> (None, None).
    """
    if not filename:
        return None, None
    path = os.path.join(save_dir, filename)
    if not os.path.exists(path):
        return None, None

    low = filename.lower()
    if low.endswith(IMAGE_EXTS):
        return None, path
    if low.endswith(".json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            key = f"step_{dep_id}"
            val = data.get(key, data)
            return json_to_text(val), None
        except Exception as e:
            return f"[error reading {filename}: {e}]", None
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip(), None
    except Exception as e:
        return f"[error reading {filename}: {e}]", None


def build_qual_prompt(base_question: str, texts: list[str]) -> str:
    """Attach upstream texts as bullet-point context."""
    if not texts:
        return base_question
    bullets = "\n".join(f"- {t}" for t in texts if t)
    return f"{base_question}\n\nContext:\n{bullets}"


# ---------- SvdH bone erosion scoring helpers ----------

def load_svdh_be_data(svdh_root: str, split: str = "test", sites: list[str] | None = None):
    svdh_root = os.path.expanduser(svdh_root)
    split_dir = os.path.join(svdh_root, split)
    annotation_path = os.path.join(split_dir, "_annotation_be_scores.json")
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
            site = score_key.replace("BE_", "", 1)
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


def create_svdh_be_prompt(sample):
    return (
        "Task: Sharp/van der Heijde (SvdH) bone erosion scoring for rheumatoid arthritis hand/wrist X-ray patches.\n"
        f"Image: {sample['image_name']}; local anatomical site: {sample['site']}.\n"
        "Score this single cropped patch for bone erosion only. Do not score joint space narrowing, "
        "osteophytes, soft tissue swelling, or global disease activity.\n"
        "Options: (0) no visible erosion (1) minimal erosion (2) mild erosion "
        "(3) moderate erosion (4) marked erosion (5) severe erosion.\n"
        "Return the final answer as JSON in this exact schema: {\"score\":0,\"reason\":\"short visual reason\"}.\n"
        "Keep the reason extremely short. Do not provide long analysis."
    )


def extract_svdh_score(text):
    match = re.search(r'"score"\s*:\s*"?([0-5])"?', text or "")
    if match:
        return int(match.group(1))
    match = re.search(r'\b(?:score|answer)\s*[:=]?\s*([0-5])\b', text or "", re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def quadratic_weighted_kappa(labels, predictions, min_rating=0, max_rating=5):
    n = max_rating - min_rating + 1
    observed = [[0 for _ in range(n)] for _ in range(n)]
    for label, prediction in zip(labels, predictions):
        observed[int(label) - min_rating][int(prediction) - min_rating] += 1

    label_counts = Counter(int(label) for label in labels)
    prediction_counts = Counter(int(prediction) for prediction in predictions)
    expected = [[0 for _ in range(n)] for _ in range(n)]
    total = len(labels)
    for i in range(n):
        for j in range(n):
            expected[i][j] = label_counts[i + min_rating] * prediction_counts[j + min_rating] / total if total else 0

    numerator = 0.0
    denominator = 0.0
    for i in range(n):
        for j in range(n):
            weight = ((i - j) ** 2) / ((n - 1) ** 2)
            numerator += weight * observed[i][j]
            denominator += weight * expected[i][j]
    return 1 - numerator / denominator if denominator else 1.0


def balanced_accuracy_percent(labels, predictions):
    recalls = []
    for cls in sorted(set(labels)):
        total = sum(1 for label in labels if label == cls)
        if total:
            correct = sum(1 for label, prediction in zip(labels, predictions) if label == cls and prediction == cls)
            recalls.append(correct / total)
    return (sum(recalls) / len(recalls) * 100) if recalls else 0.0


def positive_negative_sensitivity_percent(labels, predictions):
    binary_labels = [int(label > 0) for label in labels]
    binary_predictions = [int(prediction > 0) for prediction in predictions]
    sensitivities = []
    for cls in (1, 0):
        total = sum(1 for label in binary_labels if label == cls)
        if total:
            correct = sum(1 for label, prediction in zip(binary_labels, binary_predictions) if label == cls and prediction == cls)
            sensitivities.append(correct / total)
    return (sum(sensitivities) / len(sensitivities) * 100) if sensitivities else 0.0


def positive_negative_accuracy_percent(labels, predictions):
    binary_labels = [int(label > 0) for label in labels]
    binary_predictions = [int(prediction > 0) for prediction in predictions]
    if not binary_labels:
        return 0.0
    correct = sum(1 for label, prediction in zip(binary_labels, binary_predictions) if label == prediction)
    return correct / len(binary_labels) * 100


def compute_svdh_metrics(results):
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
        "QWK": quadratic_weighted_kappa(labels, predictions),
        "MAE": mae,
        "BACC (%)": balanced_accuracy_percent(labels, predictions),
        "ACC (%)": correct / total * 100,
        "W1-ACC (%)": within_one / total * 100,
        "P/N-SEN (%)": positive_negative_sensitivity_percent(labels, predictions),
        "P/N-ACC (%)": positive_negative_accuracy_percent(labels, predictions),
        "Params (M)": None,
        "Time (ms)": sum(times) / len(times) if times else None,
    }
