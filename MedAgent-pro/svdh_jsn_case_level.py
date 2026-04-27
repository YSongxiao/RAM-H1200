import argparse
import json
import os
import random
import re
import time
import traceback

from tqdm import tqdm

from Decider.GPT_Decider import GPT_Decider
from Decider.Pro_Decider_JSN import Pro_Decider_JSN
from SvdH_JSN.tools import (
    build_svdh_jsn_scoring_context,
    compute_patch_image_statistics,
    extract_patch_metadata,
)
from jsn_utils import (
    compute_svdh_jsn_metrics,
    create_svdh_jsn_prompt,
    extract_svdh_jsn_score,
    load_svdh_jsn_data,
)
from utils import (
    build_qual_prompt,
    command_to_fn_name,
    concise_user,
    read_prev_output,
)


TOOL_FN_REGISTRY = {
    "extract_patch_metadata": extract_patch_metadata,
    "compute_patch_image_statistics": compute_patch_image_statistics,
    "build_svdh_jsn_scoring_context": build_svdh_jsn_scoring_context,
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_json_snapshot(output_path, default):
    if not os.path.exists(output_path):
        return default
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def write_json_snapshot(output_path, payload):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
    os.replace(tmp_path, output_path)


def normalize_sites(sites):
    if not sites:
        return None
    return tuple(sorted(site.strip() for site in sites))


def svdh_sample_key(item):
    return (
        item.get("split"),
        item.get("image_name"),
        item.get("site"),
        item.get("score_key"),
        item.get("patch_path"),
    )


def upsert_record(records, key, record, key_fn):
    for index, existing in enumerate(records):
        if key_fn(existing) == key:
            records[index] = record
            return records
    records.append(record)
    return records


def remove_record(records, key, key_fn):
    return [record for record in records if key_fn(record) != key]


def sanitize_case_name(sample):
    stem = os.path.splitext(sample["image_name"])[0]
    raw = f"{sample['split']}_{stem}_{sample['site']}"
    return re.sub(r"[^0-9A-Za-z._-]+", "_", raw)


def resolve_step_inputs(step, plan_by_id, save_dir, image_path):
    resolved = []
    for dep in step.get("input_type", []) or []:
        dep = int(dep)
        if dep == 0:
            resolved.append(image_path)
        else:
            prev = plan_by_id.get(dep)
            if prev:
                resolved.append(os.path.join(save_dir, prev.get("output_path", "")))
    return resolved


def run_quantitative_step(step, tool_by_id, plan_by_id, save_dir, image_path):
    tool_ids = step.get("tool", []) or []
    if not isinstance(tool_ids, list):
        tool_ids = [tool_ids]

    for tid in tool_ids:
        tool = tool_by_id.get(int(tid))
        if tool is None:
            print(f"[warn] tool id {tid} not found in toolset")
            continue

        fn_name = command_to_fn_name(tool.get("command", ""))
        fn = TOOL_FN_REGISTRY.get(fn_name)
        if fn is None:
            print(f"[warn] command '{fn_name}' not registered; add it to TOOL_FN_REGISTRY")
            continue

        resolved = resolve_step_inputs(step, plan_by_id, save_dir, image_path)
        save_name = step.get("output_path", "")
        if fn_name == "build_svdh_jsn_scoring_context":
            fn(resolved, save_dir, save_name)
        else:
            primary = resolved[0] if resolved else image_path
            fn(primary, save_dir, save_name)


def run_qualitative_step(step, plan_by_id, save_dir, image_path, analyzer):
    image_input, text_input = [], []
    for dep in step.get("input_type", []) or []:
        dep = int(dep)
        if dep == 0:
            image_input.append(image_path)
        else:
            prev_step = plan_by_id.get(dep, {})
            prev_save_name = prev_step.get("output_path", "")
            text, image = read_prev_output(save_dir, prev_save_name, dep)
            if image:
                image_input.append(image)
            if text:
                text_input.append(text)

    prompt = build_qual_prompt(step.get("action", ""), text_input)
    prompt += "\n\nReturn ONLY JSON in this exact schema: {\"score\":0,\"evidence\":\"short visual evidence\"}."
    prompt = concise_user(prompt, "Return only compact JSON. Keep the evidence extremely short.")
    return analyzer.decide(
        output_file=os.path.join(save_dir, step.get("output_path", "diagnosis.json")),
        prompt=prompt,
        image_paths=image_input,
        field=f"step_{step.get('id')}",
    )


def load_case_artifacts(save_dir, plan):
    step_outputs = []
    for step in plan:
        output_rel = step.get("output_path", "")
        output_path = os.path.join(save_dir, output_rel)
        artifact = {
            "id": step.get("id"),
            "action_type": step.get("action_type"),
            "action": step.get("action"),
            "output_type": step.get("output_type"),
            "output_path": output_rel,
            "exists": os.path.exists(output_path),
        }
        if artifact["exists"]:
            try:
                artifact["output"] = load_json(output_path)
            except Exception as exc:
                artifact["output_error"] = str(exc)
        step_outputs.append(artifact)

    final_output_path = os.path.join(save_dir, "final_diagnosis.json")
    final_output = None
    if os.path.exists(final_output_path):
        try:
            final_output = load_json(final_output_path)
        except Exception as exc:
            final_output = {"error": str(exc)}

    return {
        "case_output_dir": save_dir,
        "step_outputs": step_outputs,
        "final_output_path": final_output_path,
        "final_output": final_output,
    }


def run_svdh_case(sample, task, plan, toolset, analyzer, decider, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plan_by_id = {int(step["id"]): step for step in plan if "id" in step}
    tool_by_id = {int(tool["id"]): tool for tool in toolset if "id" in tool}
    start_time = time.perf_counter()
    question = create_svdh_jsn_prompt(sample)

    for step in plan:
        action_type = str(step.get("action_type", "")).lower()
        if action_type == "quantitative":
            run_quantitative_step(step, tool_by_id, plan_by_id, save_dir, sample["patch_path"])
        elif action_type == "qualitative":
            run_qualitative_step(step, plan_by_id, save_dir, sample["patch_path"], analyzer)

    indicators = []
    for step in plan:
        if str(step.get("output_type", "")).lower() != "final indicator":
            continue
        output_path = os.path.join(save_dir, step.get("output_path", ""))
        if not os.path.exists(output_path):
            continue
        data = load_json(output_path)
        indicators.append({
            "indicator_name": step.get("action", ""),
            "value": data.get(f"step_{step.get('id')}", data),
        })

    final_prompt = (
        "You are the final MedAgent-Pro decision module.\n"
        f"Task input: {task.get('input', '')}\n"
        f"Goal: {task.get('disease', '')}\n\n"
        f"{question}"
    )
    final = decider.decide_svdh_score(
        output_file=os.path.join(save_dir, "final_diagnosis.json"),
        prompt=final_prompt,
        indicators=indicators,
        image_paths=[sample["patch_path"]],
        field="overall",
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    overall = final.get("overall", {})
    prediction = overall.get("score")
    if prediction is None:
        prediction = extract_svdh_jsn_score(overall.get("raw", ""))

    return {
        "question": question,
        "indicators": indicators,
        "final": overall,
        "prediction": prediction,
        "time_ms": elapsed_ms,
        "trace": {
            "question": question,
            "final_prompt": final_prompt,
            **load_case_artifacts(save_dir, plan),
        },
    }


def build_output(args, task, plan, toolset, results, failures, sites):
    metrics = compute_svdh_jsn_metrics(results)
    return {
        "task": "svdh_jsn",
        "framework": "MedAgent-Pro",
        "workflow": "case_level_tool_calling",
        "model": args.model,
        "data_root": args.data_root,
        "svdh_root": args.svdh_root,
        "split": args.split,
        "sites": sites,
        "random_sample": args.random_sample,
        "seed": args.seed,
        "num_samples": len(results),
        "num_failures": len(failures),
        "task_config": task,
        "plan": plan,
        "toolset": toolset,
        "metrics": metrics,
        "results": results,
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="SvdH_JSN")
    parser.add_argument("--svdh_root", type=str, default=os.path.expanduser("~/data/RAM-H1200/SvdH_Scoring/SvdH_JSN_Scoring"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--random_sample", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sites", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--output_dir", type=str, default="output/svdh_jsn_medagent_pro_case")
    args = parser.parse_args()

    task = load_json(os.path.join(args.data_root, "task.json"))
    plan = load_json(os.path.join(args.data_root, "plan.json"))
    toolset = load_json(os.path.join(args.data_root, "toolset.json"))

    sites = [site.strip() for site in args.sites.split(",")] if args.sites else None
    samples = load_svdh_jsn_data(args.svdh_root, split=args.split, sites=sites)
    if args.random_sample:
        random.seed(args.seed)
        random.shuffle(samples)

    analyzer = GPT_Decider(model=args.model)
    decider = Pro_Decider_JSN(model=args.model)
    os.makedirs(args.output_dir, exist_ok=True)

    output_path = os.path.join(args.output_dir, f"{args.model}_svdh_jsn_{args.split}.json")
    existing_output = load_json_snapshot(output_path, {})
    if (
        isinstance(existing_output, dict)
        and existing_output.get("task") == "svdh_jsn"
        and existing_output.get("framework") == "MedAgent-Pro"
        and existing_output.get("workflow") == "case_level_tool_calling"
        and existing_output.get("model") == args.model
        and existing_output.get("data_root") == args.data_root
        and existing_output.get("svdh_root") == args.svdh_root
        and existing_output.get("split") == args.split
        and normalize_sites(existing_output.get("sites")) == normalize_sites(sites)
        and existing_output.get("random_sample") == args.random_sample
        and existing_output.get("seed") == args.seed
    ):
        results = list(existing_output.get("results", []))
        failures = list(existing_output.get("failures", []))
    else:
        results = []
        failures = []

    processed_keys = {svdh_sample_key(result) for result in results}
    pending_samples = [sample for sample in samples if svdh_sample_key(sample) not in processed_keys]

    if results:
        print(f"[INFO] resume: found {len(results)} saved MedAgent-Pro samples in {output_path}")
    if failures:
        print(f"[INFO] resume: found {len(failures)} logged failures in {output_path}; failed samples will be retried")

    for sample in tqdm(pending_samples, initial=0, total=len(pending_samples)):
        if len(results) >= args.num_samples:
            break

        print(f"\n[INFO] no: {len(results)}")
        print(f"image: {sample['image_name']}, site: {sample['site']}")
        print(f"patch: {sample['patch_path']}")

        sample_key = svdh_sample_key(sample)
        case_name = sanitize_case_name(sample)
        case_output_dir = os.path.join(args.output_dir, case_name)

        try:
            final_decision = run_svdh_case(sample, task, plan, toolset, analyzer, decider, case_output_dir)
            prediction = final_decision["prediction"]
            print(f"label: {sample['label']}, prediction: {prediction}")

            results.append({
                "split": sample["split"],
                "image_name": sample["image_name"],
                "site": sample["site"],
                "score_key": sample["score_key"],
                "patch_path": sample["patch_path"],
                "question": final_decision.get("question"),
                "label": sample["label"],
                "prediction": prediction,
                "response": final_decision,
                "time_ms": final_decision.get("time_ms"),
                "diagnostic_trace": final_decision.get("trace"),
                "correct": prediction == sample["label"],
                "abs_error": abs(prediction - sample["label"]) if prediction is not None else None,
            })
            processed_keys.add(sample_key)
            failures = remove_record(failures, sample_key, svdh_sample_key)
        except Exception as exc:
            error_record = {
                "split": sample["split"],
                "image_name": sample["image_name"],
                "site": sample["site"],
                "score_key": sample["score_key"],
                "patch_path": sample["patch_path"],
                "label": sample["label"],
                "question": create_svdh_jsn_prompt(sample),
                "case_output_dir": case_output_dir,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures = upsert_record(failures, sample_key, error_record, svdh_sample_key)
            print(f"[ERROR] sample failed and was logged: {type(exc).__name__}: {exc}")

        write_json_snapshot(output_path, build_output(args, task, plan, toolset, results, failures, sites))

    output = build_output(args, task, plan, toolset, results, failures, sites)
    metrics = output["metrics"]
    write_json_snapshot(output_path, output)

    def fmt_percent(value):
        return "None" if value is None else f"{value:.2f}%"

    print(
        "\n[INFO] "
        f"QWK: {metrics['QWK']}, MAE: {metrics['MAE']}, "
        f"BACC: {fmt_percent(metrics['BACC (%)'])}, ACC: {fmt_percent(metrics['ACC (%)'])}, "
        f"W1-ACC: {fmt_percent(metrics['W1-ACC (%)'])}, P/N-SEN: {fmt_percent(metrics['P/N-SEN (%)'])}, "
        f"P/N-ACC: {fmt_percent(metrics['P/N-ACC (%)'])}, Params(M): {metrics['Params (M)']}, "
        f"Time(ms): {metrics['Time (ms)']}"
    )
    print(f"[INFO] saved: {output_path}")


if __name__ == "__main__":
    main()
