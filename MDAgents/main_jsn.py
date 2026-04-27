import argparse
import json
import os
import random
import traceback

from tqdm import tqdm

from jsn_utils import (
    compute_svdh_jsn_metrics,
    load_svdh_jsn_data,
    process_svdh_jsn_mdagents_query,
)


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt-4o-mini")
parser.add_argument("--difficulty", type=str, default="adaptive")
parser.add_argument("--num_samples", type=int, default=100)
parser.add_argument("--svdh_root", type=str, default=os.path.expanduser("~/data/RAM-H1200/SvdH_Scoring/SvdH_JSN_Scoring"))
parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
parser.add_argument("--sites", type=str, default=None, help="Comma-separated SvdH JSN sites, e.g. MCP-T,PIP-R,STT")
parser.add_argument("--random_sample", action="store_true")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()


def ensure_output_dir():
    path = os.path.join(os.getcwd(), "output")
    os.makedirs(path, exist_ok=True)
    return path


def write_json_snapshot(output_path, payload):
    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4, ensure_ascii=False)
    os.replace(tmp_path, output_path)


def load_json_snapshot(output_path, default):
    if not os.path.exists(output_path):
        return default
    try:
        with open(output_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return default


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


def build_svdh_output(results, failures, sites):
    metrics = compute_svdh_jsn_metrics(results)
    return {
        "task": "svdh_jsn",
        "model": args.model,
        "svdh_root": args.svdh_root,
        "split": args.split,
        "sites": sites,
        "difficulty": args.difficulty,
        "random_sample": args.random_sample,
        "seed": args.seed,
        "num_samples": len(results),
        "num_failures": len(failures),
        "metrics": metrics,
        "results": results,
        "failures": failures,
    }


def main():
    sites = [site.strip() for site in args.sites.split(",")] if args.sites else None
    samples = load_svdh_jsn_data(args.svdh_root, split=args.split, sites=sites)
    if args.random_sample:
        random.seed(args.seed)
        random.shuffle(samples)

    ensure_output_dir()
    output_path = os.path.join("output", f"{args.model}_svdh_jsn_{args.split}.json")
    existing_output = load_json_snapshot(output_path, {})
    if (
        isinstance(existing_output, dict)
        and existing_output.get("task") == "svdh_jsn"
        and existing_output.get("model") == args.model
        and existing_output.get("svdh_root") == args.svdh_root
        and existing_output.get("split") == args.split
        and normalize_sites(existing_output.get("sites")) == normalize_sites(sites)
        and existing_output.get("difficulty") == args.difficulty
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
        print(f"[INFO] resume: found {len(results)} saved svdh_jsn samples in {output_path}")
    if failures:
        print(f"[INFO] resume: found {len(failures)} logged failures in {output_path}; failed samples will be retried")

    for sample in tqdm(pending_samples, initial=0, total=len(pending_samples)):
        if len(results) >= args.num_samples:
            break

        print(f"\n[INFO] no: {len(results)}")
        print(f"image: {sample['image_name']}, site: {sample['site']}")
        print(f"patch: {sample['patch_path']}")
        sample_key = svdh_sample_key(sample)
        try:
            final_decision = process_svdh_jsn_mdagents_query(sample, args.model, args.difficulty)
            prediction = final_decision["prediction"]
            print(f"difficulty: {final_decision.get('difficulty')}")
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
                "difficulty": final_decision.get("difficulty"),
                "response": final_decision["response"],
                "time_ms": final_decision.get("time_ms"),
                "diagnostic_trace": final_decision.get("trace"),
                "agent_details": {
                    key: value for key, value in final_decision.items()
                    if key not in {"prediction", "response", "time_ms", "difficulty", "question", "trace"}
                },
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
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures = upsert_record(failures, sample_key, error_record, svdh_sample_key)
            print(f"[ERROR] sample failed and was logged: {type(exc).__name__}: {exc}")
        write_json_snapshot(output_path, build_svdh_output(results, failures, sites))

    output = build_svdh_output(results, failures, sites)
    metrics = output["metrics"]
    write_json_snapshot(output_path, output)

    print(
        "\n[INFO] "
        f"QWK: {metrics['qwk']}, MAE: {metrics['mae']}, "
        f"BACC: {metrics['bacc_percent']:.2f}%, ACC: {metrics['acc_percent']:.2f}%, "
        f"W1-ACC: {metrics['w1_acc_percent']:.2f}%, "
        f"P/N-SEN: {metrics['p_n_sen_percent']:.2f}%, "
        f"P/N-ACC: {metrics['p_n_acc_percent']:.2f}%, "
        f"Params(M): {metrics['params_m']}, Time(ms): {metrics['time_ms']}"
    )


if __name__ == "__main__":
    main()
