import argparse
import json
import os
import random
import time

from tqdm import tqdm

from Decider.GPT_Decider import GPT_Decider
from Decider.Pro_Decider import Pro_Decider
from utils import (
    compute_svdh_metrics,
    create_svdh_be_prompt,
    extract_svdh_score,
    load_svdh_be_data,
)


def run_svdh_case(sample, analyzer, decider, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    prompt = create_svdh_be_prompt(sample)
    start_time = time.perf_counter()

    analysis_prompt = (
        "You are the qualitative analysis module in MedAgent-Pro.\n"
        "Inspect the attached local joint X-ray patch and provide the evidence needed for final SvdH BE scoring.\n"
        f"{prompt}\n"
        "Return ONLY JSON in this exact schema: {\"score\":0,\"evidence\":\"short visual evidence\"}."
    )
    analysis_path = os.path.join(save_dir, "qualitative_analysis.json")
    analysis = analyzer.decide(
        output_file=analysis_path,
        prompt=analysis_prompt,
        image_paths=[sample["patch_path"]],
        field="svdh_be_indicator",
    )

    indicator_text = analysis.get("svdh_be_indicator", "")
    final_path = os.path.join(save_dir, "final_diagnosis.json")
    final = decider.decide_svdh_score(
        output_file=final_path,
        prompt=prompt,
        indicators=[{
            "indicator_name": "bone erosion visual evidence",
            "value": indicator_text,
        }],
        image_paths=[sample["patch_path"]],
        field="overall",
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    overall = final.get("overall", {})
    prediction = overall.get("score")
    if prediction is None:
        prediction = extract_svdh_score(overall.get("raw", ""))

    return {
        "question": prompt,
        "analysis": indicator_text,
        "final": overall,
        "prediction": prediction,
        "time_ms": elapsed_ms,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--svdh_root", type=str, default="../MDAgents/RAM-H1200/SvdH_Scoring/SvdH_BE_Scoring")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--random_sample", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sites", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--output_dir", type=str, default="output/svdh_be_medagent_pro")
    args = parser.parse_args()

    sites = [site.strip() for site in args.sites.split(",")] if args.sites else None
    samples = load_svdh_be_data(args.svdh_root, split=args.split, sites=sites)
    if args.random_sample:
        random.seed(args.seed)
        random.shuffle(samples)

    analyzer = GPT_Decider(model=args.model)
    decider = Pro_Decider(model=args.model)
    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for no, sample in enumerate(tqdm(samples)):
        if no == args.num_samples:
            break

        print(f"\n[INFO] no: {no}")
        print(f"image: {sample['image_name']}, site: {sample['site']}")
        print(f"patch: {sample['patch_path']}")

        case_name = f"{no:04d}_{os.path.splitext(sample['image_name'])[0]}_{sample['site']}"
        case_output_dir = os.path.join(args.output_dir, case_name)
        final_decision = run_svdh_case(sample, analyzer, decider, case_output_dir)
        prediction = final_decision["prediction"]
        print(f"label: {sample['label']}, prediction: {prediction}")

        results.append({
            "split": sample["split"],
            "image_name": sample["image_name"],
            "site": sample["site"],
            "score_key": sample["score_key"],
            "patch_path": sample["patch_path"],
            "label": sample["label"],
            "prediction": prediction,
            "response": final_decision,
            "time_ms": final_decision.get("time_ms"),
            "correct": prediction == sample["label"],
        })

    metrics = compute_svdh_metrics(results)
    output = {
        "task": "svdh_be",
        "framework": "MedAgent-Pro",
        "model": args.model,
        "svdh_root": args.svdh_root,
        "split": args.split,
        "num_samples": len(results),
        "metrics": metrics,
        "results": results,
    }
    output_path = os.path.join(args.output_dir, f"{args.model}_svdh_be_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

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
