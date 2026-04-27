import json
import os
import re

from .openai_client import build_content, call_openai_messages
from utils import concise_system, concise_user, SHORT_REASON_RULE


class Pro_Decider_JSN:
    def __init__(self, api_key: str = None, model="gpt-4o-mini"):
        self.api_key = api_key
        self.model = model

    def safe_json_parse(self, text: str):
        t = (text or "").strip()
        if t.startswith("```"):
            parts = t.split("```")
            parts = sorted((p.strip() for p in parts), key=len, reverse=True)
            for p in parts:
                if p.startswith("{") and '"threshold"' in p:
                    try:
                        return json.loads(p)
                    except Exception:
                        pass
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            s, e = t.find("{"), t.rfind("}")
            if s != -1 and e != -1 and e > s:
                return json.loads(t[s:e + 1])
            raise

    def extract_svdh_jsn_score(self, text: str):
        match = re.search(r'"score"\s*:\s*"?([0-4])"?', text or "")
        if match:
            return int(match.group(1))
        match = re.search(r'\b(?:score|answer)\s*[:=]?\s*([0-4])\b', text or "", re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def norm_yesno(self, value):
        if isinstance(value, (int, float)):
            x = float(value)
            return max(0.0, min(1.0, x))
        if isinstance(value, str):
            s = value.strip().lower()
            if s in {"yes", "true", "positive", "abnormal"}:
                return 1.0
            if s in {"no", "false", "negative", "normal"}:
                return 0.0
            try:
                x = float(s)
                return max(0.0, min(1.0, x))
            except Exception:
                return 0.5
        return 0.5

    def weights_from_model(self, obj, indicators):
        model_ws = obj.get("weights", [])
        w_map = {}
        for w in model_ws:
            name = str(w.get("indicator_name", "")).strip()
            try:
                val = float(w.get("weight"))
            except Exception:
                continue
            if name:
                w_map[name] = max(0.0, min(1.0, val))

        names = [str(it["indicator_name"]).strip() for it in indicators]
        missing = [n for n in names if n not in w_map]
        total = sum(w_map.values())
        if total > 0:
            for key in list(w_map.keys()):
                w_map[key] = w_map[key] / total
        total = sum(w_map.values())
        if missing:
            remain = max(0.0, 1.0 - total)
            share = remain / len(missing) if missing else 0.0
            for name in missing:
                w_map[name] = share
        total = sum(w_map.values())
        if total == 0:
            uniform = 1.0 / max(1, len(names))
            w_map = {name: uniform for name in names}
        else:
            for key in list(w_map.keys()):
                w_map[key] = w_map[key] / total
        return w_map

    def decide(self, output_file: str, prompt: str, indicators, image_paths=None, field: str = "overall"):
        lines = []
        for it in indicators:
            name = str(it.get("indicator_name", "")).strip()
            val = it.get("if_abnormal", "")
            if isinstance(val, (dict, list)):
                val_text = json.dumps(val, ensure_ascii=False)
            else:
                val_text = str(val)
            lines.append(f"- {name}: {val_text}")
        ind_text = "Indicators & current judgements:\n" + "\n".join(lines)

        system_msg = concise_system(
            "You are a careful clinical decision assistant. "
            "Given the task and indicator judgements, propose weights that sum to 1 and a threshold in [0,1]. "
            "Return ONLY a JSON object with keys: 'weights' (list of {'indicator_name','weight'}), "
            "'threshold' (float in [0,1]), and optional 'notes'."
        )
        user_prompt = concise_user(
            f"{prompt}\n\n{ind_text}\n\n"
            "Constraints:\n"
            "- Sum of weights must be 1.\n"
            "- Threshold must be in [0,1].\n"
            "- Do NOT include any explanation outside the JSON object.",
            "Return only compact JSON."
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": build_content(user_prompt, image_paths)},
        ]

        raw, elapsed_ms = call_openai_messages(
            messages=messages,
            model=self.model,
            api_key=self.api_key,
        )
        obj = self.safe_json_parse(raw)

        w_map = self.weights_from_model(obj, indicators)
        try:
            threshold = float(obj.get("threshold"))
        except Exception:
            threshold = 0.5
        threshold = max(0.0, min(1.0, threshold))

        contribs = []
        score = 0.0
        for it in indicators:
            name = str(it.get("indicator_name", "")).strip()
            val = self.norm_yesno(it.get("if_abnormal"))
            weight = float(w_map.get(name, 0.0))
            contribs.append({"indicator_name": name, "value": val, "weight": weight, "weighted": val * weight})
            score += val * weight

        diagnosis = "Positive" if score >= threshold else "Negative"
        result_obj = {
            "weights": w_map,
            "threshold": threshold,
            "score": score,
            "diagnosis": diagnosis,
            "contributions": contribs,
            "model_raw": obj.get("notes", "") if isinstance(obj, dict) else "",
            "time_ms": elapsed_ms,
        }

        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
        else:
            existing = {}

        existing[field] = result_obj
        existing[f"{field}_trace"] = {
            "model": self.model,
            "system_prompt": system_msg,
            "user_prompt": user_prompt,
            "image_paths": image_paths or [],
            "indicators": indicators,
            "raw_response": raw,
            "time_ms": elapsed_ms,
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=4, ensure_ascii=False)

        return existing

    def decide_svdh_score(self, output_file: str, prompt: str, indicators, image_paths=None, field: str = "overall"):
        lines = []
        for it in indicators:
            name = str(it.get("indicator_name", "")).strip()
            val = it.get("value", "")
            if isinstance(val, (dict, list)):
                val_text = json.dumps(val, ensure_ascii=False)
            else:
                val_text = str(val)
            lines.append(f"- {name}: {val_text}")

        system_msg = concise_system(
            "You are a careful musculoskeletal radiology decision assistant. "
            "Your job is to make a final Sharp/van der Heijde joint space narrowing score from 0 to 4."
        )
        user_prompt = concise_user(
            f"{prompt}\n\nIndicators and prior judgements:\n" + "\n".join(lines) + "\n\n"
            "Inspect the attached local joint patch and decide the final SvdH joint space narrowing score only. "
            "Return ONLY JSON in this exact schema: {\"score\":0,\"reason\":\"short visual reason\"}.",
            "Return only compact JSON. Keep the reason extremely short."
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": build_content(user_prompt, image_paths)},
        ]
        raw, elapsed_ms = call_openai_messages(
            messages=messages,
            model=self.model,
            api_key=self.api_key,
            temperature=0,
        )
        score = self.extract_svdh_jsn_score(raw)
        result_obj = {
            "score": score,
            "raw": raw,
            "time_ms": elapsed_ms,
        }

        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
        else:
            existing = {}

        existing[field] = result_obj
        existing[f"{field}_trace"] = {
            "model": self.model,
            "system_prompt": system_msg,
            "user_prompt": user_prompt,
            "image_paths": image_paths or [],
            "indicators": indicators,
            "raw_response": raw,
            "time_ms": elapsed_ms,
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=4, ensure_ascii=False)

        return existing
