import base64
import json
import mimetypes
import os
import time
import unicodedata
from io import BytesIO

import requests
from PIL import Image


HIDDEN_KEY_CHARS = {
    "\ufeff",  # BOM
    "\u200b",  # zero width space
    "\u200c",  # zero width non-joiner
    "\u200d",  # zero width joiner
    "\u2060",  # word joiner
    "\u00a0",  # non-breaking space
}


def sanitize_api_key(raw_key):
    if raw_key is None:
        return None, []

    key = str(raw_key).strip()
    if len(key) >= 2 and key[0] == key[-1] and key[0] in {"'", '"'}:
        key = key[1:-1].strip()

    cleaned_chars = []
    removed = []
    for ch in key:
        if ch in HIDDEN_KEY_CHARS or unicodedata.category(ch) == "Cf":
            removed.append(f"U+{ord(ch):04X}")
            continue
        if ch in {" ", "\t", "\r", "\n"}:
            continue
        cleaned_chars.append(ch)

    return "".join(cleaned_chars), removed


def get_openai_api_key(api_key=None):
    raw_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("openai_api_key")
    key, removed = sanitize_api_key(raw_key)
    if not key:
        raise EnvironmentError("Set OPENAI_API_KEY before calling OpenAI models.")
    if key == "你的_api_key":
        raise EnvironmentError('Replace OPENAI_API_KEY="你的_api_key" with your real OpenAI API key.')
    try:
        key.encode("ascii")
    except UnicodeEncodeError as exc:
        raise EnvironmentError(
            "OPENAI_API_KEY still contains non-ASCII characters after cleanup. "
            "Please re-export the real API key using plain ASCII text."
        ) from exc
    if removed:
        print(f"[WARN] cleaned hidden characters from OPENAI_API_KEY: {', '.join(removed)}")
    return key


def image_to_data_url(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type in {"image/png", "image/jpeg", "image/gif", "image/webp"}:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    with Image.open(image_path) as image:
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def build_content(prompt, image_paths=None):
    if image_paths and not isinstance(image_paths, list):
        image_paths = [image_paths]

    content = []
    for path in image_paths or []:
        content.append({
            "type": "image_url",
            "image_url": {"url": image_to_data_url(path)},
        })
    content.append({"type": "text", "text": prompt})
    return content


def call_openai_messages(messages, model="gpt-4o-mini", api_key=None, temperature=None, timeout=120):
    payload = {"model": model, "messages": messages}
    if temperature is not None:
        payload["temperature"] = temperature

    try:
        payload_json = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Failed to serialize OpenAI payload to valid JSON: {exc}") from exc

    start_time = time.perf_counter()
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {get_openai_api_key(api_key)}",
            "Content-Type": "application/json",
        },
        data=payload_json.encode("utf-8"),
        timeout=timeout,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    if not response.ok:
        raise requests.HTTPError(
            f"{response.status_code} {response.reason}: {response.text} "
            f"(model={model}, payload_bytes={len(payload_json.encode('utf-8'))})",
            response=response,
        )
    return response.json()["choices"][0]["message"]["content"], elapsed_ms
