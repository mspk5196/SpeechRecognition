"""Transformer-based sequence-to-sequence entity extractor.

Goal: Convert a natural language playback / camera command into structured
JSON containing entities (date, date_range_start, date_range_end,
start_time, end_time, camera, part_of_day, direction, etc.).

Design:
- Uses a lightweight instruction prompt + the raw text.
- Loads a small seq2seq model by default (T5-small) to balance latency vs accuracy.
- If environment TRANSFORMER_MODEL is set, uses that instead.
- Attempts to parse the model output as JSON (robust to extra text).
- Falls back to empty dict on failure.

NOTE: This is a generic prompt-based approach; accuracy will depend on the
chosen model. For more consistent extraction, fine-tune or craft a better
set of few-shot exemplars.
"""
from __future__ import annotations
import os
import json
from typing import Dict, Any

# Lazy imports so that environments not using transformer backend don't pay cost
_model = None
_tokenizer = None

DEFAULT_MODEL = os.getenv("TRANSFORMER_MODEL", "t5-small")
MAX_NEW_TOKENS = int(os.getenv("TRANSFORMER_MAX_NEW_TOKENS", "128"))
DEVICE = os.getenv("TRANSFORMER_DEVICE", "cpu")  # 'cuda' if available & desired

INSTRUCTION = (
    "Extract CCTV playback / PTZ entities as compact JSON keys. Allowed keys: "
    "intent, date, date_range_start, date_range_end, start_time, end_time, camera, part_of_day, direction. "
    "Times must be 24h HH:MM. If single time only, put it in start_time. "
    "If a range crosses midnight without explicit next-day phrase, keep given times. "
    "If relative dates (yesterday, today, tomorrow) resolve them to ISO YYYY-MM-DD (assume current locale date). "
    "If multiple relative days appear (e.g. yesterday ... today) set date_range_start and date_range_end to each resolved ISO date. "
    "If camera/channel specified output camera as integer string. "
    "Respond ONLY with minified JSON object, no commentary."
)

EXAMPLES = [
    ("playback camera 2 yesterday from 5 pm to 6 pm", '{"intent":"playback","camera":"2","date":"{YESTERDAY}","date_range_start":"{YESTERDAY}","date_range_end":"{YESTERDAY}","start_time":"17:00","end_time":"18:00"}'),
    ("show playback from 9am to 10am today camera 1", '{"intent":"playback","camera":"1","date":"{TODAY}","date_range_start":"{TODAY}","date_range_end":"{TODAY}","start_time":"09:00","end_time":"10:00"}'),
    ("move camera 3 left", '{"intent":"ptz","camera":"3","direction":"left"}')
]

from datetime import date, timedelta

def _resolve_relatives(example_json: str) -> str:
    today = date.today()
    mapping = {
        '{TODAY}': today.isoformat(),
        '{YESTERDAY}': (today - timedelta(days=1)).isoformat(),
        '{TOMORROW}': (today + timedelta(days=1)).isoformat(),
    }
    for k,v in mapping.items():
        example_json = example_json.replace(k, v)
    return example_json

PROMPT_CACHE = None

def build_prompt(command: str) -> str:
    global PROMPT_CACHE
    if PROMPT_CACHE is None:
        shots = []
        for txt, js in EXAMPLES:
            shots.append(f"Input: {txt}\nOutput: {_resolve_relatives(js)}")
        PROMPT_CACHE = INSTRUCTION + "\n\n" + "\n\n".join(shots) + "\n\n" + "Input: {user}\nOutput:"
    return PROMPT_CACHE.replace('{user}', command)


def load_model():  # pragma: no cover (heavy)
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        _tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
        _model = AutoModelForSeq2SeqLM.from_pretrained(DEFAULT_MODEL)
        if DEVICE != 'cpu':
            _model = _model.to(DEVICE)
    except Exception as e:  # pragma: no cover
        print(f"[transformer_extractor] Failed loading model {DEFAULT_MODEL}: {e}")
        _model = None; _tokenizer = None
    return _model, _tokenizer


def extract_with_transformer(text: str) -> Dict[str, Any]:
    model, tokenizer = load_model()
    if not model or not tokenizer:
        return {}
    prompt = build_prompt(text)
    try:
        encoded = tokenizer(prompt, return_tensors="pt")
        if DEVICE != 'cpu':
            encoded = {k: v.to(DEVICE) for k,v in encoded.items()}
        output_ids = model.generate(**encoded, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        # Extract first JSON object substring
        json_start = decoded.find('{')
        json_end = decoded.rfind('}')
        if json_start != -1 and json_end != -1 and json_end > json_start:
            candidate = decoded[json_start:json_end+1]
        else:
            candidate = decoded
        try:
            data = json.loads(candidate)
            # Basic post processing: coerce camera & times format if present
            if isinstance(data, dict):
                # Normalize times to HH:MM if HH:MM:SS given
                for tkey in ['start_time','end_time']:
                    if tkey in data and isinstance(data[tkey], str):
                        parts = data[tkey].split(':')
                        if len(parts) >= 2:
                            data[tkey] = f"{parts[0].zfill(2)}:{parts[1].zfill(2)}"
                if 'camera' in data and isinstance(data['camera'], (int,str)):
                    try:
                        cam_int = int(str(data['camera']).strip())
                        data['camera'] = str(cam_int)
                    except Exception:
                        pass
                return data
        except Exception:
            return {}
    except Exception as gen_err:  # pragma: no cover
        print(f"[transformer_extractor] generation error: {gen_err}")
        return {}
    return {}

if __name__ == "__main__":  # manual test
    print(extract_with_transformer("playback camera 4 yesterday from 11 pm to today 1 am"))
