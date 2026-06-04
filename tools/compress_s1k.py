#!/usr/bin/env python3
"""Compress S1K math samples into concise reasoning traces.

This script is intentionally conservative:
- it never edits the input file;
- it writes JSONL incrementally for resumability;
- it validates boxed answers and token budgets;
- it can run in dry-run mode to inspect prompts before API calls.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BAD_PHRASES = (
    "maybe",
    "wait",
    "i think",
    "let me",
    "let's",
    "we need solve",
    "we need to solve",
    "actually",
    "recheck",
)


SYSTEM_PROMPT = """You are compressing a math solution for supervised fine-tuning.

Compress the example into a concise, faithful reasoning trace.

Hard rules:
1. Do not change the problem, variables, constraints, or final answer.
2. Do not invent new reasoning not supported by the original solution.
3. Keep all key equations, substitutions, inequalities, case splits, and validity checks.
4. Remove false starts, repeated statements, conversational filler, and unnecessary arithmetic detail.
5. Put the boxed final answer/conclusion in compressed_answer and at the end of compressed_reasoning.
6. Do not use \\boxed{...} for intermediate statements.
7. Never include self-corrections such as "Wait", "Actually", "recheck", "I think", or "let me".
8. Return minified valid JSON only, with no markdown fences or commentary.
"""


USER_TEMPLATE = """Use this format for compressed_reasoning:

Method: <main method in one line>
Steps:
1. <first essential step>
2. <second essential step>
3. <third essential step>
...
Check: <constraint/domain check, if needed>
Answer: \\boxed{{...}}

Compression targets:
- compressed_question <= {max_question_tokens} GPT-2 tokens
- compressed_reasoning <= {max_reasoning_tokens} GPT-2 tokens

Writing rules:
- Keep compressed_reasoning compact: method line, essential steps, final Answer line.
- Aim for 450 GPT-2 tokens or fewer; {max_reasoning_tokens} is only the hard validation cap.
- For long proofs, keep only the main inequalities/identities and the final conclusion.
- Do not include self-corrections, uncertainty, repeated checks, or unnecessary arithmetic detail.
- If the original has a correction or false start, silently keep only the corrected final derivation.

Return minified valid JSON only:
{{
  "compressed_question": "...",
  "compressed_reasoning": "...",
  "compressed_answer": "\\\\boxed{{...}}"
}}

Final answer instruction:
{answer_instruction}

Original question:
{question}

Original solution:
{solution}

Original deepseek_attempt:
{deepseek_attempt}
"""


REPAIR_TEMPLATE = """Repair this failed compression output.

Validation errors:
{errors}

Correction rules:
- Return the same three JSON keys only: compressed_question, compressed_reasoning, compressed_answer.
- compressed_question <= {max_question_tokens} GPT-2 tokens; aim for 160 tokens or fewer.
- compressed_reasoning <= {max_reasoning_tokens} GPT-2 tokens; aim for 260 tokens or fewer.
- Use at most 4 short numbered steps. Each step must be one compact sentence.
- Keep only the key formula transformations and the final result; omit routine arithmetic and repeated simplification.
- For proofs, keep only the named theorem/lemma, the key inequality or recurrence, and the final conclusion.
- Remove all self-corrections and uncertainty words, especially "Wait", "Actually", "recheck", "I think", "let me".
- Keep only the corrected final derivation. Do not mention mistakes in the original or failed output.
- Put the final answer/conclusion exactly once at the end of compressed_reasoning as Answer: \\boxed{{...}}.

Final answer instruction:
{answer_instruction}

Original question:
{question}

Original solution:
{solution}

Failed compression JSON:
{failed_json}
"""


def load_token_counter() -> Any:
    try:
        from transformers import GPT2TokenizerFast

        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
        return tokenizer.encode
    except Exception:
        try:
            import tiktoken

            enc = tiktoken.get_encoding("gpt2")
            return enc.encode
        except Exception:
            return None


TOKEN_ENCODE = load_token_counter()


def token_count(text: str) -> int:
    if not text:
        return 0
    if TOKEN_ENCODE is not None:
        return len(TOKEN_ENCODE(text))
    # Rough fallback; only used when tokenizer packages/cache are absent.
    return max(1, int(len(text) / 4))


def normalize_answer(text: Any) -> str:
    s = str(text).strip()
    s = re.sub(r"^\\boxed\{(.+)\}$", r"\1", s)
    s = s.strip()
    s = re.sub(r"\\[dts]?frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", s)
    s = s.replace(" ", "")
    s = s.replace("\\,", "")
    s = s.replace("\\left", "").replace("\\right", "")
    return s


def strip_answer_candidate(text: str) -> str:
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    s = re.sub(r"^(?:answer|final answer)\s*[:：]\s*", "", s, flags=re.IGNORECASE)
    s = s.strip(" \t\r\n.;,$")
    while len(s) >= 2 and s[0] == "$" and s[-1] == "$":
        s = s[1:-1].strip()
    s = s.strip(" \t\r\n.;,$")
    return s


def final_equals_candidate(text: str) -> str | None:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) > 240:
        return None
    compact = compact.rstrip(" .;")
    match = re.search(r"=\s*([^=.;]+)$", compact)
    if not match:
        return None
    candidate = strip_answer_candidate(match.group(1))
    if not candidate or len(candidate) > 80:
        return None
    if re.search(r"\b(note|then|therefore|hence|because|where|with)\b", candidate, flags=re.IGNORECASE):
        return None
    return candidate


def explicit_answer_candidate(text: str) -> str | None:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) > 240:
        return None
    match = re.search(r"(?:^|[\s.])(?:answer|final answer)\s*[:：]\s*(.+?)\s*\.?$", compact, flags=re.IGNORECASE)
    if not match:
        return None
    candidate = strip_answer_candidate(match.group(1))
    if not candidate or len(candidate) > 120:
        return None
    return candidate


def boxed_inner(text: str) -> str | None:
    results: list[str] = []
    pos = 0
    text = str(text or "")
    prefix = r"\boxed{"
    while pos < len(text):
        idx = text.find(prefix, pos)
        if idx == -1:
            break
        start = idx + len(prefix)
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            results.append(text[start : i - 1].strip())
        pos = i
    return results[-1] if results else None


def answer_target(item: dict[str, Any]) -> tuple[str, bool]:
    solution = str(item.get("solution", "")).strip()
    boxed = boxed_inner(solution)
    if boxed:
        return boxed, True
    explicit = explicit_answer_candidate(solution)
    if explicit:
        return explicit, True
    final_equals = final_equals_candidate(solution)
    if final_equals:
        return final_equals, True
    compact = re.sub(r"\s+", " ", solution).strip()
    looks_like_reasoning = bool(
        re.search(
            r"\b(first|then|next|finally|prove|show|assume|therefore|hence|step|note|because|quantity|wish|find|compute|count|length)\b",
            compact,
            flags=re.IGNORECASE,
        )
    )
    if compact and len(compact) <= 200 and not looks_like_reasoning:
        return compact, True
    return "", False


def sample_id(index: int, item: dict[str, Any]) -> str:
    meta = parse_metadata(item.get("metadata"))
    raw = meta.get("ID") or item.get("id") or index
    return str(raw)


def parse_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        text = value.strip()
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        try:
            obj = ast.literal_eval(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {}


def answer_instruction(item: dict[str, Any]) -> str:
    target, is_known = answer_target(item)
    if is_known:
        return f"The final answer/conclusion must be exactly \\\\boxed{{{target}}}."
    return (
        "No short answer field is provided. If this is a proof, put the final proved "
        "claim/conclusion in \\\\boxed{...}; if it is a computation, put the computed "
        "result in \\\\boxed{...}. Do not box an intermediate statement."
    )


def build_prompt(item: dict[str, Any], max_question_tokens: int, max_reasoning_tokens: int) -> str:
    return USER_TEMPLATE.format(
        max_question_tokens=max_question_tokens,
        max_reasoning_tokens=max_reasoning_tokens,
        answer_instruction=answer_instruction(item),
        question=item.get("question", ""),
        solution=item.get("solution", ""),
        deepseek_attempt=item.get("deepseek_attempt", ""),
    )


def build_repair_prompt(
    item: dict[str, Any],
    compressed: dict[str, Any],
    validation: dict[str, Any],
    max_question_tokens: int,
    max_reasoning_tokens: int,
) -> str:
    return REPAIR_TEMPLATE.format(
        errors=", ".join(validation.get("errors") or []),
        max_question_tokens=max_question_tokens,
        max_reasoning_tokens=max_reasoning_tokens,
        answer_instruction=answer_instruction(item),
        question=item.get("question", ""),
        solution=item.get("solution", ""),
        failed_json=json.dumps(compressed, ensure_ascii=False),
    )


def parse_json_response(text: str) -> dict[str, Any]:
    original = text
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        extracted = extract_json_object(text)
        if extracted:
            return json.loads(extracted)
        preview = original[:800].replace("\n", "\\n")
        raise ValueError(f"Could not parse JSON response; preview={preview!r}")


def extract_json_object(text: str) -> str | None:
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for pos in range(start, len(text)):
            char = text[pos]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : pos + 1]
        start = text.find("{", start + 1)
    return None


def validate_compression(
    item: dict[str, Any],
    compressed: dict[str, Any],
    max_question_tokens: int,
    max_reasoning_tokens: int,
) -> dict[str, Any]:
    cq = str(compressed.get("compressed_question", "")).strip()
    cr = str(compressed.get("compressed_reasoning", "")).strip()
    ca = str(compressed.get("compressed_answer", "")).strip()
    q_tokens = token_count(cq)
    r_tokens = token_count(cr)
    boxed_answer = boxed_inner(ca)
    reasoning_boxed = boxed_inner(cr)
    expected_target, expected_known = answer_target(item)
    expected = normalize_answer(expected_target)
    actual = normalize_answer(boxed_answer or "")
    reasoning_actual = normalize_answer(reasoning_boxed or "")
    bad_hits = [phrase for phrase in BAD_PHRASES if phrase in cr.lower()]
    errors: list[str] = []
    if not cq:
        errors.append("empty_compressed_question")
    if not cr:
        errors.append("empty_compressed_reasoning")
    if boxed_answer is None:
        errors.append("compressed_answer_missing_boxed")
    if reasoning_boxed is None:
        errors.append("reasoning_missing_boxed")
    if expected_known and boxed_answer is not None and actual != expected:
        errors.append("boxed_answer_mismatch")
    if expected_known and reasoning_boxed is not None and reasoning_actual != expected:
        errors.append("reasoning_boxed_mismatch")
    if q_tokens > max_question_tokens:
        errors.append("compressed_question_too_long")
    if r_tokens > max_reasoning_tokens:
        errors.append("compressed_reasoning_too_long")
    if bad_hits:
        errors.append("bad_phrase")
    return {
        "ok": not errors,
        "errors": errors,
        "compressed_question_tokens": q_tokens,
        "compressed_reasoning_tokens": r_tokens,
        "expected_answer_known": expected_known,
        "expected_answer_norm": expected,
        "compressed_answer_norm": actual,
        "reasoning_boxed_norm": reasoning_actual,
        "bad_phrase_hits": bad_hits,
    }


@dataclass
class ApiConfig:
    model: str
    temperature: float
    max_retries: int
    retry_sleep: float


def call_openai(prompt: str, cfg: ApiConfig) -> dict[str, Any]:
    if cfg.model.lower().startswith("claude-") or os.environ.get("S1K_COMPRESS_API") == "anthropic":
        return call_anthropic_messages_http(prompt, cfg)
    try:
        from openai import OpenAI
    except Exception:
        return call_openai_http(prompt, cfg)
    client = OpenAI()
    last_error: Exception | None = None
    for attempt in range(cfg.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=cfg.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=cfg.temperature,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            return parse_json_response(content)
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            last_error = RuntimeError(f"HTTP {exc.code} {exc.reason}: {body[:1000]}")
            if attempt >= cfg.max_retries:
                break
            time.sleep(cfg.retry_sleep * (attempt + 1))
        except Exception as exc:
            last_error = exc
            if attempt >= cfg.max_retries:
                break
            time.sleep(cfg.retry_sleep * (attempt + 1))
    raise RuntimeError(f"API call failed after retries: {last_error}")


def api_url(base_url: str, suffix: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith(suffix):
        return base_url
    for known_suffix in ("/chat/completions", "/messages"):
        if base_url.endswith(known_suffix):
            base_url = base_url[: -len(known_suffix)]
            break
    return f"{base_url}{suffix}"


def call_anthropic_messages_http(prompt: str, cfg: ApiConfig) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.anthropic.com/v1")
    url = api_url(base_url, "/messages")
    max_tokens = int(os.environ.get("S1K_COMPRESS_MAX_OUTPUT_TOKENS", "1600"))
    payload = {
        "model": cfg.model,
        "max_tokens": max_tokens,
        "temperature": cfg.temperature,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
    }
    if os.environ.get("S1K_COMPRESS_DISABLE_THINKING") == "1":
        payload["thinking"] = {"type": "disabled"}
    data = json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(cfg.max_retries + 1):
        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "anthropic-version": os.environ.get("ANTHROPIC_VERSION", "2023-06-01"),
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            obj = json.loads(body)
            parts = obj.get("content", [])
            if isinstance(parts, str):
                content = parts
            else:
                content = "".join(
                    part.get("text", "")
                    for part in parts
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            if not content.strip():
                preview = json.dumps(
                    {
                        "stop_reason": obj.get("stop_reason"),
                        "content": parts,
                        "usage": obj.get("usage"),
                    },
                    ensure_ascii=False,
                )[:1000]
                raise RuntimeError(f"Anthropic response had no text content: {preview}")
            return parse_json_response(content)
        except json.JSONDecodeError as exc:
            last_error = RuntimeError(f"Could not decode API response JSON: {exc}")
            if attempt >= cfg.max_retries:
                break
            time.sleep(cfg.retry_sleep * (attempt + 1))
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            last_error = RuntimeError(f"HTTP {exc.code} {exc.reason}: {body[:1000]}")
            if attempt >= cfg.max_retries:
                break
            time.sleep(cfg.retry_sleep * (attempt + 1))
        except Exception as exc:
            last_error = exc
            if attempt >= cfg.max_retries:
                break
            time.sleep(cfg.retry_sleep * (attempt + 1))
    raise RuntimeError(f"Anthropic messages API call failed after retries: {last_error}")


def call_openai_http(prompt: str, cfg: ApiConfig) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    url = api_url(base_url, "/chat/completions")
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": cfg.temperature,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(cfg.max_retries + 1):
        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
            content = obj["choices"][0]["message"]["content"]
            return parse_json_response(content)
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            last_error = RuntimeError(f"HTTP {exc.code} {exc.reason}: {body[:1000]}")
            if attempt >= cfg.max_retries:
                break
            time.sleep(cfg.retry_sleep * (attempt + 1))
        except Exception as exc:
            last_error = exc
            if attempt >= cfg.max_retries:
                break
            time.sleep(cfg.retry_sleep * (attempt + 1))
    raise RuntimeError(f"HTTP API call failed after retries: {last_error}")


def read_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                validation = obj.get("validation")
                if isinstance(validation, dict) and not validation.get("ok"):
                    continue
                done.add(str(obj.get("sample_id")))
            except Exception:
                continue
    return done


def write_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def export_train_json(jsonl_path: Path, train_out: Path) -> None:
    rows: list[dict[str, Any]] = []
    if not jsonl_path.exists():
        return
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            validation = obj.get("validation") or {}
            if not validation.get("ok"):
                continue
            compressed = obj.get("compressed") or {}
            rows.append(
                {
                    "question": compressed.get("compressed_question", ""),
                    "solution": compressed.get("compressed_reasoning", ""),
                    "source_id": obj.get("sample_id"),
                    "metadata": obj.get("metadata"),
                }
            )
    train_out.parent.mkdir(parents=True, exist_ok=True)
    train_out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/s1K_train_599.json")
    parser.add_argument("--output_jsonl", default="data/s1K_train_599_compressed_medium.jsonl")
    parser.add_argument("--errors_jsonl", default="data/s1K_train_599_compressed_medium_errors.jsonl")
    parser.add_argument("--train_out", default="data/s1K_train_599_compressed_reasoning_medium.json")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--model", default=os.environ.get("S1K_COMPRESS_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4o-mini")))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_question_tokens", type=int, default=220)
    parser.add_argument("--max_reasoning_tokens", type=int, default=512)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--repair_attempts", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--export_only", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output_jsonl)
    error_path = Path(args.errors_jsonl)
    train_out = Path(args.train_out)
    if args.export_only:
        export_train_json(output_path, train_out)
        print(f"exported training json: {train_out}")
        return 0

    samples = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise TypeError(f"Expected list input, got {type(samples).__name__}")

    done = read_done_ids(output_path) if args.resume else set()
    selected = list(enumerate(samples))[args.start : args.start + args.limit]
    cfg = ApiConfig(
        model=args.model,
        temperature=args.temperature,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
    )

    api_available = bool(os.environ.get("OPENAI_API_KEY"))
    if args.dry_run:
        print("dry_run=true; no API calls will be made")
    elif not api_available:
        print("OPENAI_API_KEY is not set; use --dry_run or set an API key", file=sys.stderr)
        return 2

    ok_count = 0
    error_count = 0
    for index, item in selected:
        sid = sample_id(index, item)
        if sid in done:
            print(f"skip done index={index} id={sid}")
            continue
        prompt = build_prompt(item, args.max_question_tokens, args.max_reasoning_tokens)
        base = {
            "index": index,
            "sample_id": sid,
            "metadata": parse_metadata(item.get("metadata")),
            "source_type": item.get("source_type"),
            "original": {
                "question": item.get("question"),
                "solution": item.get("solution"),
                "deepseek_attempt": item.get("deepseek_attempt"),
            },
        }
        try:
            if args.dry_run:
                compressed = {
                    "compressed_question": "",
                    "compressed_reasoning": "",
                    "compressed_answer": "",
                    "prompt_preview": prompt[:4000],
                }
                validation = {"ok": False, "errors": ["dry_run"]}
            else:
                compressed = call_openai(prompt, cfg)
                validation = validate_compression(
                    item,
                    compressed,
                    args.max_question_tokens,
                    args.max_reasoning_tokens,
                )
                for repair_index in range(args.repair_attempts):
                    if validation.get("ok"):
                        break
                    repair_prompt = build_repair_prompt(
                        item,
                        compressed,
                        validation,
                        args.max_question_tokens,
                        args.max_reasoning_tokens,
                    )
                    compressed = call_openai(repair_prompt, cfg)
                    validation = validate_compression(
                        item,
                        compressed,
                        args.max_question_tokens,
                        args.max_reasoning_tokens,
                    )
                    validation["repair_attempt"] = repair_index + 1
            out = {**base, "compressed": compressed, "validation": validation}
            write_jsonl(output_path, out)
            if validation.get("ok"):
                ok_count += 1
            else:
                error_count += 1
                write_jsonl(error_path, out)
            print(f"index={index} id={sid} ok={validation.get('ok')} errors={validation.get('errors')}")
        except Exception as exc:
            error_count += 1
            out = {**base, "error": str(exc)}
            write_jsonl(error_path, out)
            print(f"index={index} id={sid} error={exc}", file=sys.stderr)

    export_train_json(output_path, train_out)
    print(f"done ok={ok_count} errors={error_count}")
    print(f"output_jsonl={output_path}")
    print(f"errors_jsonl={error_path}")
    print(f"train_out={train_out}")
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
