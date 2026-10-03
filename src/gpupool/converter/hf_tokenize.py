"""Standalone: tokenize texts with the Hugging Face tokenizer of a model directory.

Runs under the converter's python (stdlib + transformers only; gpupool is NOT installed there).

    python hf_tokenize.py <model_dir> [trust_remote_code: 0|1]   < JSON list of strings
    -> stdout: {"ids": [[...], ...]}   or   {"error": "..."}   (always exit code 0)

Special tokens are not added (add_special_tokens=False): the caller compares against
llama-tokenize --no-bos --no-parse-special on the same strings.
"""
from __future__ import annotations

import json
import sys


def run(argv: list[str], stdin_text: str) -> dict:
    try:
        if len(argv) < 1:
            return {"error": "usage: hf_tokenize.py <model_dir> [trust_remote_code 0|1]"}
        model_dir = argv[0]
        trust = len(argv) > 1 and argv[1].strip().lower() in ("1", "true", "yes")
        texts = json.loads(stdin_text)
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            return {"error": "stdin must be a JSON list of strings"}
        from transformers import AutoTokenizer  # heavy import, kept inside the try

        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=trust)
        return {"ids": [[int(i) for i in tok.encode(t, add_special_tokens=False)] for t in texts]}
    except BaseException as e:  # noqa: BLE001 - the contract is "never a traceback"
        return {"error": f"{type(e).__name__}: {e}"}


def main() -> int:
    try:
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    result = run(sys.argv[1:], sys.stdin.read())
    sys.stdout.write(json.dumps(result, ensure_ascii=True))
    sys.stdout.write("\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
