"""Turn radiology reports into graded verdicts using a local language model.

This is the weak-supervision step: it produces the training target for the
imaging model from the 4,349 studies that have a report but no label.

Two design decisions are worth knowing before changing anything here.

**Verdicts, not probabilities.** The model is asked to pick from a small fixed
vocabulary — present, probable, equivocal, absent, not_mentioned — rather than
to emit a number. Language models are poor at producing calibrated floats and
tend to cluster them at 0.9 and 0.5, but they are reliable at choosing from a
short list. The verdicts are mapped to probabilities afterwards.

**The mapping is a separate step.** Running the model over 4,407 reports is the
expensive part and should happen once. Converting verdicts to probabilities is
arithmetic, so the mapping can be tuned against the gold set as often as you
like without re-running anything. Keep that separation.

**not_mentioned is not absent.** Radiologists report by exception, so silence
usually means normal — but not always, and not equally for every finding. A
report that never mentions the ACL probably has an intact one; a report that
never mentions synovitis may simply be from a radiologist who does not comment
on it. So "not mentioned" maps to a per-label prior, not to zero. Those priors
should be estimated from the gold set once it is large enough to support it.

Usage:
    python scripts/extract_labels.py --model <hf-model-id> --out data/verdicts.jsonl
    python scripts/extract_labels.py --out data/verdicts.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.load_data import LABELS, load_train  # noqa: E402

VERDICTS = ["present", "probable", "equivocal", "absent", "not_mentioned"]

# Starting point for the verdict-to-probability mapping. Tune the first four
# against the gold set; they are only a sensible prior, not a measured one.
DEFAULT_MAPPING = {
    "present": 0.95,
    "probable": 0.75,
    "equivocal": 0.50,
    "absent": 0.03,
}

# Where "not mentioned" lands, per label. A finding a radiologist would almost
# always mention if present gets a low value; one that often goes uncommented
# gets a higher one. These are judgement calls standing in for measurements —
# replace them with rates estimated from the gold set as it grows.
DEFAULT_SILENCE_PRIOR = {
    "ACL": 0.05,
    "MCL": 0.05,
    "Medial Meniscus": 0.05,
    "Lateral Meniscus": 0.05,
    "Medial OA": 0.12,
    "Lateral OA": 0.10,
    "PF OA": 0.15,
    "Effusion": 0.10,
    "Synovitis": 0.20,
    "Baker's": 0.08,
    "Contusion": 0.12,
    "Fracture": 0.05,
}

SYSTEM_PROMPT = """You are a musculoskeletal radiologist reading knee MRI reports.

Reports may be written in any language — English, Spanish, Turkish, Greek, \
Bosnian, Croatian, German, Bulgarian, Dutch, French and others. Read the report \
in its original language. Do not translate it.

For each of the twelve findings listed, judge what the report says, and answer \
with exactly one of these words:

  present       the report states the finding is there
  probable      the report suggests it, hedged ("likely", "suspicious for")
  equivocal     the report raises it but cannot decide
  absent        the report explicitly states the finding is not there
  not_mentioned the report says nothing either way about it

Two rules that matter:

- "not_mentioned" and "absent" are different. Use "absent" only when the report \
actually denies the finding. Use "not_mentioned" when the report is silent.
- Judge only what the report says. Do not infer one finding from another.

Answer with a JSON object only — no explanation, no preamble."""

USER_TEMPLATE = """Findings to judge:
{labels}

Report:
---
{report}
---

Answer with a JSON object whose keys are exactly the twelve finding names above \
and whose values are one of: present, probable, equivocal, absent, not_mentioned."""


def build_prompt(report: str) -> list[dict[str, str]]:
    """The chat messages for one report."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                labels="\n".join(f"  - {label}" for label in LABELS),
                report=report.strip(),
            ),
        },
    ]


def parse_response(text: str) -> dict[str, str]:
    """Pull the verdict object out of a model response.

    Tolerates the usual surrounding noise: code fences, a sentence of preamble,
    trailing commentary. Any label the model omitted, or answered with a word
    outside the vocabulary, comes back as "not_mentioned" — the neutral choice.
    """
    start, end = text.find("{"), text.rfind("}")
    parsed: dict = {}
    if start != -1 and end > start:
        try:
            candidate = json.loads(text[start : end + 1])
            if isinstance(candidate, dict):
                parsed = candidate
        except json.JSONDecodeError:
            parsed = {}

    verdicts = {}
    for label in LABELS:
        value = str(parsed.get(label, "")).strip().lower()
        verdicts[label] = value if value in VERDICTS else "not_mentioned"
    return verdicts


def verdicts_to_probabilities(
    verdicts: dict[str, str],
    mapping: dict[str, float] | None = None,
    silence_prior: dict[str, float] | None = None,
) -> dict[str, float]:
    """Convert one study's verdicts into twelve probabilities."""
    mapping = mapping or DEFAULT_MAPPING
    silence_prior = silence_prior or DEFAULT_SILENCE_PRIOR
    return {
        label: (
            silence_prior[label]
            if verdicts[label] == "not_mentioned"
            else mapping[verdicts[label]]
        )
        for label in LABELS
    }


def already_done(path: Path) -> set[str]:
    """Study identifiers already written, so a run can resume after a crash."""
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                done.add(json.loads(line)["StudyInstanceUID"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Hugging Face model id of a local multilingual model.")
    parser.add_argument("--out", default=Path("data/verdicts.jsonl"), type=Path)
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N studies.")
    parser.add_argument("--max-report-chars", type=int, default=6000)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print one prompt and exit, without loading a model.",
    )
    args = parser.parse_args()

    train = load_train()
    if args.limit:
        train = train.head(args.limit)

    if args.dry_run:
        print(f"{len(train):,} studies would be processed.\n")
        print("=" * 70)
        for message in build_prompt(train.Report.iloc[0][: args.max_report_chars]):
            print(f"\n[{message['role']}]\n{message['content']}")
        print("\n" + "=" * 70)
        return 0

    if not args.model:
        print("Pass --model, or --dry-run to inspect the prompt without a model.")
        return 1

    # Imported here so that --dry-run works on a machine with no GPU stack.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    done = already_done(args.out)
    if done:
        print(f"Resuming: {len(done):,} studies already written.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("a", encoding="utf-8") as handle:
        for row in train.itertuples():
            if row.StudyInstanceUID in done:
                continue

            prompt = tokenizer.apply_chat_template(
                build_prompt(str(row.Report)[: args.max_report_chars]),
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs, max_new_tokens=400, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            reply = tokenizer.decode(
                generated[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )

            verdicts = parse_response(reply)
            handle.write(
                json.dumps(
                    {"StudyInstanceUID": row.StudyInstanceUID, "verdicts": verdicts},
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()

            written += 1
            if written % 50 == 0:
                print(f"  … {written:,} written")

    print(f"Wrote {written:,} studies to {args.out}.")
    print("Next: map the verdicts to probabilities and score them against the gold set:")
    print("  python scripts/review_classification.py data/verdicts.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
