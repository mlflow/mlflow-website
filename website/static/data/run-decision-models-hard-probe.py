"""Run the five displayed Part 2 hard examples as a new matched probe.

The original 72-row Part 2 fixture and documentation excerpts are unpublished.
This script uses the five question-answer pairs displayed by the author and
newly sourced official documentation in decision-models-hard-cases.json.
Run without --execute to verify the fixture; add --execute for live calls.
Install openai>=3.26.0 and typesafe-sdk>=0.5.7 for live calls. Pass
--env-file only if the provider keys are not already in the environment;
that option also requires python-dotenv. Local routes require an OpenJev server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from runpy import run_path
from uuid import uuid4

HERE = Path(__file__).resolve().parent
BENCHMARK = HERE / "run-decision-models-qa-benchmark.py"
MODELS = {
    "jev": "jev-1.13.0",
    "openai": "gpt-6-luna",
    "openjev": "openjev-0.1",
    "laya": "laya-1.0",
}


def summarize_probe(calls: list[dict], cases: list[dict]) -> dict:
    scored = [call for call in calls if call["scoring_status"] == "scored_negative"
              and call["status"] == "completed"]
    return {
        "case_count": len(cases),
        "attempted_count": len(calls),
        "completed_count": sum(call["status"] == "completed" for call in calls),
        "scored_negative_case_ids": [case["case_id"] for case in cases
                                     if case["scoring_status"] == "scored_negative"],
        "completed_scored_count": len(scored),
        "scored_false_accept_case_ids": [call["case_id"] for call in scored
                                         if call["prediction"] == "correct"],
        "ambiguous_case_ids": [case["case_id"] for case in cases
                               if case["scoring_status"] == "ambiguous"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", choices=MODELS)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--fixture", type=Path,
                        default=HERE / "decision-models-hard-cases.json")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--local-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--output-dir", type=Path,
                        default=Path.cwd() / "decision-models-hard-runs")
    args = parser.parse_args()

    fixture_bytes = args.fixture.read_bytes()
    fixture = json.loads(fixture_bytes)
    cases = fixture["cases"]
    if len(cases) != 5 or len({case["id"] for case in cases}) != 5:
        parser.error("expected five unique published hard examples")
    if any(not isinstance(case.get("context"), str) or not case["context"].strip()
           for case in cases):
        parser.error("all five cases need sourced documentation context")
    if any(case["published_label"] != "incorrect" for case in cases):
        parser.error("the five displayed examples are all labeled incorrect")
    if sorted(case["scoring_status"] for case in cases) != [
        "ambiguous", "scored_negative", "scored_negative", "scored_negative",
        "scored_negative",
    ]:
        parser.error("expected four scored negatives and one ambiguous example")

    helpers = run_path(str(BENCHMARK))
    if helpers["THRESHOLD"] != fixture["threshold"]:
        parser.error("the benchmark runner and hard-case threshold differ")
    canonical = helpers["canonical"]
    prepared = [
        {
            "case_id": case["id"],
            "state": {key: case[key] for key in ("question", "context", "answer")},
            "human_label": case["published_label"],
            "scoring_status": case["scoring_status"],
        }
        for case in cases
    ]
    print(f"Validated {len(prepared)} cases; fixture SHA-256 "
          f"{hashlib.sha256(fixture_bytes).hexdigest()}")
    for case in prepared:
        digest = hashlib.sha256(canonical(case["state"]).encode("utf-8")).hexdigest()
        print(case["case_id"], digest)
    if not args.execute:
        return 0

    if args.env_file:
        from dotenv import load_dotenv

        load_dotenv(args.env_file, override=True)
    output_dir = args.output_dir.resolve()
    if output_dir == HERE or HERE in output_dir.parents:
        parser.error("--output-dir must be outside website/static/data")
    client = helpers["client_for"](args.route, MODELS[args.route], args.local_url)
    calls = []
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = output_dir / f"{stamp}-{args.route}-{uuid4().hex[:8]}.json"
    record = {
        "probe": fixture["description"],
        "source_url": fixture["source_url"],
        "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "route": args.route,
        "model_requested": MODELS[args.route],
        "criterion": fixture["criterion"],
        "threshold": fixture["threshold"],
        "request_state": "canonical Unicode JSON text with question, context, answer",
        "human_labels_sent_to_models": False,
        "sdk_retries": 0,
        "sdk_versions": ({"typesafe-sdk": version("typesafe-sdk")}
                         if args.route == "jev" else {"openai": version("openai")}),
        "calls": calls,
    }

    def save() -> None:
        record["summary"] = summarize_probe(calls, prepared)
        output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")

    save()
    try:
        for case in prepared:
            call = helpers["measure"](
                client, args.route, MODELS[args.route], case,
                fixture["criterion"], "evaluation"
            )
            call["published_label"] = call.pop("human_label")
            call["scoring_status"] = case["scoring_status"]
            calls.append(call)
            save()
            print(case["case_id"], call["status"],
                  call.get("probability_correct"), flush=True)
            if len(calls) == 1 and call["status"] != "completed":
                break
    finally:
        close = getattr(client, "close", None)
        if close:
            close()

    print("Evidence:", output)
    print("Scored false acceptances:", record["summary"]["scored_false_accept_case_ids"])
    return 0 if len(calls) == 5 and all(call["status"] == "completed" for call in calls) else 1


if __name__ == "__main__":
    raise SystemExit(main())
