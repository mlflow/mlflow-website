"""Reproduce the frozen MLflow QA pilot with OpenAI's Decisions API.

Install the recorded SDK version with ``python -m pip install openai==3.26.0``.
Run without arguments to validate the fixture and print the 30-call schedule.
Use ``--execute`` for live, billable calls; ``--env-file .env`` is optional when
OPENAI_API_KEY is already in the environment. Results go to a new local folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import statistics
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter


FIXTURE = Path(__file__).with_name("jev-openai-decisions-qa-benchmark.json")
MODEL = "gpt-6-luna"
THRESHOLD = 0.5
DATASET_SHA256 = "afa658c75976bf3a58c411ae66013eb9753014ff836f94688c2cee290aaa3dde"
LABEL_SHA256 = "b2a5161878eceb46e0df50152f5738e6a192c5420ab2fa410e6faaa3ac209e71"
CRITERION_SHA256 = "c22db3302b6a81031997f6e1cbb5476fb7dbda9faf824251cc3e7b8b11f1c76b"


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Frozen benchmark validation failed: {message}")


def load_cases() -> tuple[list[dict], dict]:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    dataset, labels, revision = (
        data["dataset"], data["corrected_human_reference"], data["source_revision"]
    )
    protocol = data["openai_decisions"]["protocol"]
    dataset_hash = sha256(json.dumps(
        {"criterion": dataset["criterion"], "cases": dataset["cases"]}, sort_keys=True
    ))
    label_hash = sha256(canonical(labels))
    criterion_hash = sha256(protocol["criterion"])
    require(data["schema_version"] == 1, "schema version")
    require(dataset_hash == dataset["dataset_sha256"] == protocol["dataset_sha256"] == DATASET_SHA256,
            "cases or dataset criterion hash")
    require(label_hash == revision["corrected_human_label_sha256"] == protocol["human_label_sha256"] == LABEL_SHA256,
            "corrected human label hash")
    require(criterion_hash == revision["jev_criterion_sha256"] == protocol["criterion_sha256"] == CRITERION_SHA256,
            "decision criterion hash")
    require(protocol["criterion"] == revision["jev_criterion"], "decision criterion text")

    case_ids = [case["case_id"] for case in dataset["cases"]]
    label_ids = [row["case_id"] for row in labels]
    require(case_ids == [f"QA-{n:03d}" for n in range(1, 31)], "case IDs or order")
    require(label_ids == case_ids, "label IDs or order")
    label_by_id = {row["case_id"]: row["expectations"]["human_correctness"] for row in labels}
    require(label_by_id["QA-018"] == "unclear", "QA-018 exclusion")
    eligible = [case for case in dataset["cases"] if label_by_id[case["case_id"]] in {"correct", "incorrect"}]
    eligible_ids = [case["case_id"] for case in eligible]
    require(len(eligible) == 29 and sum(label_by_id[case_id] == "correct" for case_id in eligible_ids) == 15,
            "eligible label counts")
    require(eligible_ids == protocol["eligible_case_ids"] == revision["jev_eligible_case_ids"],
            "eligible case IDs")
    require(protocol["excluded_case_ids"] == ["QA-018"], "excluded case IDs")
    recorded_schedule = [(call["phase"], call["case_id"]) for call in data["openai_decisions"]["calls"]]
    require(recorded_schedule == [("preflight", "QA-001"), *[("evaluation", case_id) for case_id in eligible_ids]],
            "preflight or scored call schedule")
    require(protocol["model_requested"] == MODEL and protocol["threshold"] == THRESHOLD,
            "model or threshold")
    require(protocol["api"] == "POST /v1/decisions" and protocol["concurrency"] == 1,
            "API or concurrency")
    require(protocol["sdk_max_retries"] == 0 and protocol["request_timeout_seconds"] == 60,
            "SDK retry or timeout settings")
    require(protocol["openai_sdk_version"] == "3.26.0" and protocol["price_input_usd_per_million"] == 0.1,
            "SDK version or input price")
    require(protocol["request_input"] == "canonical UTF-8 JSON string containing only question, context, answer",
            "request input format")
    require(protocol["request_questions"] == [{"type": "predicate", "name": "correct", "instructions": protocol["criterion"]}],
            "predicate question")
    require(protocol["prediction"] == "correct if predicate probability >= 0.5, else incorrect",
            "prediction rule")
    require(protocol["human_labels_sent_to_provider"] is False, "label transmission setting")
    return [{"case_id": case["case_id"],
             "state": {key: case[key] for key in ("question", "context", "answer")},
             "human_label": label_by_id[case["case_id"]]} for case in eligible], protocol


def load_api_key(env_file: Path | None) -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    if env_file is not None:
        try:
            lines = env_file.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            raise ValueError("Cannot read --env-file") from None
        for line in lines:
            assignment = line.strip().removeprefix("export ").lstrip()
            name, equals, value = assignment.partition("=")
            if equals and name.strip() == "OPENAI_API_KEY":
                value = value.strip()
                try:
                    fields = shlex.split(value, comments=True)
                except ValueError:
                    raise ValueError("Invalid OPENAI_API_KEY assignment in --env-file") from None
                if len(fields) == 1 and fields[0]:
                    return fields[0]
    raise ValueError("Set OPENAI_API_KEY or provide it with --env-file")


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    point = (len(ordered) - 1) * percent / 100
    lower, upper = math.floor(point), math.ceil(point)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (point - lower)


def summarize(calls: list[dict], price: float) -> dict:
    scored = [call for call in calls if call["phase"] == "evaluation"]
    complete = [call for call in scored if call["status"] == "completed"]
    latencies = [call["latency_ms"] for call in complete]
    priced = [call["estimated_cost_usd"] for call in complete]
    all_priced = [call["estimated_cost_usd"] for call in calls if "estimated_cost_usd" in call]
    return {
        "eligible_cases": len(scored),
        "scored_requests": len(scored),
        "completed_scored_requests": len(complete),
        "failed_scored_requests": len(scored) - len(complete),
        "human_agreement_count": sum(call["prediction"] == call["human_label"] for call in complete),
        "human_agreement_denominator": len(complete),
        "false_accept_case_ids": [call["case_id"] for call in complete
                                  if call["human_label"] == "incorrect" and call["prediction"] == "correct"],
        "false_reject_case_ids": [call["case_id"] for call in complete
                                  if call["human_label"] == "correct" and call["prediction"] == "incorrect"],
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_latency_ms": percentile(latencies, 95) if latencies else None,
        "scored_input_tokens": sum(call["usage"]["input_tokens"] for call in complete),
        "scored_output_tokens": sum(call["usage"]["output_tokens"] for call in complete),
        "scored_estimated_cost_usd": sum(priced),
        "estimated_cost_per_1000_scored_usd": 1000 * sum(priced) / len(priced) if priced else None,
        "total_requests_including_preflight": len(calls),
        "total_estimated_cost_including_preflight_usd": sum(all_priced),
        "cost_missing_requests": len(calls) - len(all_priced),
        "price_input_usd_per_million": price,
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Make 30 live, billable API requests")
    parser.add_argument("--env-file", type=Path, help="Optional .env file; the environment takes precedence")
    parser.add_argument("--output-dir", type=Path, help="New result folder (default: timestamped folder in current directory)")
    args = parser.parse_args()
    cases, protocol = load_cases()
    if not args.execute:
        print(f"Validated frozen hashes and labels: 1 QA-001 preflight + {len(cases)} scored calls; no API calls made.")
        return 0

    from openai import OpenAI

    sdk_version = version("openai")
    require(sdk_version == protocol["openai_sdk_version"] == "3.26.0", "OpenAI SDK version")
    client = OpenAI(api_key=load_api_key(args.env_file), max_retries=0, timeout=60)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = args.output_dir or Path.cwd() / f"openai-decisions-qa-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fixture": FIXTURE.name,
        "dataset_sha256": DATASET_SHA256,
        "human_label_sha256": LABEL_SHA256,
        "criterion_sha256": CRITERION_SHA256,
        "model": MODEL,
        "api": protocol["api"],
        "openai_sdk_version": sdk_version,
        "sdk_max_retries": 0,
        "request_timeout_seconds": 60,
        "concurrency": 1,
        "request_input": protocol["request_input"],
        "threshold": THRESHOLD,
        "questions": protocol["request_questions"],
        "price_input_usd_per_million": protocol["price_input_usd_per_million"],
        "price_source": protocol["price_source"],
    })

    calls: list[dict] = []
    schedule = [("preflight", cases[0]), *[("evaluation", case) for case in cases]]
    with (output / "calls.jsonl").open("x", encoding="utf-8") as ledger:
        for sequence, (phase, case) in enumerate(schedule, start=1):
            entry = {"sequence": sequence, "phase": phase, "case_id": case["case_id"],
                     "human_label": case["human_label"],
                     "started_at": datetime.now(timezone.utc).isoformat()}
            started = perf_counter()
            try:
                response = client.decisions.create(
                    model=MODEL,
                    input=canonical(case["state"]),
                    questions=protocol["request_questions"],
                )
                if len(response.answers) != 1:
                    raise ValueError("Expected one answer")
                answer = response.answers[0]
                if answer.type == "refusal":
                    entry.update(status="refused", answer=answer.model_dump(mode="json"))
                else:
                    if answer.type != "predicate" or answer.name != "correct":
                        raise ValueError("Unexpected answer type or name")
                    probability = answer.probability
                    if not math.isfinite(probability) or not 0 <= probability <= 1:
                        raise ValueError("Invalid predicate probability")
                    usage = response.usage.model_dump(mode="json")
                    if not isinstance(usage.get("input_tokens"), int) or not isinstance(usage.get("output_tokens"), int):
                        raise ValueError("Missing token usage")
                    entry.update(
                        status="completed",
                        model=response.model,
                        request_id=getattr(response, "_request_id", None),
                        probability_correct=probability,
                        prediction="correct" if probability >= THRESHOLD else "incorrect",
                        answer=answer.model_dump(mode="json"),
                        usage=usage,
                        estimated_cost_usd=usage["input_tokens"] * protocol["price_input_usd_per_million"] / 1_000_000,
                    )
            except Exception as exc:
                entry.update(status="failed", error_type=type(exc).__name__)
                status_code = getattr(exc, "status_code", None)
                if isinstance(status_code, int):
                    entry["http_status"] = status_code
            finally:
                entry["latency_ms"] = (perf_counter() - started) * 1000
            calls.append(entry)
            ledger.write(canonical(entry) + "\n")
            ledger.flush()
            os.fsync(ledger.fileno())
            print(f"{sequence:02d}/30 {phase:10s} {case['case_id']} {entry['status']} {entry['latency_ms']:.1f} ms", flush=True)

    write_json(output / "summary.json", summarize(calls, protocol["price_input_usd_per_million"]))
    print(f"Results: {output}")
    return 0 if all(call["status"] == "completed" for call in calls) else 1


if __name__ == "__main__":
    raise SystemExit(main())
