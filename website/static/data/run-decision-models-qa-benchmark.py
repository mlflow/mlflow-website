"""Re-run the 29-case decision-model comparison published beside this script.

Install ``openai>=3.26.0`` and ``typesafe-sdk>=0.5.7`` before a live run.
Without ``--execute``, the script only verifies the frozen cases, labels,
criterion, and published request hashes. A live run needs ``TYPESAFE_API_KEY``
for Jev or ``OPENAI_API_KEY`` for OpenAI. For an OpenJev-compatible server, pass
``--local-url``; set ``DECISIONS_COMPAT_API_KEY`` if that server requires auth.

Examples::

    python3 run-decision-models-qa-benchmark.py jev
    python3 run-decision-models-qa-benchmark.py jev --execute
    python3 run-decision-models-qa-benchmark.py openai --execute
    python3 run-decision-models-qa-benchmark.py openjev --execute \
        --local-url http://127.0.0.1:18080/v1
    python3 run-decision-models-qa-benchmark.py laya --execute \
        --local-url http://127.0.0.1:18080/v1

Each live run makes one unscored preflight call and 29 scored calls, with no
SDK retries. It writes a new timestamped directory under ``--output-dir``.
Human labels are joined to responses after inference and are never sent to a
model. The published benchmark file is only read, never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit
from uuid import uuid4


PUBLIC_DATA = Path(__file__).resolve().with_name("decision-models-qa-benchmark.json")
ROUTES = ("jev", "openai", "openjev", "laya")
THRESHOLD = 0.5


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Frozen benchmark validation failed: {message}")


def load_cases() -> tuple[dict, list[dict], dict[str, str]]:
    data = json.loads(PUBLIC_DATA.read_text(encoding="utf-8"))
    require(data.get("schema_version") == 2, "unsupported benchmark schema")
    source = data["source_dataset"]
    hashes = data["source_hashes"]
    snapshot = data["corrected_human_reference"]
    protocol = data["protocol"]

    dataset_hash = sha256(json.dumps(
        {"criterion": source["criterion"], "cases": source["cases"]}, sort_keys=True
    ))
    require(dataset_hash == source["dataset_sha256"] == hashes["dataset_sha256"],
            "dataset hash differs")
    require(sha256(canonical(snapshot)) == hashes["human_label_sha256"],
            "human-label hash differs")
    require(sha256(protocol["criterion"]) == hashes["criterion_sha256"],
            "criterion hash differs")
    require(protocol["questions"]["jev"]["correct"]["instructions"] == protocol["criterion"],
            "Jev question differs from criterion")
    require(protocol["questions"]["decisions"][0]["instructions"] == protocol["criterion"],
            "Decisions question differs from criterion")

    labels = {row["case_id"]: row["expectations"]["human_correctness"] for row in snapshot}
    case_ids = [case["case_id"] for case in source["cases"]]
    require(len(case_ids) == len(set(case_ids)) == len(labels) == 30, "expected 30 unique cases")
    require(set(case_ids) == set(labels), "case IDs and human labels differ")
    require(labels.get("QA-018") == "unclear", "QA-018 must remain excluded")

    cases = []
    request_hashes = {}
    for source_case in source["cases"]:
        case_id = source_case["case_id"]
        label = labels[case_id]
        if label == "unclear":
            continue
        require(label in ("correct", "incorrect"), f"unexpected label for {case_id}")
        state = {key: source_case[key] for key in ("question", "context", "answer")}
        request_hashes[case_id] = sha256(canonical(state))
        cases.append({"case_id": case_id, "state": state, "human_label": label})

    require(len(cases) == protocol["scored_cases"] == 29, "expected 29 scored cases")
    require(cases[0]["case_id"] == "QA-001", "preflight case changed")
    require({label: sum(case["human_label"] == label for case in cases)
             for label in ("correct", "incorrect")} == {"correct": 15, "incorrect": 14},
            "corrected label counts changed")
    require([case_id for case_id in case_ids if case_id not in request_hashes]
            == protocol["excluded_case_ids"], "excluded cases changed")

    # The request text is reconstructed from the frozen case. Every published
    # call records its hash, so a reader can check that all four routes saw it.
    for route in ROUTES:
        require(route in protocol["routes"] and route in data["runs"], f"missing {route} route")
        calls = data["runs"][route]["calls"]
        require(len(calls) == len(cases) + 1, f"{route} call count changed")
        for call in calls:
            require(call["request_sha256"] == request_hashes[call["case_id"]],
                    f"{route} {call['case_id']} request hash differs")
            require(call["model"] == protocol["routes"][route]["model"],
                    f"{route} {call['case_id']} model differs")
    return data, cases, request_hashes


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def client_for(route: str, model: str, local_url: str):
    if route == "jev":
        from typesafe_sdk import RetryPolicy, TypeSafeClient

        key = os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise ValueError("Set TYPESAFE_API_KEY for a live Jev run")
        return TypeSafeClient(api_key=key, base_url="https://api.typesafe.ai",
                              model=model, retry=RetryPolicy(max_retries=0), timeout=60)

    from openai import OpenAI

    if route == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("Set OPENAI_API_KEY for a live OpenAI run")
        return OpenAI(api_key=key, max_retries=0, timeout=60)

    # The SDK requires an API key even when a self-hosted server does not.
    key = os.environ.get("DECISIONS_COMPAT_API_KEY", "local-benchmark")
    return OpenAI(base_url=local_url, api_key=key, max_retries=0, timeout=120)


def measure(client, route: str, model: str, case: dict, criterion: str,
            phase: str) -> dict:
    request_state = canonical(case["state"])
    entry = {
        "case_id": case["case_id"],
        "phase": phase,
        "started_at": utc_now(),
        "request_sha256": sha256(request_state),
        "model_requested": model,
    }
    started = perf_counter()
    try:
        if route == "jev":
            from typesafe_sdk import Noul

            response = client.system_one(
                state=request_state,
                questions={"correct": Noul(instructions=criterion)},
            )
            probability = response.nouls["correct"].noul
        else:
            response = client.decisions.create(
                model=model,
                input=request_state,
                questions=[{"type": "predicate", "name": "correct", "instructions": criterion}],
            )
            if len(response.answers) != 1:
                raise ValueError("Expected exactly one answer")
            answer = response.answers[0]
            if answer.type != "predicate" or answer.name != "correct":
                raise ValueError("Expected the named predicate answer")
            probability = answer.probability

        if response.model != model:
            raise ValueError("Responding model differs from pinned requested model")
        if (not isinstance(probability, (int, float)) or not math.isfinite(probability)
                or not 0 <= probability <= 1):
            raise ValueError("Invalid predicate probability")
        usage = response.usage
        if usage is None or not isinstance(usage.input_tokens, int):
            raise ValueError("Missing input token usage")
        entry.update(
            status="completed",
            model=response.model,
            probability_correct=float(probability),
            prediction="correct" if probability >= THRESHOLD else "incorrect",
            usage={"input_tokens": usage.input_tokens,
                   "output_tokens": usage.output_tokens},
        )
    except Exception as exc:
        # Provider exception text and request IDs may contain sensitive details.
        entry.update(status="failed", error_type=type(exc).__name__)
        code = getattr(exc, "status_code", getattr(exc, "status", None))
        if isinstance(code, int):
            entry["http_status"] = code
    finally:
        entry["latency_ms"] = (perf_counter() - started) * 1000
    entry["human_label"] = case["human_label"]
    return entry


def summarize(calls: list[dict]) -> dict:
    scored = [call for call in calls if call["phase"] == "evaluation"]
    completed = [call for call in scored if call["status"] == "completed"]
    latencies = [call["latency_ms"] for call in completed]
    return {
        "scored_requests": len(scored),
        "completed_scored_requests": len(completed),
        "human_agreement_count": sum(call["prediction"] == call["human_label"]
                                     for call in completed),
        "human_agreement_denominator": len(completed),
        "false_accept_case_ids": [call["case_id"] for call in completed
                                  if call["human_label"] == "incorrect"
                                  and call["prediction"] == "correct"],
        "false_reject_case_ids": [call["case_id"] for call in completed
                                  if call["human_label"] == "correct"
                                  and call["prediction"] == "incorrect"],
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "max_latency_ms": max(latencies) if latencies else None,
        "input_tokens": sum(call["usage"]["input_tokens"] for call in completed),
        "output_tokens": sum(call["usage"]["output_tokens"] or 0 for call in completed),
        "preflight_status": calls[0]["status"] if calls else "not_run",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("route", choices=ROUTES)
    parser.add_argument("--execute", action="store_true", help="Make live model calls")
    parser.add_argument("--local-url", default="http://127.0.0.1:18080/v1",
                        help="OpenJev-compatible /v1 base URL for openjev or laya")
    parser.add_argument("--output-dir", type=Path,
                        default=Path.cwd() / "decision-models-qa-runs",
                        help="Root for new, timestamped run directories")
    args = parser.parse_args()

    data, cases, _ = load_cases()
    route = args.route
    model = data["protocol"]["routes"][route]["model"]
    if not args.execute:
        print(f"Verified {len(cases)} scored cases, labels, criterion, and four published runs")
        print(f"Ready for {model}: one unscored preflight and {len(cases)} scored calls")
        return 0

    parsed_url = urlsplit(args.local_url)
    if route in ("openjev", "laya") and (
        parsed_url.scheme not in ("http", "https") or not parsed_url.netloc
        or parsed_url.username or parsed_url.password
    ):
        parser.error("--local-url must be an http(s) URL without embedded credentials")
    output_root = args.output_dir.resolve()
    if output_root == PUBLIC_DATA.parent or PUBLIC_DATA.parent in output_root.parents:
        parser.error("--output-dir must be outside website/static/data")
    client = client_for(route, model, args.local_url)
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = output_root / f"{run_name}-{route}-{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)

    protocol = {
        "created_at_utc": utc_now(),
        "source_file": PUBLIC_DATA.name,
        "source_file_sha256": hashlib.sha256(PUBLIC_DATA.read_bytes()).hexdigest(),
        "source_hashes": data["source_hashes"],
        "route": route,
        "model_requested": model,
        "criterion": data["protocol"]["criterion"],
        "threshold": THRESHOLD,
        "request_state": data["protocol"]["request_state"],
        "request_question": data["protocol"]["questions"]["jev" if route == "jev" else "decisions"],
        "eligible_case_ids": [case["case_id"] for case in cases],
        "excluded_case_ids": data["protocol"]["excluded_case_ids"],
        "preflight": "one unscored QA-001 call before 29 scored calls",
        "concurrency": 1,
        "sdk_retries": 0,
        "human_labels_sent_to_models": False,
        "latency_scope": data["protocol"]["latency_scope"],
        "endpoint": ("https://api.typesafe.ai" if route == "jev" else
                     "https://api.openai.com/v1" if route == "openai" else args.local_url),
        "sdk_version": ({"typesafe-sdk": version("typesafe-sdk")} if route == "jev"
                        else {"openai": version("openai")}),
    }
    write_json(output / "protocol.json", protocol)

    calls = []
    schedule = [("preflight", cases[0]), *[("evaluation", case) for case in cases]]
    with (output / "calls.jsonl").open("w", encoding="utf-8") as ledger:
        for index, (phase, case) in enumerate(schedule, 1):
            entry = measure(client, route, model, case, protocol["criterion"], phase)
            calls.append(entry)
            ledger.write(canonical(entry) + "\n")
            ledger.flush()
            os.fsync(ledger.fileno())
            print(f"{index:02d}/30 {phase} {case['case_id']} "
                  f"{entry['status']} {entry['latency_ms']:.1f} ms", flush=True)
            if phase == "preflight" and entry["status"] != "completed":
                break
    write_json(output / "summary.json", summarize(calls))
    print(f"Evidence: {output}")
    return 0 if len(calls) == 30 and all(call["status"] == "completed" for call in calls) else 1


if __name__ == "__main__":
    raise SystemExit(main())
