#!/usr/bin/env python3
"""Compute deterministic paired statistics for the four Phase-4 modes."""

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path


PAIRED_MODES = ("direct", "static_rag", "base_search", "search_rl")
SOURCES = ("nq", "hotpotqa")
PRIMARY_COMPARISON = (
    "search_rl_vs_base_search",
    "search_rl",
    "base_search",
)
SECONDARY_COMPARISONS = (
    ("base_search_vs_direct", "base_search", "direct"),
    ("static_rag_vs_direct", "static_rag", "direct"),
    ("search_rl_vs_static_rag", "search_rl", "static_rag"),
    ("search_rl_vs_direct", "search_rl", "direct"),
    ("base_search_vs_static_rag", "base_search", "static_rag"),
)


def _record_map(records, mode):
    mapped = {}
    for position, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{mode} row {position} is not a JSON object")
        uid = record.get("uid")
        if not isinstance(uid, str) or not uid:
            raise ValueError(f"{mode} row {position} has a missing UID")
        if uid in mapped:
            raise ValueError(f"{mode} results contain duplicate UID: {uid}")
        if record.get("mode") != mode:
            raise ValueError(
                f"{mode} result {uid} declares mode {record.get('mode')!r}"
            )
        mapped[uid] = record
    return mapped


def _artifact_binding(record, mode, uid):
    run_config = record.get("run_config")
    if not isinstance(run_config, dict):
        raise ValueError(f"{mode} result {uid} has no run_config binding")
    eval_sha256 = run_config.get("eval_sha256")
    manifest_sha256 = run_config.get("eval_manifest_sha256")
    if not isinstance(eval_sha256, str) or not eval_sha256:
        raise ValueError(f"{mode} result {uid} is not bound to an eval parquet hash")
    if not isinstance(manifest_sha256, str) or not manifest_sha256:
        raise ValueError(f"{mode} result {uid} is not bound to an eval manifest hash")
    return eval_sha256, manifest_sha256


def join_result_sets(result_sets, modes):
    """Strictly join named result sets by UID and validate paired identity fields."""

    modes = tuple(modes)
    if len(modes) < 2 or len(modes) != len(set(modes)):
        raise ValueError("paired analysis modes must be at least two unique names")
    if set(result_sets) != set(modes):
        raise ValueError(
            "paired analysis requires exactly these modes: "
            f"{modes}"
        )
    mapped = {
        mode: _record_map(result_sets[mode], mode)
        for mode in modes
    }
    uid_sets = {mode: set(records) for mode, records in mapped.items()}
    baseline_mode = modes[0]
    baseline_uids = uid_sets[baseline_mode]
    if not baseline_uids:
        raise ValueError("paired analysis requires at least one UID")
    for mode in modes[1:]:
        if uid_sets[mode] != baseline_uids:
            missing = sorted(baseline_uids - uid_sets[mode])
            unexpected = sorted(uid_sets[mode] - baseline_uids)
            raise ValueError(
                f"{mode} results have a nonidentical UID set; "
                f"missing={missing}, unexpected={unexpected}"
            )

    joined = []
    bindings = set()
    for uid in sorted(baseline_uids):
        records = {mode: mapped[mode][uid] for mode in modes}
        baseline = records[baseline_mode]
        baseline_identity = (
            baseline.get("question"),
            baseline.get("ground_truth"),
            baseline.get("data_source"),
        )
        if baseline_identity[2] not in SOURCES:
            raise ValueError(
                f"paired result {uid} has unsupported data_source: "
                f"{baseline_identity[2]!r}"
            )
        outcomes = {}
        for mode, record in records.items():
            identity = (
                record.get("question"),
                record.get("ground_truth"),
                record.get("data_source"),
            )
            if identity[0] != baseline_identity[0]:
                raise ValueError(f"{mode} result {uid} has a mismatched question")
            if identity[1] != baseline_identity[1]:
                raise ValueError(f"{mode} result {uid} has mismatched ground truth")
            if identity[2] != baseline_identity[2]:
                raise ValueError(f"{mode} result {uid} has a mismatched data source")
            exact_match = record.get("exact_match")
            if isinstance(exact_match, bool) or exact_match not in (0, 1):
                raise ValueError(f"{mode} result {uid} has non-binary exact_match")
            outcomes[mode] = int(exact_match)
            bindings.add(_artifact_binding(record, mode, uid))
        joined.append({
            "uid": uid,
            "question": baseline_identity[0],
            "ground_truth": baseline_identity[1],
            "data_source": baseline_identity[2],
            "outcomes": outcomes,
            "records": records,
        })

    if len(bindings) != 1:
        raise ValueError(
            "result files are not bound to the same eval manifest/parquet hashes: "
            f"{sorted(bindings)}"
        )
    return joined


def join_paired_results(result_sets):
    """Strictly join the four Phase-4 modes without changing its public API."""

    return join_result_sets(result_sets, PAIRED_MODES)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_joined_against_manifest(joined, manifest_path):
    """Verify that paired rows are bound to the supplied audited eval manifest."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha256 = _sha256_file(manifest_path)
    first_record = joined[0]["records"]["direct"]
    eval_sha256, bound_manifest_sha256 = _artifact_binding(
        first_record, "direct", joined[0]["uid"]
    )
    if bound_manifest_sha256 != manifest_sha256:
        raise ValueError(
            "paired results are not bound to the supplied eval manifest: "
            f"results={bound_manifest_sha256}, supplied={manifest_sha256}"
        )
    manifest_eval_sha256 = manifest.get("output", {}).get("sha256")
    if eval_sha256 != manifest_eval_sha256:
        raise ValueError(
            "paired results are not bound to the eval parquet declared by the "
            f"supplied manifest: results={eval_sha256}, "
            f"manifest={manifest_eval_sha256}"
        )
    expected_uids = [
        entry.get("uid") for entry in manifest.get("selected_source_rows", [])
    ]
    if not expected_uids or len(expected_uids) != len(set(expected_uids)):
        raise ValueError("supplied eval manifest has missing or duplicate selected UIDs")
    actual_uids = {row["uid"] for row in joined}
    if set(expected_uids) != actual_uids:
        missing = sorted(set(expected_uids) - actual_uids)
        unexpected = sorted(actual_uids - set(expected_uids))
        raise ValueError(
            "paired result UIDs do not match the supplied eval manifest; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if manifest.get("selected_row_count") != len(expected_uids):
        raise ValueError("supplied eval manifest selected-row count is inconsistent")
    return {
        "path": str(manifest_path),
        "sha256": manifest_sha256,
        "eval_parquet_sha256": manifest_eval_sha256,
        "selected_uid_count": len(expected_uids),
        "uid_set_matches": True,
    }


def _slice_rows(joined, source):
    if source is None:
        rows = list(joined)
    elif source in SOURCES:
        rows = [row for row in joined if row["data_source"] == source]
    else:
        raise ValueError(f"unsupported source slice: {source!r}")
    if not rows:
        label = "overall" if source is None else source
        raise ValueError(f"paired analysis has no rows for {label}")
    return rows


def _validate_comparison_modes(joined, mode_a, mode_b):
    if not joined or not isinstance(joined[0].get("outcomes"), dict):
        raise ValueError("paired comparison requires joined outcome rows")
    available_modes = set(joined[0]["outcomes"])
    if mode_a not in available_modes or mode_b not in available_modes:
        raise ValueError(f"unsupported paired comparison: {mode_a!r}, {mode_b!r}")
    if mode_a == mode_b:
        raise ValueError("paired comparison modes must differ")


def _percentile(values, probability):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a percentile of no values")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def paired_bootstrap(
    joined,
    mode_a,
    mode_b,
    *,
    source=None,
    samples=10_000,
    seed=42,
    confidence_level=0.95,
):
    """Return a paired percentile interval for EM(A) - EM(B), in points."""

    _validate_comparison_modes(joined, mode_a, mode_b)
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError("bootstrap samples must be a positive integer")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence level must be between zero and one")
    rows = _slice_rows(joined, source)
    differences = {
        source_name: [
            row["outcomes"][mode_a] - row["outcomes"][mode_b]
            for row in rows
            if row["data_source"] == source_name
        ]
        for source_name in SOURCES
    }
    active_sources = (source,) if source is not None else SOURCES
    if any(not differences[source_name] for source_name in active_sources):
        raise ValueError("paired bootstrap requires rows in every requested source")

    rng = random.Random(seed)
    bootstrap_differences = []
    for _ in range(samples):
        total = 0
        count = 0
        for source_name in active_sources:
            source_differences = differences[source_name]
            source_count = len(source_differences)
            for _ in range(source_count):
                total += source_differences[rng.randrange(source_count)]
            count += source_count
        bootstrap_differences.append(100.0 * total / count)

    observed = 100.0 * sum(
        row["outcomes"][mode_a] - row["outcomes"][mode_b]
        for row in rows
    ) / len(rows)
    alpha = 1.0 - confidence_level
    lower = _percentile(bootstrap_differences, alpha / 2.0)
    upper = _percentile(bootstrap_differences, 1.0 - alpha / 2.0)
    interval_method = (
        "source-stratified paired percentile"
        if source is None else "within-source paired percentile"
    )
    return {
        "difference_pp": observed,
        "bootstrap_ci_95_pp": [lower, upper],
        "ci_includes_zero": lower <= 0.0 <= upper,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "confidence_level": confidence_level,
        "interval_method": interval_method,
    }


def exact_mcnemar(joined, mode_a, mode_b, *, source=None):
    """Compute the exact two-sided McNemar test without SciPy."""

    _validate_comparison_modes(joined, mode_a, mode_b)
    rows = _slice_rows(joined, source)
    both_correct = 0
    a_only_correct = 0
    b_only_correct = 0
    both_incorrect = 0
    for row in rows:
        outcome_a = row["outcomes"][mode_a]
        outcome_b = row["outcomes"][mode_b]
        if outcome_a and outcome_b:
            both_correct += 1
        elif outcome_a:
            a_only_correct += 1
        elif outcome_b:
            b_only_correct += 1
        else:
            both_incorrect += 1
    discordant = a_only_correct + b_only_correct
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(a_only_correct, b_only_correct)
        lower_tail_numerator = sum(
            math.comb(discordant, value) for value in range(smaller + 1)
        )
        p_value = min(1.0, 2.0 * lower_tail_numerator / (2 ** discordant))
    return {
        "mode_a": mode_a,
        "mode_b": mode_b,
        "orientation": f"{mode_a} minus {mode_b}",
        "both_correct": both_correct,
        "a_only_correct": a_only_correct,
        "b_only_correct": b_only_correct,
        "both_incorrect": both_incorrect,
        "discordant_pair_count": discordant,
        "exact_two_sided_p_value": p_value,
    }


def analyze_comparison(
    joined,
    comparison,
    mode_a,
    mode_b,
    *,
    bootstrap_samples=10_000,
    seed=42,
    confidence_level=0.95,
    classification,
):
    result = {
        "comparison": comparison,
        "classification": classification,
        "mode_a": mode_a,
        "mode_b": mode_b,
        "orientation": f"EM({mode_a}) - EM({mode_b})",
    }
    for label, source in (("overall", None), ("nq", "nq"), ("hotpotqa", "hotpotqa")):
        bootstrap = paired_bootstrap(
            joined,
            mode_a,
            mode_b,
            source=source,
            samples=bootstrap_samples,
            seed=seed,
            confidence_level=confidence_level,
        )
        result[label] = {
            "sample_size": len(_slice_rows(joined, source)),
            **bootstrap,
            "mcnemar": exact_mcnemar(
                joined, mode_a, mode_b, source=source
            ),
        }
    return result


def _readiness(joined):
    source_counts = Counter(row["data_source"] for row in joined)
    agent_records = [
        row["records"][mode]
        for row in joined
        for mode in ("base_search", "search_rl")
    ]
    telemetry_complete = all(
        isinstance(record.get("search_retrieval_failure_count"), int)
        and not isinstance(record.get("search_retrieval_failure_count"), bool)
        and record["search_retrieval_failure_count"] >= 0
        for record in agent_records
    )
    retrieval_failure_counts = {
        mode: sum(
            row["records"][mode].get("search_retrieval_failure_count", 0)
            for row in joined
            if isinstance(
                row["records"][mode].get("search_retrieval_failure_count"), int
            )
            and not isinstance(
                row["records"][mode].get("search_retrieval_failure_count"), bool
            )
        )
        for mode in ("base_search", "search_rl")
    }
    retrieval_failure_count = sum(retrieval_failure_counts.values())
    evaluation_error_count = sum(
        bool(record.get("evaluation_error"))
        for row in joined
        for record in row["records"].values()
    )
    checks = {
        "all_four_files_have_exactly_64_rows": len(joined) == 64,
        "identical_uid_sets": True,
        "source_balance_is_32_nq_32_hotpotqa": (
            source_counts == Counter({"nq": 32, "hotpotqa": 32})
        ),
        "same_eval_manifest_and_parquet_hashes": True,
        "agent_retrieval_telemetry_complete": telemetry_complete,
        "no_agent_retrieval_failures": (
            telemetry_complete and retrieval_failure_count == 0
        ),
        "no_evaluation_errors": evaluation_error_count == 0,
    }
    ready = all(checks.values())
    failures = [name for name, passed in checks.items() if not passed]
    warning = None
    if failures:
        warning = (
            "Four-mode quality and inference claims are not ready; failed checks: "
            + ", ".join(failures)
            + ". Paired descriptive output is retained for audit, but failed rows "
              "must not be silently omitted."
        )
    return {
        "quality_claim_ready": ready,
        "inference_claim_ready": ready,
        "checks": checks,
        "row_count": len(joined),
        "source_counts": {source: source_counts.get(source, 0) for source in SOURCES},
        "agent_retrieval_failure_count": retrieval_failure_count,
        "agent_retrieval_failure_counts": retrieval_failure_counts,
        "evaluation_error_count": evaluation_error_count,
        "warning": warning,
    }


def _primary_interpretation(primary, readiness):
    overall = primary["overall"]
    difference = overall["difference_pp"]
    supported = (
        not overall["ci_includes_zero"]
        and overall["mcnemar"]["exact_two_sided_p_value"] < 0.05
    )
    if not readiness["inference_claim_ready"]:
        conclusion = (
            "The paired calculations are auditable, but four-mode inference-claim "
            "readiness checks did not pass, so no quality or inferential claim "
            "should be made from this artifact."
        )
    elif supported and difference > 0:
        conclusion = (
            "The measured Search-RL improvement over Base Search is supported by "
            "both the paired bootstrap interval and exact McNemar test on this "
            "64-example benchmark."
        )
    elif supported and difference < 0:
        conclusion = (
            "The measured difference favors Base Search and is supported by both "
            "the paired bootstrap interval and exact McNemar test on this "
            "64-example benchmark."
        )
    else:
        if difference > 0:
            favored = "Search-RL"
        elif difference < 0:
            favored = "Base Search"
        else:
            favored = "neither mode"
        conclusion = (
            f"The observed point estimate favors {favored}, but this 64-example "
            "benchmark does not provide strong paired statistical evidence of a "
            "difference."
        )
    scope = (
        "This is a deterministic held-out benchmark with n = 64 "
        "(32 NQ / 32 HotpotQA). Its limited sample size and exact-match's sparse "
        "binary signal constrain interpretation."
    )
    return f"{conclusion} {scope}"


def analyze_paired_results(
    result_sets,
    *,
    bootstrap_samples=10_000,
    seed=42,
    confidence_level=0.95,
):
    joined = join_paired_results(result_sets)
    readiness = _readiness(joined)
    first_record = joined[0]["records"]["direct"]
    eval_sha256, eval_manifest_sha256 = _artifact_binding(
        first_record, "direct", joined[0]["uid"]
    )
    primary_name, primary_a, primary_b = PRIMARY_COMPARISON
    primary = analyze_comparison(
        joined,
        primary_name,
        primary_a,
        primary_b,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        confidence_level=confidence_level,
        classification="primary",
    )
    secondary = {
        name: analyze_comparison(
            joined,
            name,
            mode_a,
            mode_b,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
            confidence_level=confidence_level,
            classification="secondary_exploratory",
        )
        for name, mode_a, mode_b in SECONDARY_COMPARISONS
    }
    primary["interpretation"] = _primary_interpretation(primary, readiness)
    return {
        "configuration": {
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": seed,
            "confidence_level": confidence_level,
            "interval": "percentile",
            "overall_bootstrap": "source-stratified paired percentile",
            "source_bootstrap": "within-source paired percentile",
            "mcnemar": "exact two-sided binomial on discordant pairs",
            "eval_parquet_sha256": eval_sha256,
            "eval_manifest_sha256": eval_manifest_sha256,
        },
        "quality_claim_ready": readiness["quality_claim_ready"],
        "inference_claim_ready": readiness["inference_claim_ready"],
        "readiness": readiness,
        "primary": primary,
        "secondary": secondary,
        "interpretation_guardrails": {
            "primary_estimand": (
                "The primary paired difference measures the result associated with "
                "20-step GRPO post-training while holding the Search Agent and "
                "retrieval configuration fixed; it remains specific to this run."
            ),
            "primary": primary["interpretation"],
            "secondary": (
                "Secondary comparisons are exploratory. Their p-values are not "
                "independently confirmatory evidence and cannot support broad or "
                "guaranteed claims."
            ),
            "scope": (
                "Results apply only to this deterministic held-out benchmark; "
                "they should not be generalized beyond this evaluation."
            ),
        },
    }


def _format_number(value):
    return f"{float(value):.6g}"


def render_markdown(report):
    readiness = report["readiness"]
    lines = [
        "# Phase-4 paired statistics",
        "",
        f"- Quality claim ready: `{str(readiness['quality_claim_ready']).lower()}`",
        f"- Inference claim ready: `{str(readiness['inference_claim_ready']).lower()}`",
        f"- Rows: {readiness['row_count']} "
        f"({readiness['source_counts']['nq']} NQ / "
        f"{readiness['source_counts']['hotpotqa']} HotpotQA)",
    ]
    if readiness["warning"]:
        lines.extend(["", f"> {readiness['warning']}"])
    lines.extend([
        "",
        "## Primary comparison",
        "",
        "| Comparison | Slice | Difference (pp) | 95% paired-bootstrap CI | Exact McNemar p |",
        "|---|---:|---:|---:|---:|",
    ])
    primary = report["primary"]
    for slice_name in ("overall", "nq", "hotpotqa"):
        values = primary[slice_name]
        lower, upper = values["bootstrap_ci_95_pp"]
        lines.append(
            f"| {primary['comparison']} | {slice_name} | "
            f"{_format_number(values['difference_pp'])} | "
            f"[{_format_number(lower)}, {_format_number(upper)}] | "
            f"{_format_number(values['mcnemar']['exact_two_sided_p_value'])} |"
        )
    lines.extend(["", primary["interpretation"], "", "## Secondary comparisons", ""])
    lines.extend([
        "| Comparison | Overall difference (pp) | 95% paired-bootstrap CI | Exact McNemar p |",
        "|---|---:|---:|---:|",
    ])
    for comparison in report["secondary"].values():
        overall = comparison["overall"]
        lower, upper = overall["bootstrap_ci_95_pp"]
        lines.append(
            f"| {comparison['comparison']} | "
            f"{_format_number(overall['difference_pp'])} | "
            f"[{_format_number(lower)}, {_format_number(upper)}] | "
            f"{_format_number(overall['mcnemar']['exact_two_sided_p_value'])} |"
        )
    lines.extend([
        "",
        report["interpretation_guardrails"]["secondary"],
        "",
        report["interpretation_guardrails"]["scope"],
        "",
    ])
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    return parser.parse_args()


def output_paths(results_dir, output=None, markdown_output=None):
    results_dir = Path(results_dir).expanduser().resolve()
    json_path = (
        Path(output).expanduser().resolve()
        if output else results_dir / "paired_statistics.json"
    )
    markdown_path = (
        Path(markdown_output).expanduser().resolve()
        if markdown_output else results_dir / "paired_statistics.md"
    )
    return json_path, markdown_path


def main():
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    from experiments.phase4_benchmark.run_benchmark import read_result_file

    result_sets = {
        mode: read_result_file(results_dir / f"{mode}.jsonl", mode)
        for mode in PAIRED_MODES
    }
    joined = join_paired_results(result_sets)
    manifest_audit = validate_joined_against_manifest(joined, args.eval_manifest)
    report = analyze_paired_results(
        result_sets,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        confidence_level=args.confidence_level,
    )
    report["configuration"]["eval_manifest_audit"] = manifest_audit
    output_path, markdown_path = output_paths(
        results_dir, args.output, args.markdown_output
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Phase-4 paired-statistics Markdown: {markdown_path}")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Phase-4 paired statistics: {output_path}")


if __name__ == "__main__":
    main()
