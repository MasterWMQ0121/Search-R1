import importlib.util
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pytest

from experiments.phase6_retriever_serving import build_ivfpq_index as builder


HAS_FAISS = importlib.util.find_spec("faiss") is not None


def _isolated_build_worker(connection, config, kwargs, reject_verification):
    """Run native FAISS training like the real builder CLI: in a fresh process."""

    try:
        if reject_verification:
            def reject(*args, **unused_kwargs):
                raise ValueError("synthetic verification failure")

            builder.verify_ivfpq_index = reject
        result = builder.build_ivfpq_index(config, **kwargs)
        connection.send(("ok", result))
    except BaseException as error:
        connection.send(("error", type(error).__name__, str(error)))
    finally:
        connection.close()


def _run_isolated_build(config, *, reject_verification=False, **kwargs):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_isolated_build_worker,
        args=(child, config, kwargs, reject_verification),
    )
    process.start()
    child.close()
    process.join(60)
    if process.is_alive():
        process.terminate()
        process.join(10)
        pytest.fail("isolated FAISS build test exceeded 60 seconds")
    assert process.exitcode == 0, (
        "isolated FAISS build process crashed; "
        f"exitcode={process.exitcode}"
    )
    assert parent.poll(), "isolated FAISS build returned no result"
    return parent.recv()


def test_training_ranges_are_deterministic_exact_disjoint_and_broad():
    first = builder.plan_training_ranges(10_000_003, 257, block_size=32, seed=42)
    second = builder.plan_training_ranges(10_000_003, 257, block_size=32, seed=42)
    other_seed = builder.plan_training_ranges(10_000_003, 257, block_size=32, seed=43)
    assert first == second
    assert first != other_seed
    assert sum(item["count"] for item in first) == 257
    assert len(first) == 9
    assert all(item["count"] <= 32 for item in first)
    assert all(left["end"] <= right["start"] for left, right in zip(first, first[1:]))
    assert first[0]["start"] < 10_000_003 // 9
    assert first[-1]["start"] >= (8 * 10_000_003) // 9


@pytest.mark.parametrize(
    "ntotal,sample,block",
    [(0, 1, 1), (10, 0, 1), (10, 11, 2), (10, 2, 0)],
)
def test_training_range_validation(ntotal, sample, block):
    with pytest.raises(ValueError):
        builder.plan_training_ranges(ntotal, sample, block)


def _write_flat(path, metric="ip", count=256, dimension=8):
    if not HAS_FAISS:
        pytest.skip("faiss-cpu is unavailable")
    import faiss

    rng = np.random.default_rng(123)
    vectors = rng.normal(size=(count, dimension)).astype(np.float32)
    if metric == "ip":
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        index = faiss.IndexFlatIP(dimension)
    else:
        index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    faiss.write_index(index, str(path))
    return vectors


def _config(source, output, **overrides):
    values = {
        "source_index": str(source),
        "output_index": str(output),
        "nlist": 4,
        "m": 2,
        "nbits": 4,
        "training_sample_size": 64,
        "training_block_size": 16,
        "add_chunk_size": 31,
        "seed": 42,
        "expected_metric": "inner_product",
        "disk_safety_factor": 1.0,
        "memory_safety_factor": 1.0,
        "progress_interval_vectors": 40,
    }
    values.update(overrides)
    return builder.IVFPQBuildConfig(**values)


@pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu is unavailable")
def test_synthetic_flat_to_ivfpq_is_atomic_hash_bound_and_preserves_ids(tmp_path):
    import faiss

    source = tmp_path / "source.index"
    output = tmp_path / "candidate.index"
    _write_flat(source)
    source_bytes = source.read_bytes()
    source_sha = builder.sha256_file(source)
    config = _config(source, output)
    status, result = _run_isolated_build(
        config,
        available_disk_bytes=10**15,
        available_ram_bytes=10**15,
    )

    assert status == "ok"
    assert result["status"] == "built"
    assert source.read_bytes() == source_bytes
    assert builder.sha256_file(source) == source_sha
    assert output.is_file()
    assert not (tmp_path / "candidate.index.tmp").exists()
    assert not (tmp_path / "candidate.build-state.json").exists()
    manifest_path = tmp_path / "candidate.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["source"]["sha256"] == source_sha
    assert manifest["output"]["sha256"] == builder.sha256_file(output)
    assert manifest["source_sha256_unchanged"] is True
    assert manifest["training_sample_vector_count"] == 64
    assert manifest["resume_semantics"] == "restart_from_zero_only"
    assert all(manifest["verification"]["checks"].values())

    verified = builder.build_ivfpq_index(config, verify_only=True)
    assert verified["status"] == "verified"
    assert verified["manifest_sha256"] == builder.sha256_file(manifest_path)
    stale_manifest = dict(manifest)
    stale_manifest["config_fingerprint"] = "different-config"
    manifest_path.write_text(json.dumps(stale_manifest))
    with pytest.raises(ValueError, match="different build config"):
        builder.build_ivfpq_index(config, verify_only=True)
    manifest_path.write_text(json.dumps(manifest))

    candidate = faiss.read_index(str(output))
    stored_ids = []
    for list_number in range(candidate.nlist):
        size = candidate.invlists.list_size(list_number)
        if size:
            stored_ids.extend(
                faiss.rev_swig_ptr(candidate.invlists.get_ids(list_number), size).tolist()
            )
    assert sorted(stored_ids) == list(range(256))

    reused = builder.build_ivfpq_index(
        config,
        available_disk_bytes=10**15,
        available_ram_bytes=10**15,
    )
    assert reused["status"] == "existing_complete_verified"


@pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu is unavailable")
def test_dimension_metric_and_training_size_validation(tmp_path):
    source = tmp_path / "source.index"
    output = tmp_path / "candidate.index"
    _write_flat(source, dimension=7)
    with pytest.raises(ValueError, match="divisible"):
        builder.build_ivfpq_index(
            _config(source, output),
            preflight_only=True,
            available_disk_bytes=10**15,
            available_ram_bytes=10**15,
        )
    source_l2 = tmp_path / "source-l2.index"
    _write_flat(source_l2, metric="l2")
    with pytest.raises(ValueError, match="source metric"):
        builder.build_ivfpq_index(
            _config(source_l2, output),
            preflight_only=True,
            available_disk_bytes=10**15,
            available_ram_bytes=10**15,
        )
    with pytest.raises(ValueError, match="at least nlist"):
        builder.validate_config(_config(source_l2, output, nlist=65))


@pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu is unavailable")
def test_preflight_failure_refuses_build_and_preserves_source(tmp_path):
    source = tmp_path / "source.index"
    output = tmp_path / "candidate.index"
    _write_flat(source)
    before = source.read_bytes()
    with pytest.raises(RuntimeError, match="preflight failed"):
        builder.build_ivfpq_index(
            _config(source, output),
            available_disk_bytes=1,
            available_ram_bytes=10**15,
        )
    assert source.read_bytes() == before
    assert not output.exists()


@pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu is unavailable")
def test_stale_temp_requires_validated_resume_or_overwrite(tmp_path):
    source = tmp_path / "source.index"
    output = tmp_path / "candidate.index"
    _write_flat(source)
    config = _config(source, output)
    temporary = tmp_path / "candidate.index.tmp"
    state = tmp_path / "candidate.build-state.json"
    temporary.write_bytes(b"partial")

    with pytest.raises(FileExistsError, match="stale/incomplete"):
        builder.build_ivfpq_index(
            config,
            available_disk_bytes=10**15,
            available_ram_bytes=10**15,
        )
    with pytest.raises(ValueError, match="build-state"):
        builder.build_ivfpq_index(
            config,
            resume=True,
            available_disk_bytes=10**15,
            available_ram_bytes=10**15,
        )

    state.write_text(json.dumps({
        "config_fingerprint": "wrong",
        "source_sha256": builder.sha256_file(source),
    }))
    with pytest.raises(ValueError, match="different IVF-PQ configuration"):
        builder.build_ivfpq_index(
            config,
            resume=True,
            available_disk_bytes=10**15,
            available_ram_bytes=10**15,
        )
    assert temporary.exists()

    state.write_text(json.dumps({
        "config_fingerprint": config.fingerprint(),
        "source_sha256": "different-source",
    }))
    with pytest.raises(ValueError, match="different Flat source"):
        builder.build_ivfpq_index(
            config,
            resume=True,
            available_disk_bytes=10**15,
            available_ram_bytes=10**15,
        )

    state.write_text(json.dumps({
        "config_fingerprint": config.fingerprint(),
        "source_sha256": builder.sha256_file(source),
    }))
    status, result = _run_isolated_build(
        config,
        resume=True,
        available_disk_bytes=10**15,
        available_ram_bytes=10**15,
    )
    assert status == "ok"
    assert result["status"] == "built"


@pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu is unavailable")
def test_verification_failure_never_atomically_publishes_output(tmp_path):
    source = tmp_path / "source.index"
    output = tmp_path / "candidate.index"
    _write_flat(source)
    source_sha = builder.sha256_file(source)

    result = _run_isolated_build(
        _config(source, output),
        reject_verification=True,
        available_disk_bytes=10**15,
        available_ram_bytes=10**15,
    )
    assert result[:2] == ("error", "ValueError")
    assert "synthetic verification failure" in result[2]
    assert not output.exists()
    assert builder.sha256_file(source) == source_sha
    assert (tmp_path / "candidate.index.tmp").exists()


def test_resource_estimate_uses_discovered_metadata_and_safety_factors():
    config = builder.IVFPQBuildConfig(
        nlist=4,
        m=2,
        nbits=4,
        training_sample_size=64,
        training_block_size=16,
        add_chunk_size=32,
        disk_safety_factor=1.5,
        memory_safety_factor=1.25,
    )
    estimates = builder.estimate_build_resources(
        {"dimension": 8, "ntotal": 256, "size_bytes": 8192}, config
    )
    assert estimates["bytes_per_pq_code"] == 1
    assert estimates["required_free_disk_bytes"] >= int(
        estimates["candidate_final_index_upper_bound_bytes"] * 1.5
    )
    assert estimates["required_available_ram_bytes"] >= int(
        estimates["candidate_build_ram_estimate_bytes"] * 1.25
    )
