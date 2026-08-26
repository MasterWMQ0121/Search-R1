#!/usr/bin/env python3
"""Build a verified CPU IVF-PQ index from an existing reconstructable Flat index."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np


DEFAULT_SOURCE = "/workspace/searchr1-assets/wiki18/e5_Flat.index"
DEFAULT_OUTPUT = (
    "/workspace/searchr1-assets/wiki18/"
    "e5_IVFPQ_nlist4096_m96_nbits8.index"
)
MANIFEST_SCHEMA_VERSION = 1


def _require_faiss():
    try:
        return importlib.import_module("faiss")
    except ImportError as error:
        raise RuntimeError(
            "FAISS is required for IVF-PQ inspection/building; use the existing "
            "Retriever environment with faiss-cpu installed"
        ) from error


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-json")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def metric_name(metric_type: int) -> str:
    faiss = _require_faiss()
    names = {
        int(faiss.METRIC_INNER_PRODUCT): "inner_product",
        int(faiss.METRIC_L2): "l2",
    }
    return names.get(int(metric_type), f"faiss_metric_{int(metric_type)}")


@dataclass(frozen=True)
class IVFPQBuildConfig:
    source_index: str = DEFAULT_SOURCE
    output_index: str = DEFAULT_OUTPUT
    nlist: int = 4096
    m: int = 96
    nbits: int = 8
    training_sample_size: int = 262144
    training_block_size: int = 4096
    add_chunk_size: int = 32768
    seed: int = 42
    expected_metric: str = "inner_product"
    disk_safety_factor: float = 1.25
    memory_safety_factor: float = 1.20
    progress_interval_vectors: int = 1048576

    def normalized(self):
        values = asdict(self)
        values["source_index"] = str(Path(self.source_index).expanduser().resolve())
        values["output_index"] = str(Path(self.output_index).expanduser().resolve())
        return values

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.normalized(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def validate_config(config: IVFPQBuildConfig, dimension: Optional[int] = None) -> None:
    integer_fields = {
        "nlist": config.nlist,
        "m": config.m,
        "nbits": config.nbits,
        "training_sample_size": config.training_sample_size,
        "training_block_size": config.training_block_size,
        "add_chunk_size": config.add_chunk_size,
        "progress_interval_vectors": config.progress_interval_vectors,
    }
    for name, value in integer_fields.items():
        if isinstance(value, bool) or int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (
        ("disk_safety_factor", config.disk_safety_factor),
        ("memory_safety_factor", config.memory_safety_factor),
    ):
        if not math.isfinite(value) or value < 1.0:
            raise ValueError(f"{name} must be finite and at least 1.0")
    if config.expected_metric not in ("inner_product", "l2"):
        raise ValueError("expected_metric must be inner_product or l2")
    if config.training_sample_size < config.nlist:
        raise ValueError("training_sample_size must be at least nlist")
    if config.training_sample_size < (1 << config.nbits):
        raise ValueError("training_sample_size must cover all PQ codebook centroids")
    if dimension is not None and dimension % config.m != 0:
        raise ValueError(
            f"index dimension {dimension} must be divisible by PQ m={config.m}"
        )


def plan_training_ranges(
    ntotal: int,
    sample_size: int,
    block_size: int = 4096,
    seed: int = 42,
):
    """Select exact-size, disjoint contiguous blocks from broad index strata."""

    for name, value in (
        ("ntotal", ntotal),
        ("sample_size", sample_size),
        ("block_size", block_size),
    ):
        if isinstance(value, bool) or int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if sample_size > ntotal:
        raise ValueError("training sample size cannot exceed source ntotal")
    block_count = int(math.ceil(sample_size / block_size))
    rng = random.Random(seed)
    ranges = []
    for block_number in range(block_count):
        stratum_start = (block_number * ntotal) // block_count
        stratum_end = ((block_number + 1) * ntotal) // block_count
        sample_start_offset = (block_number * sample_size) // block_count
        sample_end_offset = ((block_number + 1) * sample_size) // block_count
        count = sample_end_offset - sample_start_offset
        if count <= 0 or count > block_size or count > stratum_end - stratum_start:
            raise RuntimeError("invalid deterministic sampling stratum allocation")
        latest_start = stratum_end - count
        start = rng.randint(stratum_start, latest_start)
        ranges.append({"start": start, "count": count, "end": start + count})
    if sum(item["count"] for item in ranges) != sample_size:
        raise RuntimeError("deterministic ranges do not cover the exact sample size")
    for left, right in zip(ranges, ranges[1:]):
        if left["end"] > right["start"]:
            raise RuntimeError("deterministic training ranges overlap")
    return ranges


def reconstruct_ranges(index, ranges):
    reconstructed = []
    dimension = int(index.d)
    for item in ranges:
        vectors = np.asarray(
            index.reconstruct_n(int(item["start"]), int(item["count"])),
            dtype=np.float32,
            order="C",
        )
        expected_shape = (int(item["count"]), dimension)
        if vectors.shape != expected_shape:
            raise RuntimeError(
                f"reconstruct_n returned {vectors.shape}, expected {expected_shape}"
            )
        if not np.isfinite(vectors).all():
            raise ValueError("source index reconstruction produced non-finite vectors")
        reconstructed.append(vectors)
    return np.ascontiguousarray(np.concatenate(reconstructed, axis=0), dtype=np.float32)


def _read_source_index(path: Path):
    faiss = _require_faiss()
    read_mode = "standard"
    try:
        flags = int(faiss.IO_FLAG_MMAP) | int(faiss.IO_FLAG_READ_ONLY)
        index = faiss.read_index(str(path), flags)
        read_mode = "mmap_read_only_requested"
    except Exception as mmap_error:
        available_ram, ram_source = _available_memory_bytes()
        guarded_requirement = int(math.ceil(path.stat().st_size * 1.25))
        if available_ram is None or available_ram < guarded_requirement:
            raise RuntimeError(
                "read-only mmap loading failed and a standard FAISS load could "
                "exceed available RAM; refusing before fallback: "
                f"source_size={path.stat().st_size}, "
                f"required_available={guarded_requirement}, "
                f"available={available_ram}, source={ram_source}"
            ) from mmap_error
        index = faiss.read_index(str(path))
        read_mode = "standard_read_fallback_after_memory_guard"
    return index, read_mode


def inspect_source_index(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"source Flat index does not exist or is empty: {path}")
    faiss = _require_faiss()
    index, read_mode = _read_source_index(path)
    class_name = type(index).__name__
    if not isinstance(index, faiss.IndexFlat):
        raise ValueError(
            f"source index must be a FAISS Flat index; found {class_name}"
        )
    if int(index.ntotal) <= 0 or int(index.d) <= 0:
        raise ValueError("source index must contain at least one nonempty vector")
    try:
        probe = np.asarray(index.reconstruct_n(0, 1), dtype=np.float32)
    except Exception as error:
        raise ValueError("source Flat index does not support vector reconstruction") from error
    if probe.shape != (1, int(index.d)):
        raise ValueError("source Flat index reconstruction returned an invalid shape")
    metadata = {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "faiss_class": class_name,
        "ntotal": int(index.ntotal),
        "dimension": int(index.d),
        "metric_type": int(index.metric_type),
        "metric": metric_name(index.metric_type),
        "read_mode": read_mode,
        "reconstruction_verified": True,
    }
    return index, metadata


def _available_memory_bytes():
    proc_meminfo = Path("/proc/meminfo")
    if proc_meminfo.is_file():
        for line in proc_meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024, "proc_meminfo_MemAvailable"
    if sys.platform == "darwin":
        try:
            completed = subprocess.run(
                ["vm_stat"], check=True, capture_output=True, text=True
            )
            lines = completed.stdout.splitlines()
            page_size = int(lines[0].split("page size of ", 1)[1].split(" bytes", 1)[0])
            counts = {}
            for line in lines[1:]:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                counts[key] = int(value.strip().rstrip("."))
            available_pages = sum(
                counts.get(key, 0)
                for key in ("Pages free", "Pages inactive", "Pages speculative")
            )
            return available_pages * page_size, "darwin_vm_stat_estimate"
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            pass
    return None, "unavailable"


def estimate_build_resources(metadata, config: IVFPQBuildConfig):
    dimension = int(metadata["dimension"])
    ntotal = int(metadata["ntotal"])
    bytes_per_code = int(math.ceil(config.m * config.nbits / 8.0))
    vector_codes_and_ids = ntotal * (bytes_per_code + 8)
    coarse_centroids = config.nlist * dimension * 4
    pq_centroids = (1 << config.nbits) * dimension * 4
    structural_allowance = 64 * 1024 * 1024
    raw_index_estimate = (
        vector_codes_and_ids + coarse_centroids + pq_centroids + structural_allowance
    )
    final_index_upper_bound = int(math.ceil(raw_index_estimate * 1.35))
    training_bytes = config.training_sample_size * dimension * 4
    chunk_bytes = config.add_chunk_size * dimension * 4
    source_resident_bytes = max(
        int(metadata["size_bytes"]), ntotal * dimension * 4
    )
    build_ram_estimate = (
        source_resident_bytes
        + final_index_upper_bound
        + 3 * training_bytes
        + 2 * chunk_bytes
        + 512 * 1024 * 1024
    )
    return {
        "method": "conservative_metadata_formula_v1",
        "bytes_per_pq_code": bytes_per_code,
        "source_resident_bytes": source_resident_bytes,
        "training_sample_bytes": training_bytes,
        "add_chunk_bytes": chunk_bytes,
        "candidate_final_index_upper_bound_bytes": final_index_upper_bound,
        "candidate_build_ram_estimate_bytes": build_ram_estimate,
        "required_free_disk_bytes": int(
            math.ceil(final_index_upper_bound * config.disk_safety_factor)
        ),
        "required_available_ram_bytes": int(
            math.ceil(build_ram_estimate * config.memory_safety_factor)
        ),
        "disk_safety_factor": config.disk_safety_factor,
        "memory_safety_factor": config.memory_safety_factor,
    }


def run_preflight(
    metadata,
    config: IVFPQBuildConfig,
    output_parent=None,
    available_disk_bytes=None,
    available_ram_bytes=None,
):
    output_parent = Path(
        output_parent or Path(config.output_index).expanduser().resolve().parent
    )
    output_parent.mkdir(parents=True, exist_ok=True)
    estimates = estimate_build_resources(metadata, config)
    if available_disk_bytes is None:
        available_disk_bytes = shutil.disk_usage(output_parent).free
        disk_source = "shutil.disk_usage"
    else:
        disk_source = "caller_override"
    if available_ram_bytes is None:
        available_ram_bytes, ram_source = _available_memory_bytes()
    else:
        ram_source = "caller_override"
    if available_ram_bytes is None:
        raise RuntimeError(
            "available RAM could not be determined; refusing to start the IVF-PQ build"
        )
    checks = {
        "disk": available_disk_bytes >= estimates["required_free_disk_bytes"],
        "memory": available_ram_bytes >= estimates["required_available_ram_bytes"],
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "available_free_disk_bytes": int(available_disk_bytes),
        "available_disk_source": disk_source,
        "available_ram_bytes": int(available_ram_bytes),
        "available_ram_source": ram_source,
        "source_index_size_bytes": int(metadata["size_bytes"]),
        "estimates": estimates,
    }
    if not report["passed"]:
        raise RuntimeError(f"IVF-PQ build preflight failed: {report}")
    return report


def _matching_quantizer(faiss, dimension: int, metric_type: int):
    if int(metric_type) == int(faiss.METRIC_INNER_PRODUCT):
        return faiss.IndexFlatIP(dimension)
    if int(metric_type) == int(faiss.METRIC_L2):
        return faiss.IndexFlatL2(dimension)
    raise ValueError(f"unsupported source metric for IVF-PQ: {metric_type}")


def verify_ivfpq_index(path, expected, query_vector=None):
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"IVF-PQ index does not exist or is empty: {path}")
    faiss = _require_faiss()
    index = faiss.read_index(str(path))
    if not isinstance(index, faiss.IndexIVFPQ):
        raise ValueError(f"output index is not IndexIVFPQ: {type(index).__name__}")
    checks = {
        "trained": bool(index.is_trained),
        "dimension_matches": int(index.d) == int(expected["dimension"]),
        "ntotal_matches": int(index.ntotal) == int(expected["ntotal"]),
        "metric_matches": int(index.metric_type) == int(expected["metric_type"]),
        "nlist_matches": int(index.nlist) == int(expected["nlist"]),
        "m_matches": int(index.pq.M) == int(expected["m"]),
        "nbits_matches": int(index.pq.nbits) == int(expected["nbits"]),
    }
    if not all(checks.values()):
        raise ValueError(f"IVF-PQ verification failed: {checks}")
    if query_vector is None:
        query_vector = np.zeros((1, int(index.d)), dtype=np.float32)
        query_vector[0, 0] = 1.0
    query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)
    if query_vector.shape != (1, int(index.d)) or not np.isfinite(query_vector).all():
        raise ValueError("verification query must be one finite vector of index dimension")
    index.nprobe = min(8, int(index.nlist))
    scores, ids = index.search(query_vector, min(3, int(index.ntotal)))
    nonempty_search = ids.shape[1] > 0 and int(ids[0, 0]) >= 0
    if not nonempty_search:
        raise ValueError("IVF-PQ verification search returned no valid result")
    checks["nonempty_search"] = True
    return {
        "checks": checks,
        "faiss_class": type(index).__name__,
        "dimension": int(index.d),
        "ntotal": int(index.ntotal),
        "metric_type": int(index.metric_type),
        "metric": metric_name(index.metric_type),
        "nlist": int(index.nlist),
        "m": int(index.pq.M),
        "nbits": int(index.pq.nbits),
        "sample_result_id": int(ids[0, 0]),
        "sample_result_score": float(scores[0, 0]),
    }


def _manifest_path(output: Path) -> Path:
    return output.with_suffix(".manifest.json")


def _state_path(output: Path) -> Path:
    return output.with_suffix(".build-state.json")


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _remove_restart_artifacts(paths) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def build_ivfpq_index(
    config: IVFPQBuildConfig,
    preflight_only=False,
    verify_only=False,
    resume=False,
    overwrite=False,
    available_disk_bytes=None,
    available_ram_bytes=None,
):
    """Build atomically; ``resume`` means validated restart, not add-stage resume."""

    validate_config(config)
    source = Path(config.source_index).expanduser().resolve()
    output = Path(config.output_index).expanduser().resolve()
    manifest_path = _manifest_path(output)
    state_path = _state_path(output)
    temporary = output.with_name(output.name + ".tmp")
    if source == output:
        raise ValueError("IVF-PQ output must not overwrite the Flat source index")

    source_index, source_metadata = inspect_source_index(source)
    validate_config(config, dimension=source_metadata["dimension"])
    if source_metadata["metric"] != config.expected_metric:
        raise ValueError(
            f"source metric is {source_metadata['metric']}, expected "
            f"{config.expected_metric}"
        )
    if config.training_sample_size > source_metadata["ntotal"]:
        raise ValueError("training_sample_size cannot exceed source ntotal")
    ranges = plan_training_ranges(
        source_metadata["ntotal"],
        config.training_sample_size,
        config.training_block_size,
        config.seed,
    )
    expected = {
        "dimension": source_metadata["dimension"],
        "ntotal": source_metadata["ntotal"],
        "metric_type": source_metadata["metric_type"],
        "nlist": config.nlist,
        "m": config.m,
        "nbits": config.nbits,
    }
    if verify_only:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "independent IVF-PQ verification requires the build manifest: "
                f"{manifest_path}"
            )
        manifest = _load_json(manifest_path)
        output_sha256 = sha256_file(output)
        if manifest.get("source", {}).get("sha256") != source_metadata["sha256"]:
            raise ValueError("IVF-PQ manifest is bound to a different Flat source")
        if manifest.get("config_fingerprint") != config.fingerprint():
            raise ValueError("IVF-PQ manifest is bound to a different build config")
        if manifest.get("output", {}).get("sha256") != output_sha256:
            raise ValueError("IVF-PQ output SHA256 does not match its build manifest")
        if manifest.get("source_sha256_unchanged") is not True:
            raise ValueError("IVF-PQ manifest does not attest an unchanged Flat source")
        verification = verify_ivfpq_index(output, expected)
        return {
            "status": "verified",
            "output_sha256": output_sha256,
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "source_sha256": source_metadata["sha256"],
            "config_fingerprint": config.fingerprint(),
            "verification": verification,
        }

    complete_exists = output.is_file() and manifest_path.is_file()
    if complete_exists and not overwrite:
        stale_paths = [path for path in (temporary, state_path) if path.exists()]
        if stale_paths:
            raise FileExistsError(
                "a complete IVF-PQ output exists alongside stale build artifacts; "
                "refusing to discard either implicitly"
            )
        manifest = _load_json(manifest_path)
        if manifest.get("source", {}).get("sha256") != source_metadata["sha256"]:
            raise ValueError("existing IVF-PQ manifest is bound to a different source")
        if manifest.get("config_fingerprint") != config.fingerprint():
            raise ValueError("existing IVF-PQ manifest is bound to a different config")
        if manifest.get("output", {}).get("sha256") != sha256_file(output):
            raise ValueError("existing IVF-PQ output SHA256 does not match its manifest")
        verification = verify_ivfpq_index(output, expected)
        return {"status": "existing_complete_verified", "manifest": manifest,
                "verification": verification}

    preflight = run_preflight(
        source_metadata,
        config,
        output_parent=output.parent,
        available_disk_bytes=available_disk_bytes,
        available_ram_bytes=available_ram_bytes,
    )
    if preflight_only:
        return {
            "status": "preflight_passed",
            "config": config.normalized(),
            "config_fingerprint": config.fingerprint(),
            "source": source_metadata,
            "training_sample_ranges": ranges,
            "preflight": preflight,
            "resume_semantics": "restart_from_zero_only",
        }
    print(
        json.dumps(
            {
                "phase6_ivfpq_build_preflight": preflight,
                "source": source_metadata,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )

    stale_paths = [path for path in (temporary, state_path) if path.exists()]
    partial_complete = output.exists() != manifest_path.exists()
    if (stale_paths or partial_complete or (complete_exists and overwrite)):
        if not (resume or overwrite):
            raise FileExistsError(
                "stale/incomplete IVF-PQ build artifacts exist; pass --resume for "
                "a validated restart from zero or --overwrite"
            )
        if resume and not overwrite:
            if not state_path.is_file():
                raise ValueError(
                    "--resume restart requires a build-state file so the source "
                    "and configuration can be validated"
                )
            try:
                stale_state = _load_json(state_path)
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError("stale build-state file is invalid") from error
            if stale_state.get("config_fingerprint") != config.fingerprint():
                raise ValueError(
                    "stale build state uses a different IVF-PQ configuration"
                )
            if stale_state.get("source_sha256") != source_metadata["sha256"]:
                raise ValueError("stale build state uses a different Flat source")
        _remove_restart_artifacts(
            [temporary, state_path, output, manifest_path]
        )

    started = time.perf_counter()
    state = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage": "sampling",
        "config_fingerprint": config.fingerprint(),
        "source_sha256": source_metadata["sha256"],
        "resume_semantics": "restart_from_zero_only",
        "vectors_added": 0,
    }
    _write_json_atomic(state_path, state)
    training_vectors = reconstruct_ranges(source_index, ranges)
    faiss = _require_faiss()
    quantizer = _matching_quantizer(
        faiss, source_metadata["dimension"], source_metadata["metric_type"]
    )
    candidate = faiss.IndexIVFPQ(
        quantizer,
        source_metadata["dimension"],
        config.nlist,
        config.m,
        config.nbits,
        source_metadata["metric_type"],
    )
    candidate.cp.seed = config.seed
    candidate.pq.cp.seed = config.seed
    state.update({"stage": "training", "training_vectors": len(training_vectors)})
    _write_json_atomic(state_path, state)
    candidate.train(training_vectors)
    if not candidate.is_trained:
        raise RuntimeError("FAISS IVF-PQ training did not produce a trained index")

    state["stage"] = "adding"
    last_progress = 0
    for start in range(0, source_metadata["ntotal"], config.add_chunk_size):
        count = min(config.add_chunk_size, source_metadata["ntotal"] - start)
        vectors = np.asarray(
            source_index.reconstruct_n(start, count), dtype=np.float32, order="C"
        )
        if vectors.shape != (count, source_metadata["dimension"]):
            raise RuntimeError("source chunk reconstruction returned an invalid shape")
        ids = np.arange(start, start + count, dtype=np.int64)
        candidate.add_with_ids(vectors, ids)
        if int(candidate.ntotal) != start + count:
            raise RuntimeError("IVF-PQ add stage did not preserve sequential ID count")
        if (
            start + count - last_progress >= config.progress_interval_vectors
            or start + count == source_metadata["ntotal"]
        ):
            state["vectors_added"] = start + count
            _write_json_atomic(state_path, state)
            last_progress = start + count

    state["stage"] = "writing_temporary"
    _write_json_atomic(state_path, state)
    faiss.write_index(candidate, str(temporary))
    _fsync_file(temporary)
    verification = verify_ivfpq_index(
        temporary, expected, query_vector=training_vectors[:1]
    )
    source_sha_final = sha256_file(source)
    if source_sha_final != source_metadata["sha256"]:
        raise RuntimeError("Flat source index changed during IVF-PQ construction")
    os.replace(temporary, output)
    _fsync_directory(output.parent)
    output_sha = sha256_file(output)
    finished = time.perf_counter()
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "config": config.normalized(),
        "config_fingerprint": config.fingerprint(),
        "source": source_metadata,
        "output": {
            "path": str(output),
            "sha256": output_sha,
            "size_bytes": output.stat().st_size,
        },
        "compression_ratio_source_over_output": (
            source.stat().st_size / output.stat().st_size
        ),
        "ntotal": source_metadata["ntotal"],
        "dimension": source_metadata["dimension"],
        "metric_type": source_metadata["metric_type"],
        "metric": source_metadata["metric"],
        "nlist": config.nlist,
        "m": config.m,
        "nbits": config.nbits,
        "training_sample_ranges": ranges,
        "training_sample_vector_count": sum(item["count"] for item in ranges),
        "add_chunk_size": config.add_chunk_size,
        "build_time_s": finished - started,
        "faiss_version": getattr(faiss, "__version__", "unknown"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "preflight": preflight,
        "verification": verification,
        "id_assignment": "explicit original row IDs added in sequential chunks",
        "sequential_original_id_assignment": True,
        "source_sha256_unchanged": source_sha_final == source_metadata["sha256"],
        "resume_semantics": "restart_from_zero_only",
    }
    _write_json_atomic(manifest_path, manifest)
    if state_path.exists():
        state_path.unlink()
        _fsync_directory(state_path.parent)
    return {"status": "built", "manifest": manifest}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", default=DEFAULT_SOURCE)
    parser.add_argument("--output-index", default=DEFAULT_OUTPUT)
    parser.add_argument("--nlist", type=int, default=4096)
    parser.add_argument("--m", type=int, default=96)
    parser.add_argument("--nbits", type=int, default=8)
    parser.add_argument("--training-sample-size", type=int, default=262144)
    parser.add_argument("--training-block-size", type=int, default=4096)
    parser.add_argument("--add-chunk-size", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-metric", choices=("inner_product", "l2"),
                        default="inner_product")
    parser.add_argument("--disk-safety-factor", type=float, default=1.25)
    parser.add_argument("--memory-safety-factor", type=float, default=1.20)
    parser.add_argument("--progress-interval-vectors", type=int, default=1048576)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.preflight_only and args.verify_only:
        raise ValueError("--preflight-only and --verify-only are mutually exclusive")
    config = IVFPQBuildConfig(
        source_index=args.source_index,
        output_index=args.output_index,
        nlist=args.nlist,
        m=args.m,
        nbits=args.nbits,
        training_sample_size=args.training_sample_size,
        training_block_size=args.training_block_size,
        add_chunk_size=args.add_chunk_size,
        seed=args.seed,
        expected_metric=args.expected_metric,
        disk_safety_factor=args.disk_safety_factor,
        memory_safety_factor=args.memory_safety_factor,
        progress_interval_vectors=args.progress_interval_vectors,
    )
    result = build_ivfpq_index(
        config,
        preflight_only=args.preflight_only,
        verify_only=args.verify_only,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
