import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FSDP_UTILS_PATH = ROOT / "verl" / "utils" / "fsdp_utils.py"
WORKER_PATH = ROOT / "verl" / "workers" / "fsdp_workers.py"
VLLM_ROLLOUT_PATH = ROOT / "verl" / "workers" / "rollout" / "vllm_rollout" / "vllm_rollout.py"
RAY_TRAINER_PATH = ROOT / "verl" / "trainer" / "ppo" / "ray_trainer.py"

FSDP_UTILS_SOURCE = FSDP_UTILS_PATH.read_text(encoding="utf-8")
WORKER_SOURCE = WORKER_PATH.read_text(encoding="utf-8")
VLLM_ROLLOUT_SOURCE = VLLM_ROLLOUT_PATH.read_text(encoding="utf-8")
RAY_TRAINER_SOURCE = RAY_TRAINER_PATH.read_text(encoding="utf-8")


def _load_fsdp_utils_functions(*names, globals_dict):
    tree = ast.parse(FSDP_UTILS_SOURCE)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == set(names)

    namespace = dict(globals_dict)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(FSDP_UTILS_PATH), "exec"), namespace)
    return [namespace[name] for name in names]


class _FakeDevice:
    def __init__(self, device_type, index=None):
        self.type = device_type
        self.index = index

    def __str__(self):
        return self.type if self.index is None else f"{self.type}:{self.index}"


class _FakeStorage:
    _next_pointer = 1

    def __init__(self, device, nbytes=4096):
        self.device = device
        self._nbytes = nbytes
        self.pointer = _FakeStorage._next_pointer
        _FakeStorage._next_pointer += 1

    def nbytes(self):
        return self._nbytes


class _FakeTensor:
    def __init__(self, storage, shape=(1024,)):
        self.storage = storage
        self.shape = shape

    @property
    def device(self):
        return self.storage.device

    def data_ptr(self):
        return self.storage.pointer

    def size(self):
        return self.shape

    def untyped_storage(self):
        return self.storage


class _FakeFlatParameter:
    def __init__(self, device):
        self._storage = _FakeStorage(device)
        self._local_shard = self.data
        self.grad = None

    @property
    def data(self):
        return _FakeTensor(self._storage)

    @data.setter
    def data(self, tensor):
        self._storage = tensor.storage

    def to(self, device, non_blocking=False):
        del non_blocking
        return _FakeTensor(_FakeStorage(device, self._storage.nbytes()))


class _FakeHandle:
    def __init__(self, flat_param, native_offload=False):
        self.flat_param = flat_param
        self._offload_params = native_offload
        self.moves = []

    def flat_param_to(self, device, non_blocking=False):
        self.moves.append((str(device), non_blocking))
        self.flat_param.data = self.flat_param.to(device, non_blocking=non_blocking)


class _FakeFSDP:
    def __init__(self, handles):
        self._all_handles = handles
        self._is_root = True
        self.lazy_init_calls = 0

    def named_parameters(self):
        return [(f"flat_{index}", handle.flat_param) for index, handle in enumerate(self._all_handles)]

    def named_buffers(self):
        return []


class _FakeNoGrad:
    def __call__(self, function=None):
        return self if function is None else function


class _FakeCuda:
    def __init__(self):
        self.empty_cache_calls = 0

    def empty_cache(self):
        self.empty_cache_calls += 1


class _FakeTorch:
    def __init__(self):
        self.no_grad = _FakeNoGrad()
        self.cuda = _FakeCuda()

    @staticmethod
    def device(device_type, index=None):
        return _FakeDevice(device_type, index)


def _fake_lazy_init(model, root_model):
    assert model is root_model
    model.lazy_init_calls += 1


def test_fsdp_model_offload_moves_handle_owned_flat_parameter_storage():
    fake_torch = _FakeTorch()
    offload_model, load_model = _load_fsdp_utils_functions(
        "offload_fsdp_model_to_cpu",
        "load_fsdp_model_to_gpu",
        globals_dict={"torch": fake_torch, "FSDP": _FakeFSDP, "_lazy_init": _fake_lazy_init},
    )

    flat_param = _FakeFlatParameter(_FakeDevice("cuda", 0))
    handle = _FakeHandle(flat_param)
    model = _FakeFSDP([handle])

    offload_model(model)

    assert model.lazy_init_calls == 1
    assert handle.moves == [("cpu", True)]
    assert str(flat_param.data.device) == "cpu"
    assert str(flat_param._local_shard.device) == "cpu"
    assert flat_param.data.data_ptr() == flat_param._local_shard.data_ptr()

    load_model(model, device_id=0)

    assert model.lazy_init_calls == 2
    assert handle.moves[-1] == ("cuda:0", True)
    assert str(flat_param.data.device) == "cuda:0"
    assert str(flat_param._local_shard.device) == "cuda:0"
    assert flat_param.data.data_ptr() == flat_param._local_shard.data_ptr()
    assert fake_torch.cuda.empty_cache_calls == 1


def test_fsdp_model_offload_does_not_override_native_cpu_offload_handles():
    fake_torch = _FakeTorch()
    (offload_model,) = _load_fsdp_utils_functions(
        "offload_fsdp_model_to_cpu",
        globals_dict={"torch": fake_torch, "FSDP": _FakeFSDP, "_lazy_init": _fake_lazy_init},
    )
    handle = _FakeHandle(_FakeFlatParameter(_FakeDevice("cpu")), native_offload=True)

    offload_model(_FakeFSDP([handle]))

    assert handle.moves == []


def test_fsdp_storage_summary_reports_canonical_alias_devices():
    fake_torch = _FakeTorch()
    offload_model, storage_nbytes, get_summary, format_summary = _load_fsdp_utils_functions(
        "offload_fsdp_model_to_cpu",
        "_tensor_storage_nbytes",
        "get_fsdp_model_device_summary",
        "format_fsdp_model_device_summary",
        globals_dict={"torch": fake_torch, "FSDP": _FakeFSDP, "_lazy_init": _fake_lazy_init},
    )
    del storage_nbytes
    model = _FakeFSDP([_FakeHandle(_FakeFlatParameter(_FakeDevice("cuda", 0)))])

    offload_model(model)
    summary = get_summary(model)
    formatted = format_summary(model)

    assert summary["parameter_data"]["cpu"]["bytes"] == 4096
    assert summary["flat_param_data"]["cpu"]["bytes"] == 4096
    assert summary["flat_param_local_shard"]["cpu"]["bytes"] == 4096
    assert "cuda" not in formatted
    assert "parameter_data[cpu=" in formatted
    assert "flat_param_local_shard[cpu=" in formatted


def test_existing_param_and_grad_entrypoints_delegate_to_fsdp_handle_helpers():
    tree = ast.parse(FSDP_UTILS_SOURCE)
    calls_by_function = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        calls_by_function[node.name] = {
            call.func.id for call in ast.walk(node) if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        }

    assert "offload_fsdp_model_to_cpu" in calls_by_function["offload_fsdp_param_and_grad"]
    assert "load_fsdp_model_to_gpu" in calls_by_function["load_fsdp_param_and_grad"]


def test_actor_params_are_offloaded_before_rollout_build():
    tree = ast.parse(WORKER_SOURCE)
    init_model = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "init_model"
    )
    calls = [node for node in ast.walk(init_model) if isinstance(node, ast.Call)]

    offload_line = min(
        node.lineno
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "offload_fsdp_param_and_grad"
    )
    rollout_line = min(
        node.lineno
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "_build_rollout"
    )

    assert offload_line < rollout_line


def test_colocated_reference_is_initialized_before_actor_rollout():
    assert "create_colocated_worker_cls(class_dict=class_dict)" in RAY_TRAINER_SOURCE
    assert RAY_TRAINER_SOURCE.index("self.ref_policy_wg.init_model()") < RAY_TRAINER_SOURCE.index(
        "self.actor_rollout_wg.init_model()"
    )


def test_init_and_vllm_construction_have_visible_memory_boundaries():
    for expected in (
        "Before offload reference params during init",
        "After offload reference params during init",
        "Reference FSDP storage after init offload",
        "Before offload actor params during init",
        "After offload actor params during init",
        "Actor FSDP storage immediately before rollout build",
    ):
        assert expected in WORKER_SOURCE

    for expected in (
        "Before vLLM LLM construction",
        "After vLLM LLM construction before weight offload",
        "After vLLM model weight offload",
    ):
        assert expected in VLLM_ROLLOUT_SOURCE
