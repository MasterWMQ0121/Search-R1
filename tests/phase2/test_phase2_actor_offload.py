from pathlib import Path


SOURCE = Path("verl/workers/fsdp_workers.py").read_text()


def test_actor_params_are_offloaded_before_rollout_build():
    offload_call = (
        "offload_fsdp_param_and_grad(\n"
        "                    module=self.actor_module_fsdp,"
    )
    rollout_build = (
        "self.rollout, self.rollout_sharding_manager = self._build_rollout()"
    )

    assert offload_call in SOURCE
    assert rollout_build in SOURCE

    assert SOURCE.index(offload_call) < SOURCE.index(rollout_build)


def test_init_does_not_use_grad_only_for_param_offload():
    old_bug = (
        "if self._is_offload_param:\n"
        "                # param is require during state_dict in sharding manager\n"
        "                offload_fsdp_grad(module=self.actor_module_fsdp)"
    )

    assert old_bug not in SOURCE
