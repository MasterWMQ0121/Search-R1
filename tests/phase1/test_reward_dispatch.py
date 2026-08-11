import subprocess
import sys

import pytest

from verl.utils.reward_score import qa_em
from verl.utils.reward_score.dispatch import _select_rm_score_fn


PROMPT_EXAMPLE = "For example, <answer> Beijing </answer>. Question: capital of France?"
GROUND_TRUTH = {"target": ["Paris"]}


def test_reward_dispatch_import_has_no_training_dependencies():
    code = """
import sys
from verl.utils.reward_score.dispatch import _select_rm_score_fn
forbidden = ('ray', 'codetiming', 'vllm', 'torch')
loaded = [name for name in sys.modules if name.split('.')[0] in forbidden or name.startswith('verl.trainer')]
if loaded:
    raise SystemExit(f'unexpected heavyweight imports: {loaded}')
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_phase1_data_source_dispatches_to_qa_exact_match():
    assert _select_rm_score_fn("nq_phase1_smoke") is qa_em.compute_score_em


def test_phase1_dispatch_scores_correct_and_incorrect_answers():
    scorer = _select_rm_score_fn("nq_phase1_smoke")

    correct = scorer(
        solution_str=PROMPT_EXAMPLE + " <answer>Paris</answer>",
        ground_truth=GROUND_TRUTH,
    )
    incorrect = scorer(
        solution_str=PROMPT_EXAMPLE + " <answer>London</answer>",
        ground_truth=GROUND_TRUTH,
    )

    assert correct == 1.0
    assert incorrect == 0.0


@pytest.mark.parametrize(
    "data_source",
    ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"],
)
def test_existing_qa_dispatch_is_unchanged(data_source):
    assert _select_rm_score_fn(data_source) is qa_em.compute_score_em


def test_unknown_data_source_remains_unsupported():
    with pytest.raises(NotImplementedError):
        _select_rm_score_fn("unknown_phase1_source")
