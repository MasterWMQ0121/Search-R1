import pytest

from verl.trainer.main_ppo import _select_rm_score_fn
from verl.utils.reward_score import qa_em


PROMPT_EXAMPLE = "For example, <answer> Beijing </answer>. Question: capital of France?"
GROUND_TRUTH = {"target": ["Paris"]}


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
