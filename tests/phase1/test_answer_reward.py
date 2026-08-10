from verl.utils.reward_score.qa_em import (
    compute_score_em,
    em_check,
    extract_solution,
    normalize_answer,
)


PROMPT_EXAMPLE = "For example, <answer> Beijing </answer>. Question: capital of France?"


def test_extracts_last_answer_after_prompt_example():
    solution = PROMPT_EXAMPLE + " <think>known</think><answer> Paris </answer>"
    assert extract_solution(solution) == "Paris"


def test_prompt_example_alone_is_not_a_generated_answer():
    assert extract_solution(PROMPT_EXAMPLE) is None
    assert compute_score_em(PROMPT_EXAMPLE, {"target": ["Beijing"]}) == 0


def test_exact_match_normalization_and_reward():
    assert normalize_answer("The, PARIS!") == "paris"
    assert em_check("The, PARIS!", ["Paris"]) == 1
    solution = PROMPT_EXAMPLE + " <answer>The, PARIS!</answer>"
    assert compute_score_em(solution, {"target": ["Paris", "City of Paris"]}) == 1.0


def test_wrong_but_well_formed_answer_receives_format_score():
    solution = PROMPT_EXAMPLE + " <answer>London</answer>"
    assert compute_score_em(solution, {"target": ["Paris"]}, format_score=0.25) == 0.25
