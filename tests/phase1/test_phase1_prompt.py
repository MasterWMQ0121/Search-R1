import re

from scripts.data_process.phase1_smoke_nq import make_prompt


QUESTION = "Who wrote The Hobbit?"


def test_phase1_prompt_requires_first_response_to_search():
    prompt = make_prompt(QUESTION)
    normalized = " ".join(prompt.split())

    assert "FIRST assistant response" in prompt
    assert "must contain a brief reason inside <think>...</think>" in normalized
    assert "end in exactly one non-empty <search>...</search>" in normalized
    assert "Stop immediately after </search>" in prompt


def test_phase1_prompt_requires_nonempty_search_content():
    prompt = make_prompt(QUESTION)

    assert "at least one non-whitespace character" in prompt
    assert f"<search>{QUESTION}</search>" in prompt


def test_phase1_prompt_forbids_answer_before_information():
    prompt = make_prompt(QUESTION)
    normalized = " ".join(prompt.split())

    assert "Do not answer the question" in prompt
    assert "do not produce any final-answer action before receiving an <information>...</information> block" in normalized
    assert "must search and must not answer" in prompt


def test_phase1_prompt_contains_search_r1_protocol_tags_and_reward_example():
    prompt = make_prompt(QUESTION)

    for tag in (
        "<think>", "</think>",
        "<search>", "</search>",
        "<information>", "</information>",
        "<answer>", "</answer>",
    ):
        assert tag in prompt

    # QA EM expects the prompt's single format example plus the generated answer.
    assert len(re.findall(r"<answer>.*?</answer>", prompt, re.DOTALL)) == 1


def test_phase1_prompt_is_explicitly_not_model_quality_evidence():
    assert "validates plumbing, not model quality" in make_prompt(QUESTION)
