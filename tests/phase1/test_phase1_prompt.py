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
    assert "so search and do not answer" in prompt


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


def test_phase1_prompt_requires_tagged_answer_after_information():
    prompt = make_prompt(QUESTION)
    normalized = " ".join(prompt.split())

    assert "If the retrieved evidence contains the answer" in prompt
    assert "your NEXT assistant response MUST consist of exactly two tagged blocks and nothing else" in normalized
    assert "<think>The retrieved fixture contains explicit accepted answer evidence.</think>" in prompt
    assert "<answer>ANSWER_TEXT</answer>" in prompt
    assert "Once an information block appears, follow AFTER RECEIVING INFORMATION instead" in normalized


def test_phase1_fixture_answer_prefix_is_a_terminal_signal():
    prompt = make_prompt(QUESTION)
    normalized = " ".join(prompt.split())

    assert 'literal text "Accepted answer evidence:"' in prompt
    assert "You MUST NOT search again" in prompt
    assert "Copy the answer text immediately following that prefix" in prompt
    assert "retrieval is sufficient regardless of whether you think more information would be useful" in normalized
    assert "Replace ANSWER_TEXT with the copied evidence and stop immediately after </answer>" in normalized
    assert "only to this deliberately answer-leaky Phase-1 fixture" in prompt


def test_phase1_prompt_forbids_plain_prose_answers():
    prompt = make_prompt(QUESTION)

    assert '"the answer is ..."' in prompt
    assert '"to answer: ..."' in prompt
    assert "any other text outside protocol tags" in prompt


def test_phase1_prompt_is_explicitly_not_model_quality_evidence():
    assert "validates plumbing, not model quality" in make_prompt(QUESTION)
