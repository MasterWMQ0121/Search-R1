import re

from scripts.data_process.phase1_smoke_nq import make_prompt


QUESTION = "Who wrote The Hobbit?"


def test_phase1_prompt_state_a_requires_search_action():
    prompt = make_prompt(QUESTION)
    normalized = " ".join(prompt.split())

    assert "STATE A — BEFORE INFORMATION EXISTS" in prompt
    assert "current action must be exactly two tagged blocks" in normalized
    assert "followed by exactly one non-empty search" in normalized
    assert "End the current action immediately after </search>" in prompt
    assert "Do not answer in State A" in prompt


def test_phase1_prompt_requires_nonempty_search_content():
    prompt = make_prompt(QUESTION)

    assert "at least one non-whitespace character" in prompt
    assert f"<search>{QUESTION}</search>" in prompt


def test_phase1_prompt_forbids_answer_before_information():
    prompt = make_prompt(QUESTION)

    assert "STATE A — BEFORE INFORMATION EXISTS" in prompt
    assert "Do not answer in State A" in prompt
    assert prompt.index("STATE A — BEFORE INFORMATION EXISTS") < prompt.index("STATE B — AFTER INFORMATION EXISTS")


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

    assert "STATE B — AFTER INFORMATION EXISTS" in prompt
    assert "continue with exactly one answer block and nothing else" in normalized
    assert "Do not add a think block or prose" in prompt
    assert "another <search> action is INVALID" in prompt
    assert "Stop immediately after </answer> and terminate" in prompt


def test_phase1_fixture_selects_only_the_matching_question_document():
    prompt = make_prompt(QUESTION)
    normalized = " ".join(prompt.split())

    assert "whose Question: text is exactly equal to the original question" in normalized
    assert "Ignore every other retrieved document" in prompt
    assert 'even if it also contains "Accepted answer evidence:"' in prompt
    assert "From ONLY the matching document" in prompt
    assert "copy only the text immediately following \"Accepted answer evidence:\"" in prompt
    assert "Once that matching answer-bearing document is found" in normalized
    assert "You MUST NOT search again" in prompt


def test_phase1_fixture_uses_concrete_example_without_symbolic_placeholder():
    prompt = make_prompt(QUESTION)

    assert "Original question: what color is the daytime sky?" in prompt
    assert "Generated:\n<think>I need evidence.</think>\n<search>daytime sky color</search>" in prompt
    assert "Environment inserts into the same trajectory:\n<information>" in prompt
    assert "Question: what color is the daytime sky?" in prompt
    assert "Accepted answer evidence: blue" in prompt
    assert "Generation continues in the same trajectory:\n<answer>blue</answer>" in prompt
    assert "<answer>blue</answer>" in prompt
    assert "ANSWER_TEXT" not in prompt
    assert "only to this deliberately answer-leaky Phase-1 fixture" in prompt


def test_phase1_prompt_forbids_plain_prose_answers():
    prompt = make_prompt(QUESTION)

    assert "exactly one answer block and nothing else" in prompt
    assert "Do not add a think block or prose" in prompt


def test_phase1_prompt_describes_one_evolving_trajectory_without_chat_response_terms():
    prompt = make_prompt(QUESTION)
    lower_prompt = prompt.lower()

    assert "ONE evolving assistant trajectory" in prompt
    assert "SAME evolving assistant trajectory" in prompt
    assert "not a new user message or a new chat turn" in prompt
    assert "Generation then continues in State B, not State A" in prompt
    for forbidden in (
        "first assistant response",
        "next assistant response",
        "previous response",
        "next response",
    ):
        assert forbidden not in lower_prompt


def test_phase1_prompt_is_explicitly_not_model_quality_evidence():
    assert "validates plumbing, not model quality" in make_prompt(QUESTION)
