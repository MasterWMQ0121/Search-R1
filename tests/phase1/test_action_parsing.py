import pytest

from search_r1.llm_agent.generation import LLMGenerationManager


@pytest.fixture
def manager():
    return LLMGenerationManager.__new__(LLMGenerationManager)


def test_parses_search_and_answer_actions(manager):
    actions, contents = manager.postprocess_predictions([
        "<think>need facts</think><search> capital of France </search>",
        "<think>done</think><answer> Paris </answer>",
    ])
    assert actions == ["search", "answer"]
    assert contents == ["capital of France", "Paris"]


def test_first_well_formed_action_wins(manager):
    actions, contents = manager.postprocess_predictions([
        "prefix <search>first query</search> then <answer>later</answer>"
    ])
    assert actions == ["search"]
    assert contents == ["first query"]


@pytest.mark.parametrize("prediction", ["plain text", "<search>missing close", "<tool>x</tool>"])
def test_invalid_action_is_reported_without_content(manager, prediction):
    actions, contents = manager.postprocess_predictions([prediction])
    assert actions == [None]
    assert contents == [""]


def test_non_string_prediction_is_rejected(manager):
    with pytest.raises(ValueError):
        manager.postprocess_predictions([123])
