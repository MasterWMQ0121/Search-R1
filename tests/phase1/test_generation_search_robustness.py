from types import SimpleNamespace

import pytest

import search_r1.llm_agent.generation as generation
from search_r1.llm_agent.generation import LLMGenerationManager


INVALID_OBSERVATION_FRAGMENT = "My previous action is invalid."


@pytest.fixture
def manager():
    instance = LLMGenerationManager.__new__(LLMGenerationManager)
    instance.config = SimpleNamespace(
        search_url="http://127.0.0.1:8000/retrieve",
        topk=2,
    )
    return instance


@pytest.mark.parametrize("prediction", ["<search></search>", "<search>   </search>"])
def test_blank_search_is_invalid_and_does_not_call_retriever(manager, prediction):
    def unexpected_search(_queries):
        raise AssertionError("blank search must not call the retriever")

    manager.batch_search = unexpected_search
    next_obs, dones, valid_action, is_search = manager.execute_predictions(
        [prediction], pad_token="<pad>", active_mask=[True]
    )

    assert INVALID_OBSERVATION_FRAGMENT in next_obs[0]
    assert dones == [0]
    assert valid_action == [0]
    assert is_search == [0]


def test_valid_search_is_stripped_and_preserves_search_behavior(manager):
    received_queries = []

    def fake_search(queries):
        received_queries.extend(queries)
        return ["retrieved passage"]

    manager.batch_search = fake_search
    next_obs, dones, valid_action, is_search = manager.execute_predictions(
        ["<search>  capital of France  </search>"],
        pad_token="<pad>",
        active_mask=[True],
    )

    assert received_queries == ["capital of France"]
    assert next_obs == ["\n\n<information>retrieved passage</information>\n\n"]
    assert dones == [0]
    assert valid_action == [1]
    assert is_search == [1]


class FakeResponse:
    def __init__(self, status_code, text, json_payload):
        self.status_code = status_code
        self.text = text
        self._json_payload = json_payload

    def json(self):
        return self._json_payload


def test_valid_nonempty_search_preserves_retriever_contract(manager, monkeypatch):
    captured = {}

    def fake_post(url, json):
        captured["url"] = url
        captured["payload"] = json
        return FakeResponse(200, '{"result":[...]}', {
            "result": [[{
                "document": {"contents": '"Paris"\nParis is the capital of France.'},
                "score": 0.9,
            }]],
        })

    monkeypatch.setattr(generation.requests, "post", fake_post)
    results = manager.batch_search(["  capital of France  "])

    assert captured == {
        "url": "http://127.0.0.1:8000/retrieve",
        "payload": {
            "queries": ["capital of France"],
            "topk": 2,
            "return_scores": True,
        },
    }
    assert results == ['Doc 1(Title: "Paris") Paris is the capital of France.\n']


def test_retriever_4xx_has_contextual_error(manager, monkeypatch):
    long_body = "blank query rejected: " + "x" * 600
    monkeypatch.setattr(
        generation.requests,
        "post",
        lambda *_args, **_kwargs: FakeResponse(400, long_body, {"detail": "bad query"}),
    )

    with pytest.raises(RuntimeError) as exc_info:
        manager._batch_search(["bad query"])

    message = str(exc_info.value)
    assert "status=400" in message
    assert "queries=['bad query']" in message
    assert "response_body='blank query rejected:" in message
    assert len(message) < len(long_body)


def test_successful_retriever_json_missing_result_has_contextual_error(manager, monkeypatch):
    monkeypatch.setattr(
        generation.requests,
        "post",
        lambda *_args, **_kwargs: FakeResponse(200, '{"detail":"unexpected"}', {"detail": "unexpected"}),
    )

    with pytest.raises(RuntimeError, match="missing 'result'") as exc_info:
        manager._batch_search(["valid query"])

    message = str(exc_info.value)
    assert "status=200" in message
    assert "queries=['valid query']" in message
    assert "response_body=" in message
