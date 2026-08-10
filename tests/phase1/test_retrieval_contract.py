from types import SimpleNamespace

from fastapi.testclient import TestClient

import search_r1.search.retrieval_server as server


class FakeRetriever:
    def batch_search(self, query_list, num, return_score):
        documents = [[
            {"id": f"doc:{query}:0", "contents": "first"},
            {"id": f"doc:{query}:1", "contents": "second"},
        ][:num] for query in query_list]
        scores = [[0.9, 0.7][:num] for _ in query_list]
        return documents, scores


def client(monkeypatch):
    monkeypatch.setattr(server, "config", SimpleNamespace(retrieval_topk=2))
    monkeypatch.setattr(server, "retriever", FakeRetriever())
    return TestClient(server.app)


def test_retrieval_response_contract(monkeypatch):
    response = client(monkeypatch).post("/retrieve", json={
        "queries": ["known query"], "topk": 2, "return_scores": True
    })
    assert response.status_code == 200
    hits = response.json()["result"][0]
    assert [hit["document"]["id"] for hit in hits] == ["doc:known query:0", "doc:known query:1"]
    assert [hit["score"] for hit in hits] == [0.9, 0.7]


def test_default_topk_and_scoreless_contract(monkeypatch):
    response = client(monkeypatch).post("/retrieve", json={"queries": ["q"]})
    assert response.status_code == 200
    assert response.json()["result"][0][0]["id"] == "doc:q:0"


def test_malformed_requests_are_rejected(monkeypatch):
    test_client = client(monkeypatch)
    assert test_client.post("/retrieve", json={"queries": []}).status_code == 400
    assert test_client.post("/retrieve", json={"queries": [" "]}).status_code == 400
    assert test_client.post("/retrieve", json={"queries": ["q"], "topk": 0}).status_code == 400
    assert test_client.post("/retrieve", json={"topk": 2}).status_code == 422
