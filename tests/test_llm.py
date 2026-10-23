import json

import httpx
import pytest
from groq import Groq

import cv_assistant.llm as llm
from cv_assistant.llm import AnalysisError, GroqInterviewer


def install_transport(monkeypatch, handler):
    def client(**kwargs):
        return Groq(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)))

    monkeypatch.setattr(llm, "Groq", client)


def completion(content, finish_reason="stop"):
    return {
        "id": "test", "object": "chat.completion", "created": 0,
        "model": "openai/gpt-oss-120b",
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": content}}],
    }


def test_groq_request_and_structured_response(monkeypatch):
    def handler(request):
        body = json.loads(request.content)
        assert str(request.url) == "https://api.groq.com/openai/v1/chat/completions"
        assert body["model"] == "openai/gpt-oss-120b"
        assert body["response_format"]["json_schema"]["strict"] is True
        assert body["reasoning_effort"] == "low"
        assert body["max_completion_tokens"] == 3000
        schema = body["response_format"]["json_schema"]["schema"]
        for obj in [schema, *schema["$defs"].values()]:
            assert obj["additionalProperties"] is False
            assert set(obj["required"]) == set(obj["properties"])
        payload = json.loads(body["messages"][1]["content"])
        assert payload["turns"][0]["answer"] == "Учусь Python"
        answer = {"facts": [{"topic": "skills", "text": "Изучает Python", "evidence": "Учусь Python"}],
                  "gaps": [{"topic": "experience", "key": "overview", "question": "Есть ли опыт работы?"}]}
        return httpx.Response(200, json=completion(json.dumps(answer)))

    install_transport(monkeypatch, handler)
    result = GroqInterviewer("test-key").analyze(
        "Developer", [{"topic": "about", "question": "О себе?", "answer": "Учусь Python"}]
    )
    assert result.facts[0].topic == "skills"


@pytest.mark.parametrize("status, expected", [(429, "лимит"), (401, "GROQ_API_KEY"), (500, "отклонил")])
def test_api_errors_are_safe_and_not_retried(monkeypatch, status, expected):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, json={"error": {"message": "private data must not be shown"}})

    install_transport(monkeypatch, handler)
    with pytest.raises(AnalysisError) as error:
        GroqInterviewer("test-key").analyze("Developer", [])
    assert expected in str(error.value)
    assert "private data" not in str(error.value)
    assert len(requests) == 1


@pytest.mark.parametrize("content, reason", [("not JSON", "stop"), (None, "stop"), ("{}", "length")])
def test_bad_output_is_rejected(monkeypatch, content, reason):
    install_transport(monkeypatch, lambda request: httpx.Response(200, json=completion(content, reason)))
    with pytest.raises(AnalysisError):
        GroqInterviewer("test-key").analyze("Developer", [])


def test_timeout_is_safe(monkeypatch):
    def handler(request):
        raise httpx.ReadTimeout("private data", request=request)

    install_transport(monkeypatch, handler)
    with pytest.raises(AnalysisError, match="недоступен"):
        GroqInterviewer("test-key").analyze("Developer", [])


def test_reviewer_requests_authored_strengths_with_strict_source_schema(monkeypatch):
    from cv_assistant.reviewer import GroqReviewer

    def handler(request):
        body = json.loads(request.content)
        schema = body["response_format"]["json_schema"]["schema"]
        assert "strengths" in schema["properties"]
        assert "highlight_fact_ids" not in schema["properties"]
        for obj in [schema, *schema["$defs"].values()]:
            assert obj["additionalProperties"] is False
            assert set(obj["required"]) == set(obj["properties"])
        assert "source_fact_ids" in body["messages"][0]["content"]
        return httpx.Response(200, json=completion(json.dumps({
            "strengths": [{"text": "Изучает язык Python.", "source_fact_ids": [0]}],
            "checks": [], "suggestions": []})))

    install_transport(monkeypatch, handler)
    result = GroqReviewer("test-key").review(
        "Developer", [{"topic": "skills", "text": "учу Python", "evidence": "учу Python"}], [])
    assert result.strengths[0].text == "Изучает язык Python."
