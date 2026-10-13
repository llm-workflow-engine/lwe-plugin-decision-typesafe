"""Exercise the real LangChain wrapper against a controlled HTTP response."""

from __future__ import annotations

import httpx2
import pytest

from lwe.decisions import (
    Choice,
    DecisionRequest,
    DecisionService,
    DecisionSettings,
    Noul,
    Score,
)
from lwe_plugin_decision_typesafe import TypeSafeDecisionProvider


def test_typesafe_question_translation_and_response_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the actual pinned classifier to catch changes in its typed interface."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret")
    payloads: list[object] = []

    def post(client: httpx2.Client, url: str, **kwargs: object) -> httpx2.Response:
        assert url == "https://api.typesafe.ai/v1/systemone"
        payloads.append(kwargs["json"])
        return httpx2.Response(
            200,
            request=httpx2.Request("POST", url),
            json={
                "model": "jev-test",
                "usage": {"input_tokens": 42, "output_tokens": 0},
                "answers": {
                    "urgent": {"type": "noul", "noul": 0},
                    "team": {
                        "type": "choice",
                        "choice": "support",
                        "probabilities": {"support": 1},
                        "confidence": 1,
                    },
                    "severity": {
                        "type": "score",
                        "score": 0.25,
                        "probabilities": {"0": 0.75, "1": 0.25},
                        "legend": {"0": "low", "1": "high"},
                        "confidence": 0.3,
                    },
                },
            },
        )

    monkeypatch.setattr(httpx2.Client, "post", post)
    request = DecisionRequest(
        state={"record": 42},
        questions={
            "urgent": Noul(instructions="Urgent?"),
            "team": Choice(instructions="Which team?", criteria={"support": "Help"}),
            "severity": Score(instructions="Severity?", criteria=["low", "high"]),
        },
    )
    service = DecisionService(
        DecisionSettings(provider="typesafe", model="jev-test"),
        TypeSafeDecisionProvider(),
    )
    result = service.decide(request).model_dump(mode="json")
    assert result["answers"]["severity"]["score"] == 0.25
    assert result["answers"]["urgent"]["noul"] == 0
    assert result["usage"]["input_tokens"] == 42
    assert payloads == [
        {
            "state": {"record": 42},
            "model": "jev-test",
            "questions": request.model_dump(exclude_none=True)["questions"],
        }
    ]
    with pytest.raises(ValueError, match="do not support media"):
        service.decide(
            DecisionRequest.model_validate(
                {**request.model_dump(), "images": ["data:image/png;base64,eA=="]}
            )
        )
    assert len(payloads) == 1
