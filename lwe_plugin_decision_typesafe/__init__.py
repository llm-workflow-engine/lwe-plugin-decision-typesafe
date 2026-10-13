"""Adapt LangChain's TypeSafe classifier to LWE's decision API."""

from __future__ import annotations

from typing import cast

import httpx2
from langchain_typesafe import TypeSafeClassifier
from langchain_typesafe.client import (
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPITimeoutError,
)
from langchain_typesafe.types import Question, State
from pydantic import TypeAdapter

from lwe.decisions import DecisionError, DecisionRequest, DecisionSettings


class TypeSafeDecisionProvider:
    """Execute synchronous Jev decisions using the published classifier wrapper."""

    def validate(self, request: DecisionRequest, settings: DecisionSettings) -> None:
        """Reject media unsupported by the classifier and unknown options."""
        if request.has_media:
            raise ValueError("TypeSafe decision requests do not support media")
        if settings.options:
            raise ValueError(
                "The TypeSafe decision adapter does not accept provider options"
            )

    def invoke(
        self, request: DecisionRequest, settings: DecisionSettings
    ) -> dict[str, object]:
        """Convert question types and serialize answers into the shared wire schema.

        Only the synchronous client is used and acquires network resources. The
        classifier's automatically created asynchronous client never sends requests.
        """
        questions = TypeAdapter(dict[str, Question]).validate_python(
            {
                name: question.model_dump(exclude_none=True)
                for name, question in request.questions.items()
            }
        )
        try:
            token = settings.credentials("TYPESAFE_API_KEY")
        except ValueError as error:
            raise DecisionError("TypeSafe API credentials are missing") from error
        try:
            with httpx2.Client(timeout=settings.timeout) as client:
                classifier = TypeSafeClassifier(
                    model=settings.model,
                    api_key=token,
                    timeout=settings.timeout,
                    client=client,
                )
                response = classifier.invoke(
                    {"state": cast(State, request.state), "questions": questions}
                )
                return response.model_dump(mode="json")
        except (TypeSafeAPIConnectionError, TypeSafeAPITimeoutError) as error:
            raise DecisionError(
                "TypeSafe decision transport failed", retryable=True
            ) from error

        except TypeSafeAPIError as error:
            status = error.status_code
            raise DecisionError(
                f"TypeSafe decision request failed (HTTP {status})",
                retryable=status in {408, 429} or status >= 500,
            ) from error

    def prepare(
        self, request: DecisionRequest, settings: DecisionSettings
    ) -> DecisionRequest:
        """Return the validated text/JSON request without any external media loading."""
        return request
