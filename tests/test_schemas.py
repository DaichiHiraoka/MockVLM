from __future__ import annotations

import pytest
from pydantic import ValidationError

from privacy_vlm_poc.schemas import VLMResponse


def test_vlm_response_schema_accepts_valid_json() -> None:
    response = VLMResponse.model_validate(
        {
            "unauthorized_object_interaction_suspected": False,
            "confidence": 0.2,
            "target_object": None,
            "evidence_frames": [],
            "reason": "根拠が不足しています。",
            "privacy_sensitive_description_included": False,
            "limitations": "限定されたフレームのみです。",
        }
    )
    assert response.confidence == 0.2


def test_vlm_response_rejects_invalid_confidence() -> None:
    with pytest.raises(ValidationError):
        VLMResponse.model_validate(
            {
                "unauthorized_object_interaction_suspected": True,
                "confidence": 1.2,
                "target_object": "object",
                "evidence_frames": [1],
                "reason": "invalid",
                "privacy_sensitive_description_included": False,
                "limitations": "invalid",
            }
        )
