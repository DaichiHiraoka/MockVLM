"""Research model selection and local runtime diagnostics."""

from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from privacy_vlm_poc.config import Settings, get_settings

RECOMMENDED_OLLAMA_MODEL = "gemma3:4b"
HIGHER_QUALITY_OLLAMA_MODEL = "gemma3:12b"
EFFICIENT_ALTERNATIVE_MODEL = "qwen2.5vl:3b"


@dataclass(frozen=True)
class ModelCandidate:
    name: str
    role: str
    expected_download_size: str
    rationale: str
    command: str


@dataclass
class OllamaDoctorResult:
    ollama_command_available: bool
    ollama_executable: str | None
    host: str
    host_reachable: bool
    configured_model: str
    configured_model_present: bool
    installed_models: list[str]
    recommended_model: str
    pull_command: str
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def find_ollama_executable() -> str | None:
    configured = os.getenv("OLLAMA_EXE")
    if configured and Path(configured).exists():
        return configured

    from_path = shutil.which("ollama")
    if from_path:
        return from_path

    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        default_path = Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe"
        if default_path.exists():
            return str(default_path)
    return None


def model_candidates() -> list[ModelCandidate]:
    return [
        ModelCandidate(
            name=RECOMMENDED_OLLAMA_MODEL,
            role="minimum_research_default",
            expected_download_size="about 3.3GB",
            rationale=(
                "Small local Gemma 3 vision model with text+image input, long context, and multilingual support. "
                "Use this as the default local model for repeated sampling/masking experiments."
            ),
            command=f"ollama pull {RECOMMENDED_OLLAMA_MODEL}",
        ),
        ModelCandidate(
            name=HIGHER_QUALITY_OLLAMA_MODEL,
            role="higher_quality_local",
            expected_download_size="about 8.1GB",
            rationale=(
                "Better quality target on machines that can tolerate more memory use and latency. "
                "Use it to confirm whether conclusions are stable under a stronger local VLM."
            ),
            command=f"ollama pull {HIGHER_QUALITY_OLLAMA_MODEL}",
        ),
        ModelCandidate(
            name=EFFICIENT_ALTERNATIVE_MODEL,
            role="cross_model_check",
            expected_download_size="about 3.2GB",
            rationale=(
                "Qwen2.5-VL small vision model. Use as a cross-model robustness check "
                "after Gemma 3 experiments are working."
            ),
            command=f"ollama pull {EFFICIENT_ALTERNATIVE_MODEL}",
        ),
    ]


def _get_json(url: str, timeout: float = 5.0) -> dict:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ollama_doctor(settings: Settings | None = None) -> OllamaDoctorResult:
    settings = settings or get_settings()
    notes: list[str] = []
    executable = find_ollama_executable()
    command_available = executable is not None
    if not executable:
        notes.append("Ollama command is not on PATH. Install Ollama before running the local VLM backend.")

    installed_models: list[str] = []
    host_reachable = False
    try:
        data = _get_json(f"{settings.ollama_host.rstrip('/')}/api/tags")
        host_reachable = True
        installed_models = [item.get("name", "") for item in data.get("models", []) if item.get("name")]
    except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
        notes.append(f"Ollama host is not reachable at {settings.ollama_host}: {exc}")

    model_present = settings.ollama_model in installed_models
    if host_reachable and not model_present:
        notes.append(f"Configured model is not installed: {settings.ollama_model}")
    if not settings.ollama_enabled:
        notes.append("OLLAMA_ENABLED is false. The pipeline will not call Ollama until it is true.")

    return OllamaDoctorResult(
        ollama_command_available=command_available,
        ollama_executable=executable,
        host=settings.ollama_host,
        host_reachable=host_reachable,
        configured_model=settings.ollama_model,
        configured_model_present=model_present,
        installed_models=installed_models,
        recommended_model=RECOMMENDED_OLLAMA_MODEL,
        pull_command=f"ollama pull {settings.ollama_model or RECOMMENDED_OLLAMA_MODEL}",
        notes=notes,
    )
