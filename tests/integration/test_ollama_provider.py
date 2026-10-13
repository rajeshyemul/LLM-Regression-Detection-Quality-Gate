import httpx
import pytest

from app.feature.classifier import ClassificationOutput
from app.llm.base import LLMRequest, Message
from app.llm.ollama import OllamaProvider


def is_ollama_accessible(host: str = "http://localhost:11434") -> bool:
    """Check if the local Ollama daemon is reachable and responding."""
    try:
        resp = httpx.get(f"{host}/api/tags", timeout=1.5)
        return resp.status_code == 200
    except Exception:
        return False


def get_available_models(host: str = "http://localhost:11434") -> list[str]:
    """Retrieve list of currently pulled models from Ollama."""
    try:
        resp = httpx.get(f"{host}/api/tags", timeout=1.5)
        if resp.status_code == 200:
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_ollama_structured_chat_call():
    host = "http://localhost:11434"
    if not is_ollama_accessible(host):
        pytest.skip("Local Ollama daemon is not running or unreachable at localhost:11434")

    models = get_available_models(host)
    model_name = "qwen2.5:3b"
    # Find matching model tag (qwen2.5:3b or qwen2.5:3b-...)
    matched = [m for m in models if m.startswith(model_name)]
    if not matched:
        if not models:
            pytest.skip(f"Model '{model_name}' not found and no other models available")
        model_name = models[0]
    else:
        model_name = matched[0]

    provider = OllamaProvider(host=host, timeout=60.0)
    request = LLMRequest(
        messages=[
            Message(
                role="system",
                content=(
                    "You are a customer support classification assistant. "
                    "Classify the customer email into one of: billing, technical, account, general. "
                    "Provide a concise summary."
                ),
            ),
            Message(
                role="user",
                content="I was charged $49.99 twice on my credit card for this month's subscription.",
            ),
        ],
        model=model_name,
        temperature=0.0,
        response_schema=ClassificationOutput.model_json_schema(),
    )

    response = await provider.generate(request)

    # 1. Verify provider metadata
    assert response.provider == "ollama"
    assert response.model == model_name

    # 2. Verify token metrics and latency
    assert response.input_tokens is not None and response.input_tokens > 0
    assert response.output_tokens is not None and response.output_tokens > 0
    assert response.latency_ms > 0

    # 3. Verify structured JSON output parses cleanly into ClassificationOutput
    output = ClassificationOutput.model_validate_json(response.content)
    assert output.category == "billing"
    assert len(output.summary) > 0
