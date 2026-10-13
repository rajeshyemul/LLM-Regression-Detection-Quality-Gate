import httpx
import pytest

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
async def test_live_ollama_chat_call():
    host = "http://localhost:11434"
    if not is_ollama_accessible(host):
        pytest.skip("Local Ollama daemon is not running or unreachable at localhost:11434")

    models = get_available_models(host)
    if not models:
        pytest.skip("Ollama is running but no models are downloaded locally")

    # Pick the first available model (e.g. qwen2.5:3b, or whatever is pulled)
    test_model = models[0]

    provider = OllamaProvider(host=host, timeout=30.0)
    request = LLMRequest(
        messages=[
            Message(role="system", content="You are a helpful customer service classifier."),
            Message(role="user", content="I cannot log in to my account. Please respond with JSON."),
        ],
        model=test_model,
        temperature=0.0,
    )

    response = await provider.generate(request)

    assert response.provider == "ollama"
    assert response.model.startswith(test_model.split(":")[0]) or response.model == test_model
    assert len(response.content) > 0
    assert response.latency_ms > 0
