import pytest
from pathlib import Path

from app.llm.base import LLMRequest, LLMResponse, Message
from app.llm.factory import get_provider
from app.llm.mock import MockProvider
from app.llm.replay import ReplayProvider, compute_fixture_key
from app.llm.ollama import OllamaProvider


def test_factory_returns_mock_provider():
    provider = get_provider("mock")
    assert isinstance(provider, MockProvider)


def test_factory_returns_mock_with_custom_response():
    custom = '{"category":"technical","summary":"App crash"}'
    provider = get_provider("mock", mock_response=custom)
    assert isinstance(provider, MockProvider)
    assert provider.response == custom


def test_factory_returns_replay_provider(tmp_path: Path):
    provider = get_provider("replay", fixture_dir=tmp_path)
    assert isinstance(provider, ReplayProvider)
    assert provider.fixture_dir == tmp_path


def test_factory_returns_real_provider():
    provider = get_provider("real", host="http://my-ollama:11434")
    assert isinstance(provider, OllamaProvider)
    assert provider.host == "http://my-ollama:11434"


def test_factory_rejects_unsupported_mode():
    with pytest.raises(ValueError, match="Unsupported execution mode: 'xyz'"):
        get_provider("xyz")


def test_fixture_key_determinism():
    req1 = LLMRequest(
        messages=[
            Message(role="system", content="You are a classifier."),
            Message(role="user", content="Where is my refund?"),
        ],
        model="qwen2.5:3b",
        temperature=0.0,
    )
    req2 = LLMRequest(
        messages=[
            Message(role="system", content="You are a classifier."),
            Message(role="user", content="Where is my refund?"),
        ],
        model="qwen2.5:3b",
        temperature=0.0,
    )

    key1 = compute_fixture_key(req1)
    key2 = compute_fixture_key(req2)
    assert key1 == key2
    assert len(key1) == 16


def test_fixture_key_changes_on_model_or_prompt_difference():
    base_req = LLMRequest(
        messages=[Message(role="user", content="Refund status")],
        model="qwen2.5:3b",
        temperature=0.0,
    )
    model_diff = LLMRequest(
        messages=[Message(role="user", content="Refund status")],
        model="llama3.2:3b",
        temperature=0.0,
    )
    prompt_diff = LLMRequest(
        messages=[Message(role="user", content="Login failed")],
        model="qwen2.5:3b",
        temperature=0.0,
    )
    temp_diff = LLMRequest(
        messages=[Message(role="user", content="Refund status")],
        model="qwen2.5:3b",
        temperature=0.7,
    )

    base_key = compute_fixture_key(base_req)
    assert compute_fixture_key(model_diff) != base_key
    assert compute_fixture_key(prompt_diff) != base_key
    assert compute_fixture_key(temp_diff) != base_key


@pytest.mark.asyncio
async def test_replay_record_and_generate(tmp_path: Path):
    provider = ReplayProvider(fixture_dir=tmp_path)
    request = LLMRequest(
        messages=[Message(role="user", content="I was charged twice.")],
        model="test-model",
    )
    recorded_response = LLMResponse(
        content='{"category":"billing","summary":"Double charge"}',
        model="test-model",
        provider="ollama",
        input_tokens=45,
        output_tokens=15,
        latency_ms=850.0,
        metadata={"captured": True},
    )

    # 1. Record fixture
    fixture_path = provider.record(request, recorded_response)
    assert fixture_path.exists()

    # 2. Replay generation
    replayed = await provider.generate(request)
    assert replayed.content == recorded_response.content
    assert replayed.model == "test-model"
    assert replayed.provider == "replay"
    assert replayed.input_tokens == 45
    assert replayed.output_tokens == 15
    assert replayed.latency_ms == 850.0
    assert replayed.metadata == {"captured": True}


@pytest.mark.asyncio
async def test_replay_missing_fixture_raises_file_not_found(tmp_path: Path):
    provider = ReplayProvider(fixture_dir=tmp_path)
    request = LLMRequest(
        messages=[Message(role="user", content="Unrecorded request")],
        model="test-model",
    )

    with pytest.raises(FileNotFoundError, match="Replay fixture not found"):
        await provider.generate(request)


@pytest.mark.asyncio
async def test_ollama_provider_successful_generate(monkeypatch):
    import httpx
    from app.llm.ollama import OllamaProvider

    fake_response_data = {
        "model": "qwen2.5:3b",
        "message": {
            "role": "assistant",
            "content": '{"category":"billing","summary":"Double charge"}',
        },
        "prompt_eval_count": 50,
        "eval_count": 20,
        "total_duration": 120000000,
        "load_duration": 10000000,
    }

    class MockAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, json):
            # Assert payload structure
            assert url == "http://localhost:11434/api/chat"
            assert json["model"] == "qwen2.5:3b"
            assert json["options"]["temperature"] == 0.0
            assert json["format"] == {"type": "object"}
            request_obj = httpx.Request("POST", url)
            return httpx.Response(200, json=fake_response_data, request=request_obj)

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

    provider = OllamaProvider(host="http://localhost:11434")
    request = LLMRequest(
        messages=[Message(role="user", content="Need refund")],
        model="qwen2.5:3b",
        temperature=0.0,
        response_schema={"type": "object"},
    )
    response = await provider.generate(request)

    assert response.provider == "ollama"
    assert response.model == "qwen2.5:3b"
    assert response.content == '{"category":"billing","summary":"Double charge"}'
    assert response.input_tokens == 50
    assert response.output_tokens == 20
    assert response.latency_ms > 0


@pytest.mark.asyncio
async def test_ollama_provider_handles_timeout(monkeypatch):
    import httpx
    from app.llm.ollama import OllamaProvider, OllamaProviderError

    class MockTimeoutClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, json):
            raise httpx.TimeoutException("Connection timed out")

    monkeypatch.setattr(httpx, "AsyncClient", MockTimeoutClient)

    provider = OllamaProvider(host="http://localhost:11434", timeout=5.0)
    request = LLMRequest(
        messages=[Message(role="user", content="Slow request")],
        model="qwen2.5:3b",
    )

    with pytest.raises(OllamaProviderError, match="timed out after 5.0s"):
        await provider.generate(request)
