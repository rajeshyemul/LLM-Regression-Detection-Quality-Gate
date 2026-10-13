import hashlib
import json
from pathlib import Path
from app.llm.base import LLMProvider, LLMRequest, LLMResponse


def compute_fixture_key(request: LLMRequest) -> str:
    """Compute a deterministic 16-character SHA-256 hash from request parameters.

    Derived purely from model, temperature, response_schema, and message contents
    without leaking replay concerns into the LLMRequest contract.
    """
    hasher = hashlib.sha256()
    hasher.update(request.model.encode("utf-8"))
    hasher.update(str(request.temperature).encode("utf-8"))

    if request.response_schema is not None:
        hasher.update(
            json.dumps(request.response_schema, sort_keys=True).encode("utf-8")
        )

    for msg in request.messages:
        hasher.update(msg.role.encode("utf-8"))
        hasher.update(msg.content.encode("utf-8"))

    return hasher.hexdigest()[:16]


class ReplayProvider(LLMProvider):
    def __init__(self, fixture_dir: str | Path = "fixtures/replay"):
        self.fixture_dir = Path(fixture_dir)
        self.fixture_dir.mkdir(parents=True, exist_ok=True)

    def get_fixture_path(self, request: LLMRequest) -> Path:
        key = compute_fixture_key(request)
        return self.fixture_dir / f"{key}.json"

    def record(self, request: LLMRequest, response: LLMResponse) -> Path:
        """Record an LLMResponse to disk as a replay fixture."""
        fixture_path = self.get_fixture_path(request)
        data = {
            "content": response.content,
            "model": response.model,
            "provider": response.provider,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency_ms": response.latency_ms,
            "metadata": response.metadata,
        }
        with open(fixture_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return fixture_path

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response by reading recorded fixture.

        Fails explicitly if the fixture is missing.
        """
        fixture_path = self.get_fixture_path(request)
        if not fixture_path.exists():
            raise FileNotFoundError(
                f"Replay fixture not found for request: {fixture_path}. "
                "Record the fixture first using real execution mode."
            )

        with open(fixture_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return LLMResponse(
            content=data["content"],
            model=data.get("model", request.model),
            provider="replay",
            input_tokens=data.get("input_tokens"),
            output_tokens=data.get("output_tokens"),
            latency_ms=data.get("latency_ms", 0.0),
            metadata=data.get("metadata", {}),
        )
