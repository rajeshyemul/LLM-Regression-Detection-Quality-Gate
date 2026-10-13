import time
from typing import Any
import httpx

from app.llm.base import LLMProvider, LLMRequest, LLMResponse


class OllamaProviderError(RuntimeError):
    """Raised when Ollama request fails, times out, or returns an error."""
    pass


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        host: str = "http://localhost:11434",
        timeout: float = 60.0,
    ):
        self.host = host.rstrip("/")
        self.timeout = timeout

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response via Ollama's /api/chat endpoint."""
        url = f"{self.host}/api/chat"
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {"role": m.role, "content": m.content} for m in request.messages
            ],
            "stream": False,
            "options": {
                "temperature": request.temperature,
            },
        }

        if request.response_schema is not None:
            payload["format"] = request.response_schema

        start_time = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException as exc:
            raise OllamaProviderError(
                f"Ollama request timed out after {self.timeout}s: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise OllamaProviderError(
                f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.RequestError as exc:
            raise OllamaProviderError(
                f"Failed to connect to Ollama at {self.host}: {exc}"
            ) from exc
        except Exception as exc:
            raise OllamaProviderError(
                f"Unexpected error communicating with Ollama: {exc}"
            ) from exc

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        content = data.get("message", {}).get("content", "")
        return LLMResponse(
            content=content,
            model=data.get("model", request.model),
            provider="ollama",
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            latency_ms=latency_ms,
            metadata={
                "total_duration_ns": data.get("total_duration"),
                "load_duration_ns": data.get("load_duration"),
            },
        )
