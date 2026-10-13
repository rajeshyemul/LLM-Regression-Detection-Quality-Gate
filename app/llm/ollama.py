import httpx
from app.llm.base import LLMProvider, LLMRequest, LLMResponse


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        host: str = "http://localhost:11434",
    ):
        self.host = host.rstrip("/")

    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError(
            "OllamaProvider implementation will be added in Phase 0.5."
        )
