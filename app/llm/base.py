from abc import ABC, abstractmethod
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str


class LLMRequest(BaseModel):
    messages: list[Message]
    model: str
    temperature: float = 0.0
    response_schema: dict | None = None


class LLMResponse(BaseModel):
    content: str
    model: str
    provider: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float
    metadata: dict = Field(default_factory=dict)


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from the configured LLM provider."""
        pass
