from app.llm.base import LLMProvider, LLMRequest, LLMResponse


class MockProvider(LLMProvider):
    def __init__(
        self,
        response: str = '{"category":"billing","summary":"Customer reports a duplicate charge."}',
    ):
        self.response = response

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            content=self.response,
            model=request.model,
            provider="mock",
            input_tokens=10,
            output_tokens=10,
            latency_ms=1.0,
            metadata={
                "execution_mode": "mock",
            },
        )
