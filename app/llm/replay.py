from pathlib import Path
from app.llm.base import LLMProvider, LLMRequest, LLMResponse


class ReplayProvider(LLMProvider):
    def __init__(self, fixture_dir: str = "fixtures/replay"):
        self.fixture_dir = Path(fixture_dir)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError(
            "ReplayProvider will be implemented after the replay contract is finalized."
        )
