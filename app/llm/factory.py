from pathlib import Path
from app.llm.base import LLMProvider
from app.llm.mock import MockProvider
from app.llm.replay import ReplayProvider
from app.llm.ollama import OllamaProvider


def get_provider(
    mode: str,
    *,
    host: str | None = None,
    fixture_dir: str | Path | None = None,
    mock_response: str | None = None,
) -> LLMProvider:
    """Factory to instantiate the appropriate LLMProvider based on execution mode.

    Args:
        mode: 'mock', 'replay', or 'real'.
        host: Ollama host URL (used only in 'real' mode).
        fixture_dir: Directory where replay fixtures live (used in 'replay' mode).
        mock_response: Custom static response for MockProvider (used in 'mock' mode).

    Raises:
        ValueError: If mode is unsupported.
    """
    normalized_mode = mode.lower().strip()

    if normalized_mode == "mock":
        return (
            MockProvider(response=mock_response)
            if mock_response is not None
            else MockProvider()
        )

    if normalized_mode == "replay":
        return (
            ReplayProvider(fixture_dir=fixture_dir)
            if fixture_dir is not None
            else ReplayProvider()
        )

    if normalized_mode == "real":
        return (
            OllamaProvider(host=host)
            if host is not None
            else OllamaProvider()
        )

    raise ValueError(
        f"Unsupported execution mode: '{mode}'. Must be one of: 'mock', 'replay', 'real'."
    )
