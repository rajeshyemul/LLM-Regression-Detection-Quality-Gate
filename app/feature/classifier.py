import json
from typing import Literal
from pydantic import BaseModel, ValidationError

from app.feature.prompt_loader import PromptConfig
from app.llm.base import LLMProvider, LLMRequest, Message


class ClassificationOutput(BaseModel):
    """Structured contract expected from the classifier feature."""
    category: Literal[
        "billing",
        "technical",
        "account",
        "general",
    ]
    summary: str


class ClassificationError(RuntimeError):
    """Raised when LLM output cannot be parsed or fails schema validation."""
    def __init__(self, message: str, raw_output: str | None = None):
        super().__init__(message)
        self.raw_output = raw_output


async def classify_email(
    email_text: str,
    provider: LLMProvider,
    prompt_config: PromptConfig,
) -> ClassificationOutput:
    """Classify a customer support email using the configured prompt and provider.

    Args:
        email_text: The raw incoming customer email content.
        provider: An initialized LLMProvider instance (Mock, Replay, or Ollama).
        prompt_config: The versioned prompt configuration to use.

    Returns:
        Validated ClassificationOutput object.

    Raises:
        ClassificationError: If the model response is malformed or invalid.
    """
    messages: list[Message] = [
        Message(role="system", content=prompt_config.system_prompt)
    ]

    # In-context few-shot demonstration pairs (if configured in PromptConfig)
    for example in prompt_config.few_shot_examples:
        messages.append(Message(role="user", content=example.input))
        messages.append(Message(role="assistant", content=example.output))

    # Add the target email to classify
    messages.append(
        Message(
            role="user",
            content=f"Customer Email:\n\"\"\"\n{email_text.strip()}\n\"\"\"",
        )
    )

    request = LLMRequest(
        messages=messages,
        model=prompt_config.model.name,
        temperature=prompt_config.model.temperature,
        response_schema=ClassificationOutput.model_json_schema(),
    )

    response = await provider.generate(request)

    try:
        return ClassificationOutput.model_validate_json(response.content)
    except (ValidationError, json.JSONDecodeError) as exc:
        raise ClassificationError(
            f"Failed to validate LLM response into ClassificationOutput: {exc}",
            raw_output=response.content,
        ) from exc