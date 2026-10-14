from pathlib import Path
import pytest

from app.feature.classifier import (
    ClassificationError,
    ClassificationOutput,
    classify_email,
)
from app.feature.prompt_loader import (
    FewShotExample,
    PromptConfig,
    PromptModelConfig,
    load_prompt,
)
from app.llm.base import LLMProvider, LLMRequest, LLMResponse
from app.llm.mock import MockProvider


def test_load_prompt_v1():
    config = load_prompt("prompts/v1.yaml")
    assert config.version == "v1.0"
    assert config.model.name == "qwen2.5:3b"
    assert config.model.temperature == 0.0
    assert "billing" in config.system_prompt
    assert "technical" in config.system_prompt
    assert isinstance(config.few_shot_examples, list)


def test_load_prompt_missing_file_raises():
    with pytest.raises(FileNotFoundError, match="Prompt configuration file not found"):
        load_prompt("prompts/non_existent.yaml")


def test_load_prompt_invalid_yaml_raises(tmp_path: Path):
    invalid_file = tmp_path / "invalid.yaml"
    invalid_file.write_text("- item1\n- item2\n", encoding="utf-8")  # A list, not a dict

    with pytest.raises(ValueError, match="expected a dictionary"):
        load_prompt(invalid_file)


@pytest.mark.asyncio
async def test_classify_email_success():
    prompt = load_prompt("prompts/v1.yaml")
    provider = MockProvider(
        response='{"category":"billing","summary":"Customer requested a full refund."}'
    )

    result = await classify_email(
        email_text="Please refund my order #12345.",
        provider=provider,
        prompt_config=prompt,
    )

    assert isinstance(result, ClassificationOutput)
    assert result.category == "billing"
    assert result.summary == "Customer requested a full refund."


@pytest.mark.asyncio
async def test_classify_email_with_few_shots():
    # Capture the outgoing LLMRequest to verify few-shot formatting
    captured_request: list[LLMRequest] = []

    class CapturingMockProvider(LLMProvider):
        async def generate(self, request: LLMRequest) -> LLMResponse:
            captured_request.append(request)
            return LLMResponse(
                content='{"category":"technical","summary":"Login failure"}',
                model=request.model,
                provider="mock",
                latency_ms=1.0,
            )

    prompt = PromptConfig(
        version="v1.1-test",
        model=PromptModelConfig(name="qwen2.5:3b", temperature=0.0),
        system_prompt="Test classifier instructions.",
        few_shot_examples=[
            FewShotExample(
                input="Error 500 when saving profile.",
                output='{"category":"technical","summary":"Profile save error 500."}',
            )
        ],
    )

    provider = CapturingMockProvider()
    result = await classify_email(
        email_text="Cannot reset my password.",
        provider=provider,
        prompt_config=prompt,
    )

    assert result.category == "technical"
    assert len(captured_request) == 1
    messages = captured_request[0].messages

    # Verify message sequence: system -> user few-shot -> assistant few-shot -> user target email
    assert messages[0].role == "system"
    assert messages[1].role == "user"
    assert "Error 500" in messages[1].content
    assert messages[2].role == "assistant"
    assert messages[3].role == "user"
    assert "Cannot reset my password." in messages[3].content


@pytest.mark.asyncio
async def test_classify_email_malformed_json_raises_classification_error():
    prompt = load_prompt("prompts/v1.yaml")
    raw_bad_output = "I cannot classify this email."
    provider = MockProvider(response=raw_bad_output)

    with pytest.raises(ClassificationError) as exc_info:
        await classify_email(
            email_text="Hello support team.",
            provider=provider,
            prompt_config=prompt,
        )

    assert exc_info.value.raw_output == raw_bad_output


@pytest.mark.asyncio
async def test_classify_email_invalid_category_raises_classification_error():
    prompt = load_prompt("prompts/v1.yaml")
    # "sales" is not in Literal["billing", "technical", "account", "general"]
    raw_invalid_category = '{"category":"sales","summary":"Wants to buy enterprise tier."}'
    provider = MockProvider(response=raw_invalid_category)

    with pytest.raises(ClassificationError) as exc_info:
        await classify_email(
            email_text="I want to buy 100 enterprise licenses.",
            provider=provider,
            prompt_config=prompt,
        )

    assert exc_info.value.raw_output == raw_invalid_category