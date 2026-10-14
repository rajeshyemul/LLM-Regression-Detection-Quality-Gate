from pydantic import BaseModel, Field
from pathlib import Path
import yaml


class PromptModelConfig(BaseModel):
    """Execution parameters bound to this specific prompt version."""
    name: str = Field(..., description="Target model name, e.g. 'qwen2.5:3b'")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature")


class FewShotExample(BaseModel):
    """Optional few-shot example for in-context demonstration."""
    input: str = Field(..., description="Sample customer email text")
    output: str = Field(..., description="Expected JSON output string")


class PromptConfig(BaseModel):
    """A versioned, self-contained prompt artifact."""
    version: str = Field(..., description="Semantic prompt version, e.g. 'v1.0'")
    description: str = Field(default="", description="Change rationale or prompt intent")
    model: PromptModelConfig
    system_prompt: str = Field(..., min_length=10, description="System instructions for the model")
    few_shot_examples: list[FewShotExample] = Field(default_factory=list)

def load_prompt(prompt_path: str | Path) -> PromptConfig:
    """Load and validate a PromptConfig from a YAML file.

    Args:
        prompt_path: Path to the YAML prompt file.

    Returns:
        Validated PromptConfig instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file content is empty or not a valid dictionary.
        pydantic.ValidationError: If the content does not match PromptConfig schema.
    """
    path = Path(prompt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Prompt configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid prompt configuration in {path}: expected a dictionary, got {type(data).__name__}")

    return PromptConfig.model_validate(data)