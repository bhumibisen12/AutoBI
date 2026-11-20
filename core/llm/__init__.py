"""Provider-agnostic LLM wrapper, MockLLM, caching, and hardened prompt builders."""

from core.llm.client import (
    BaseLLMClient,
    LLMConfigurationError,
    LLMError,
    LLMPrompt,
    LLMRequestError,
    LLMSchemaValidationError,
    LLMTransientError,
    MockLLM,
    MockLLMClient,
    OpenAICompatibleLLMClient,
    RawLLMResult,
    ResponseCache,
    compute_prompt_hash,
    create_llm_client,
    estimate_cost_usd,
)
from core.llm.prompts import (
    build_dataset_understanding_prompt,
    build_safe_dataset_context,
    build_user_intent_prompt,
    parse_user_intent,
    understand_dataset,
)

__all__ = [
    "BaseLLMClient",
    "LLMConfigurationError",
    "LLMError",
    "LLMPrompt",
    "LLMRequestError",
    "LLMSchemaValidationError",
    "LLMTransientError",
    "MockLLM",
    "MockLLMClient",
    "OpenAICompatibleLLMClient",
    "RawLLMResult",
    "ResponseCache",
    "build_dataset_understanding_prompt",
    "build_safe_dataset_context",
    "build_user_intent_prompt",
    "compute_prompt_hash",
    "create_llm_client",
    "estimate_cost_usd",
    "parse_user_intent",
    "understand_dataset",
]
