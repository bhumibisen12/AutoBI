"""Provider-agnostic LLM client interface, OpenAI-compatible HTTP client, and MockLLM."""

from __future__ import annotations

import hashlib
import json
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, SecretStr, ValidationError

from core.config import Settings, get_settings
from core.logging import get_logger
from core.schemas.llm import (
    AmbiguityQuestion,
    CandidateKPI,
    ColumnUnderstanding,
    DatasetUnderstanding,
    LLMCallMetadata,
    TokenUsage,
    UserIntent,
)

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Base exception for all LLM client and structured-output errors."""


class LLMConfigurationError(LLMError):
    """Raised when LLM provider configuration or credentials are missing/invalid."""


class LLMRequestError(LLMError):
    """Raised when an LLM request fails permanently."""


class LLMTransientError(LLMRequestError):
    """Raised on retryable transport, timeout, rate-limit, or 5xx provider errors."""


class LLMSchemaValidationError(LLMError):
    """Raised when an LLM response fails JSON parsing or Pydantic schema validation."""


def compute_prompt_hash(
    *,
    provider: str,
    model: str,
    schema_name: str,
    schema_json: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    temperature: float,
) -> str:
    """Compute a deterministic SHA-256 cache key for an LLM request."""
    payload = {
        "provider": provider,
        "model": model,
        "schema_name": schema_name,
        "schema": schema_json,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "temperature": temperature,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ResponseCache:
    """Thread-safe prompt-hash response cache with optional JSON-file persistence."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        """Initialize an in-memory cache optionally backed by ``cache_dir``."""
        self._memory: dict[str, str] = {}
        self._cache_dir = cache_dir
        self._lock = Lock()

    def get(self, prompt_hash: str) -> str | None:
        """Return cached validated JSON content for ``prompt_hash`` if present."""
        with self._lock:
            if prompt_hash in self._memory:
                return self._memory[prompt_hash]
            if self._cache_dir is not None:
                cache_file = self._cache_dir / f"{prompt_hash}.json"
                if cache_file.is_file():
                    content = cache_file.read_text(encoding="utf-8")
                    self._memory[prompt_hash] = content
                    return content
        return None

    def set(self, prompt_hash: str, content: str) -> None:
        """Store validated JSON content under ``prompt_hash``."""
        with self._lock:
            self._memory[prompt_hash] = content
            if self._cache_dir is not None:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file = self._cache_dir / f"{prompt_hash}.json"
                cache_file.write_text(content, encoding="utf-8")

    def clear(self) -> None:
        """Clear in-memory cached entries."""
        with self._lock:
            self._memory.clear()

logger = get_logger("llm")

# Approximate per-1M-token USD pricing for common models (prompt, completion).
_MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "gemini-2.0-flash": (0.10, 0.40),
}

_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})
_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class LLMPrompt:
    """Structured prompt payload passed to an ``BaseLLMClient`` implementation."""

    system_prompt: str
    user_prompt: str
    temperature: float = 0.0
    max_tokens: int = 2048


@dataclass(frozen=True)
class RawLLMResult:
    """Raw text/JSON output and token counts returned by a provider call."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost from token counts using known model pricing (0.0 if unknown/local)."""
    lowered = model.strip().lower()
    for prefix, (in_rate, out_rate) in _MODEL_PRICING_PER_1M.items():
        if lowered == prefix or lowered.startswith(f"{prefix}-"):
            cost = (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000.0
            return round(cost, 8)
    return 0.0


def _strip_json_fences(raw_text: str) -> str:
    """Remove optional Markdown ```json fences around a JSON response."""
    match = _CODE_FENCE_RE.match(raw_text)
    return match.group(1) if match else raw_text.strip()


class BaseLLMClient(ABC):
    """Provider-agnostic LLM client with Pydantic validation, retries, caching, and cost logging."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        base_delay_seconds: float = 0.5,
        cache: ResponseCache | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        """Configure shared retry, timeout, caching, and telemetry behavior."""
        self.provider = provider
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(1, max_retries)
        self.base_delay_seconds = max(0.0, base_delay_seconds)
        self.cache = cache if cache is not None else ResponseCache()
        self._sleep = sleep_fn

    @abstractmethod
    def _invoke_raw(
        self,
        prompt: LLMPrompt,
        response_model: type[BaseModel],
    ) -> RawLLMResult:
        """Execute a single provider call and return raw JSON text + token counts."""

    def generate_structured(
        self,
        prompt: LLMPrompt,
        response_model: type[T],
        *,
        use_cache: bool = True,
    ) -> tuple[T, LLMCallMetadata]:
        """Generate and validate a Pydantic-typed response with caching and exponential backoff."""
        schema_json = response_model.model_json_schema()
        prompt_hash = compute_prompt_hash(
            provider=self.provider,
            model=self.model,
            schema_name=response_model.__name__,
            schema_json=schema_json,
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt,
            temperature=prompt.temperature,
        )

        if use_cache:
            cached_json = self.cache.get(prompt_hash)
            if cached_json is not None:
                parsed_cached = response_model.model_validate_json(cached_json)
                metadata = LLMCallMetadata(
                    provider=self.provider,
                    model=self.model,
                    prompt_hash=prompt_hash,
                    cached=True,
                    attempts=1,
                    latency_ms=0.0,
                    usage=TokenUsage(),
                )
                logger.info(
                    "LLM cache hit",
                    extra={
                        "provider": self.provider,
                        "model": self.model,
                        "schema": response_model.__name__,
                        "prompt_hash": prompt_hash[:12],
                        "cached": True,
                    },
                )
                return parsed_cached, metadata

        started = time.perf_counter()
        last_error: Exception | None = None
        raw_result: RawLLMResult | None = None
        attempts = 0

        for attempt in range(1, self.max_retries + 1):
            attempts = attempt
            try:
                raw_result = self._invoke_raw(prompt, response_model)
                break
            except LLMTransientError as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise LLMRequestError(
                        f"LLM request failed after {attempts} attempt(s): {exc}"
                    ) from exc
                delay = self.base_delay_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Transient LLM error on attempt %d/%d; retrying in %.2fs",
                    attempt,
                    self.max_retries,
                    delay,
                    extra={
                        "provider": self.provider,
                        "model": self.model,
                        "attempt": attempt,
                    },
                )
                self._sleep(delay)

        assert raw_result is not None, f"Unexpected empty result; last_error={last_error}"
        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)

        cleaned_json = _strip_json_fences(raw_result.content)
        try:
            parsed = response_model.model_validate_json(cleaned_json)
        except ValidationError as exc:
            raise LLMSchemaValidationError(
                f"LLM response failed {response_model.__name__} validation: {exc}"
            ) from exc

        total_tokens = raw_result.prompt_tokens + raw_result.completion_tokens
        cost = estimate_cost_usd(
            self.model,
            raw_result.prompt_tokens,
            raw_result.completion_tokens,
        )
        usage = TokenUsage(
            prompt_tokens=raw_result.prompt_tokens,
            completion_tokens=raw_result.completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost,
        )

        if use_cache:
            self.cache.set(prompt_hash, parsed.model_dump_json())

        metadata = LLMCallMetadata(
            provider=self.provider,
            model=self.model,
            prompt_hash=prompt_hash,
            cached=False,
            attempts=attempts,
            latency_ms=latency_ms,
            usage=usage,
        )
        logger.info(
            "LLM structured call completed",
            extra={
                "provider": self.provider,
                "model": self.model,
                "schema": response_model.__name__,
                "prompt_hash": prompt_hash[:12],
                "cached": False,
                "attempts": attempts,
                "latency_ms": latency_ms,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "estimated_cost_usd": usage.estimated_cost_usd,
            },
        )
        return parsed, metadata


class OpenAICompatibleLLMClient(BaseLLMClient):
    """Real provider implementation calling OpenAI-compatible ``/chat/completions`` via httpx."""

    def __init__(
        self,
        *,
        api_key: SecretStr,
        model: str,
        provider: str = "openai",
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        base_delay_seconds: float = 0.5,
        cache: ResponseCache | None = None,
        http_client: httpx.Client | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        """Initialize the OpenAI-compatible HTTP client without ever logging ``api_key``."""
        super().__init__(
            provider=provider,
            model=model,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            base_delay_seconds=base_delay_seconds,
            cache=cache,
            sleep_fn=sleep_fn,
        )
        raw_secret = api_key.get_secret_value().strip()
        if not raw_secret and provider != "ollama":
            raise LLMConfigurationError(
                f"LLM_API_KEY is required for provider '{provider}'. Set it in .env."
            )
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._http_client = http_client

    def _invoke_raw(
        self,
        prompt: LLMPrompt,
        response_model: type[BaseModel],
    ) -> RawLLMResult:
        """Send a structured JSON completion request to the OpenAI-compatible endpoint."""
        url = f"{self._base_url}/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        secret_value = self._api_key.get_secret_value().strip()
        if secret_value:
            headers["Authorization"] = f"Bearer {secret_value}"

        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": prompt.temperature,
            "max_tokens": prompt.max_tokens,
            "messages": [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": response_model.model_json_schema(),
                },
            },
        }

        owns_client = self._http_client is None
        client = self._http_client or httpx.Client(timeout=self.timeout_seconds)
        try:
            response = client.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise LLMTransientError(
                f"Network/timeout error calling {self.provider}: {exc}"
            ) from exc
        finally:
            if owns_client:
                client.close()

        if response.status_code in _RETRYABLE_STATUS_CODES:
            raise LLMTransientError(
                f"Provider '{self.provider}' returned retryable HTTP {response.status_code}."
            )
        if response.status_code >= 400:
            raise LLMRequestError(
                f"Provider '{self.provider}' returned HTTP {response.status_code}."
            )

        try:
            data = response.json()
            content = str(data["choices"][0]["message"]["content"])
            usage = data.get("usage", {})
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            completion_tokens = int(usage.get("completion_tokens", 0))
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMSchemaValidationError(
                f"Malformed response envelope from '{self.provider}': {exc}"
            ) from exc

        return RawLLMResult(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


def _default_mock_dataset_understanding(prompt: LLMPrompt) -> DatasetUnderstanding:
    """Build a deterministic, schema-valid ``DatasetUnderstanding`` from a prompt context."""
    # Extract sanitized JSON inside <UNTRUSTED_DATASET_PROFILE> tags when present.
    match = re.search(
        r"<UNTRUSTED_DATASET_PROFILE>\s*(\{.*?\})\s*</UNTRUSTED_DATASET_PROFILE>",
        prompt.user_prompt,
        re.DOTALL,
    )
    columns: list[ColumnUnderstanding] = []
    candidate_kpis: list[CandidateKPI] = []
    domain = "general_analytics"

    if match:
        try:
            ctx = json.loads(match.group(1))
            tables = ctx.get("tables", [])
            all_col_names: list[str] = []
            for tbl in tables:
                t_name = str(tbl.get("table_name", "dataset"))
                for col in tbl.get("columns", []):
                    c_name = str(col.get("column_name", "col"))
                    sem = str(col.get("semantic_type", "categorical_dimension"))
                    all_col_names.append(c_name.lower())
                    if sem in {"numeric_measure", "currency", "percent"}:
                        role = "measure"
                        aggs = ["sum", "avg"]
                    elif sem == "date_time":
                        role = "date"
                        aggs = ["min", "max"]
                    elif sem == "id":
                        role = "id"
                        aggs = ["distinct_count"]
                    elif sem == "free_text":
                        role = "text"
                        aggs = ["count"]
                    else:
                        role = "dimension"
                        aggs = ["distinct_count"]

                    friendly = c_name.replace("_", " ").strip().title() or "Column"
                    columns.append(
                        ColumnUnderstanding(
                            table_name=t_name,
                            column_name=c_name,
                            role=role,
                            friendly_name=friendly,
                            description=f"Inferred {role} column '{c_name}' in table '{t_name}'.",
                            suggested_aggregations=aggs,
                        )
                    )
                    if role == "measure" and len(candidate_kpis) < 3:
                        candidate_kpis.append(
                            CandidateKPI(
                                name=f"Total {friendly}",
                                description=f"Sum of {friendly} across selected filters.",
                                rationale=(
                                    f"'{c_name}' is a numeric/currency measure in '{t_name}'."
                                ),
                                related_columns=[f"{t_name}.{c_name}"],
                                suggested_aggregation="sum",
                            )
                        )

            joined = " ".join(all_col_names)
            if any(
                k in joined
                for k in (
                    "visit",
                    "billed",
                    "readmitted",
                    "patient",
                    "heart",
                    "disease",
                    "cholesterol",
                    "blood",
                    "bp",
                    "clinical",
                    "diagnosis",
                    "ekg",
                    "max hr",
                )
            ):
                domain = "healthcare"
            elif any(k in joined for k in ("salary", "employee", "attrition", "hire_date")):
                domain = "human_resources"
            elif any(k in joined for k in ("budget", "variance", "ledger", "journal", "expense")):
                domain = "finance"
            elif any(k in joined for k in ("gpa", "tuition", "student", "enrollment", "campus")):
                domain = "education"
            elif any(
                k in joined for k in ("mandi", "commodity", "modal_price", "arrivals", "crop")
            ):
                domain = "agriculture"
            elif any(k in joined for k in ("lead", "campaign", "conversion", "telegram", "click")):
                domain = "marketing"
            elif any(k in joined for k in ("revenue", "order", "sales", "product")):
                domain = "sales"
        except (ValueError, TypeError):
            pass

    return DatasetUnderstanding(
        business_domain=domain,
        summary=f"Deterministic MockLLM understanding for {domain} dataset.",
        columns=columns,
        candidate_kpis=candidate_kpis,
        ambiguities=[
            AmbiguityQuestion(
                question="Which primary date column should drive time-intelligence KPIs?",
                context="Multiple date or operational tables may be present.",
                options=["Order Date", "Transaction Date", "Hire Date"],
            )
        ],
    )


def _default_mock_user_intent(prompt: LLMPrompt) -> UserIntent:
    """Build a deterministic, schema-valid ``UserIntent`` from a user prompt."""
    lowered = prompt.user_prompt.lower()
    if "quarter" in lowered or "qtr" in lowered:
        grain = "quarter"
    elif "month" in lowered:
        grain = "month"
    elif "week" in lowered:
        grain = "week"
    elif "year" in lowered or "annual" in lowered:
        grain = "year"
    elif "day" in lowered or "daily" in lowered:
        grain = "day"
    else:
        grain = "month"

    visuals = ["kpi_card", "line_chart", "bar_chart"]
    if "table" in lowered:
        visuals.append("table")

    return UserIntent(
        goal="Analyze key business KPIs and trends from the curated dataset.",
        audience="executives" if "exec" in lowered else "business analysts",
        time_grain=grain,
        filters=[],
        requested_visuals=visuals,
        focus_kpis=["Total Revenue"] if "revenue" in lowered else [],
    )


class MockLLMClient(BaseLLMClient):
    """Deterministic zero-network LLM test double supporting scripted responses and errors."""

    def __init__(
        self,
        *,
        responses_by_model: Mapping[type[BaseModel], Any] | None = None,
        response_queue: Sequence[Any] | None = None,
        model: str = "mock-llm-v1",
        max_retries: int = 3,
        base_delay_seconds: float = 0.0,
        cache: ResponseCache | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        """Initialize MockLLM with optional scripted responses/exceptions and call recording."""
        super().__init__(
            provider="mock",
            model=model,
            timeout_seconds=5.0,
            max_retries=max_retries,
            base_delay_seconds=base_delay_seconds,
            cache=cache,
            sleep_fn=sleep_fn,
        )
        self._responses_by_model = dict(responses_by_model or {})
        self._queue: list[Any] = list(response_queue or [])
        self.recorded_prompts: list[LLMPrompt] = []
        self.call_count: int = 0

    def _invoke_raw(
        self,
        prompt: LLMPrompt,
        response_model: type[BaseModel],
    ) -> RawLLMResult:
        """Return scripted or default deterministic JSON without network access."""
        self.recorded_prompts.append(prompt)
        self.call_count += 1

        scripted: Any = None
        if self._queue:
            scripted = self._queue.pop(0)
        elif response_model in self._responses_by_model:
            scripted = self._responses_by_model[response_model]

        if isinstance(scripted, Exception):
            raise scripted

        if scripted is None:
            if response_model is DatasetUnderstanding:
                scripted = _default_mock_dataset_understanding(prompt)
            elif response_model is UserIntent:
                scripted = _default_mock_user_intent(prompt)
            else:
                raise LLMRequestError(
                    f"MockLLMClient has no default generator or scripted response for "
                    f"{response_model.__name__}."
                )

        if isinstance(scripted, BaseModel):
            content = scripted.model_dump_json()
        elif isinstance(scripted, dict):
            content = json.dumps(scripted)
        else:
            content = str(scripted)

        prompt_tokens = max(1, (len(prompt.system_prompt) + len(prompt.user_prompt)) // 4)
        completion_tokens = max(1, len(content) // 4)
        return RawLLMResult(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


MockLLM = MockLLMClient


def create_llm_client(
    settings: Settings | None = None,
    *,
    cache: ResponseCache | None = None,
    base_url: str | None = None,
    timeout_seconds: float = 30.0,
    max_retries: int = 3,
) -> BaseLLMClient:
    """Instantiate the configured LLM client from ``Settings``."""
    cfg = settings or get_settings()
    if cfg.llm_provider == "mock":
        return MockLLMClient(model=cfg.llm_model, max_retries=max_retries, cache=cache)

    default_urls = {
        "openai": "https://api.openai.com/v1",
        "openai_compatible": "https://api.openai.com/v1",
        "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
        "groq": "https://api.groq.com/openai/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "ollama": "http://localhost:11434/v1",
    }
    if cfg.llm_provider in default_urls:
        return OpenAICompatibleLLMClient(
            api_key=cfg.llm_api_key,
            model=cfg.llm_model,
            provider=cfg.llm_provider,
            base_url=base_url or default_urls[cfg.llm_provider],
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            cache=cache,
        )
    raise LLMConfigurationError(
        f"Unsupported LLM_PROVIDER '{cfg.llm_provider}'. "
        f"Supported: mock, {', '.join(sorted(default_urls))}."
    )
