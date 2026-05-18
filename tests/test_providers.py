"""Tests for the Provider abstraction and factory.

We don't hit any real network here — providers are exercised via
mocked clients or the factory's known-endpoints registry."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from reviewer.providers import (
    AnthropicProvider, ChatResponse, Provider,
    ProviderPermanentError, ProviderTransientError,
    Usage, build_provider, known_providers, PROVIDER_REGISTRY,
)
from reviewer.providers.github_models import GitHubModelsProvider
from reviewer.providers.openai_compat import OpenAICompatProvider


# ---------- base types ----------

def test_usage_total_tokens():
    u = Usage(prompt_tokens=100, completion_tokens=50)
    assert u.total_tokens == 150


def test_chat_response_dataclass():
    r = ChatResponse(content="hi", usage=Usage(10, 5))
    assert r.content == "hi"
    assert r.usage.total_tokens == 15


def test_provider_transient_error_carries_status_and_retry_after():
    e = ProviderTransientError("rate limited", status_code=429,
                                retry_after_seconds=12.0)
    assert e.status_code == 429
    assert e.retry_after_seconds == 12.0


def test_provider_permanent_error_carries_status():
    e = ProviderPermanentError("forbidden", status_code=403)
    assert e.status_code == 403


# ---------- factory ----------

def test_known_providers_includes_github_models_and_open_ones():
    names = known_providers()
    for expected in ("github-models", "openai", "anthropic", "groq",
                     "openrouter", "nvidia", "together", "anyscale",
                     "cerebras", "ollama"):
        assert expected in names, f"missing {expected}"


def test_build_provider_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown provider"):
        build_provider("not-a-real-provider", env={})


def test_build_provider_default_is_github_models(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_" + "x" * 36)
    p = build_provider(env={"GITHUB_TOKEN": "ghp_" + "x" * 36})
    assert p.name == "github-models"
    assert isinstance(p, GitHubModelsProvider)


def test_build_provider_from_env_variable():
    p = build_provider(env={
        "REVIEWER_PROVIDER": "groq",
        "GROQ_API_KEY": "gsk_test",
    })
    assert p.name == "groq"
    assert isinstance(p, OpenAICompatProvider)


def test_build_provider_missing_required_api_key_raises():
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        build_provider("groq", env={})


def test_build_provider_ollama_does_not_require_auth():
    p = build_provider("ollama", env={})
    assert p.name == "ollama"


def test_build_provider_custom_provider_via_env():
    p = build_provider(env={
        "REVIEWER_PROVIDER": "custom",
        "REVIEWER_BASE_URL": "https://my-endpoint.example.com/v1",
        "REVIEWER_API_KEY_ENV": "MY_API_KEY",
        "MY_API_KEY": "sk-test",
    })
    assert p.name == "custom"
    assert isinstance(p, OpenAICompatProvider)


def test_build_provider_custom_provider_without_base_url_raises():
    with pytest.raises(RuntimeError, match="REVIEWER_BASE_URL"):
        build_provider(env={"REVIEWER_PROVIDER": "custom"})


def test_provider_registry_groq_endpoint():
    spec = PROVIDER_REGISTRY["groq"]
    assert spec.base_url == "https://api.groq.com/openai/v1"
    assert spec.auth_env == "GROQ_API_KEY"


# ---------- GitHubModelsProvider behavior via injected client ----------

class _FakeAzureUsage:
    def __init__(self, p, c):
        self.prompt_tokens = p
        self.completion_tokens = c
        self.total_tokens = p + c


class _FakeAzureMessage:
    def __init__(self, content):
        self.content = content


class _FakeAzureChoice:
    def __init__(self, content):
        self.message = _FakeAzureMessage(content)


class _FakeAzureResponse:
    def __init__(self, content, usage=None):
        self.choices = [_FakeAzureChoice(content)]
        self.usage = usage


class _FakeAzureClient:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def test_github_models_provider_translates_response_to_chat_response():
    client = _FakeAzureClient(_FakeAzureResponse(
        "ok", usage=_FakeAzureUsage(100, 50)))
    p = GitHubModelsProvider(client=client)
    out = p.complete(
        model="openai/gpt-4o-mini",
        messages=[{"role": "system", "content": "sys"},
                  {"role": "user", "content": "user"}],
        temperature=0.2,
    )
    assert out.content == "ok"
    assert out.usage.prompt_tokens == 100
    assert out.usage.completion_tokens == 50


def test_github_models_provider_converts_role_dicts_to_sdk_messages():
    """The provider must translate {"role","content"} → Azure SDK objects
    so the existing test fakes keep working through the legacy adapter."""
    from azure.ai.inference.models import SystemMessage, UserMessage
    client = _FakeAzureClient(_FakeAzureResponse("ok"))
    p = GitHubModelsProvider(client=client)
    p.complete(
        model="x",
        messages=[{"role": "system", "content": "s"},
                  {"role": "user", "content": "u"}],
        temperature=0.0,
    )
    sent = client.calls[0]["messages"]
    assert isinstance(sent[0], SystemMessage)
    assert isinstance(sent[1], UserMessage)


def test_github_models_provider_translates_429_to_transient_error():
    from azure.core.exceptions import HttpResponseError

    class _Boom:
        def complete(self, **kw):
            e = HttpResponseError(message="rate limited")
            e.status_code = 429
            raise e

    p = GitHubModelsProvider(client=_Boom())
    with pytest.raises(ProviderTransientError) as exc:
        p.complete(model="x", messages=[{"role": "user", "content": "x"}],
                   temperature=0.0)
    assert exc.value.status_code == 429


def test_github_models_provider_translates_401_to_permanent_error():
    from azure.core.exceptions import HttpResponseError

    class _Boom:
        def complete(self, **kw):
            e = HttpResponseError(message="unauthorized")
            e.status_code = 401
            raise e

    p = GitHubModelsProvider(client=_Boom())
    with pytest.raises(ProviderPermanentError) as exc:
        p.complete(model="x", messages=[{"role": "user", "content": "x"}],
                   temperature=0.0)
    assert exc.value.status_code == 401


def test_github_models_provider_translates_connection_error_to_transient():
    from azure.core.exceptions import ServiceRequestError

    class _Boom:
        def complete(self, **kw):
            raise ServiceRequestError(message="conn reset")

    p = GitHubModelsProvider(client=_Boom())
    with pytest.raises(ProviderTransientError):
        p.complete(model="x", messages=[{"role": "user", "content": "x"}],
                   temperature=0.0)


# ---------- OpenAICompatProvider boundary ----------

def test_openai_compat_provider_uses_no_auth_placeholder_for_ollama(monkeypatch):
    """A provider with no API key shouldn't blow up at construction
    (Ollama / vLLM / on-prem endpoints often have no auth)."""
    # Just constructing should not raise.
    p = OpenAICompatProvider(
        name="ollama",
        base_url="http://localhost:11434/v1",
        api_key=None,
    )
    assert p.name == "ollama"


# ---------- AnthropicProvider with prompt caching ----------

class _FakeAnthropicUsage:
    def __init__(self, input_tokens, output_tokens,
                 cache_creation_input_tokens=0, cache_read_input_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens
        self.cache_read_input_tokens = cache_read_input_tokens


class _FakeAnthropicTextBlock:
    def __init__(self, text):
        self.text = text
        self.type = "text"


class _FakeAnthropicMessage:
    def __init__(self, text, usage=None):
        self.content = [_FakeAnthropicTextBlock(text)]
        self.usage = usage


class _FakeAnthropicMessages:
    def __init__(self, response, exc_to_raise=None):
        self._response = response
        self._exc = exc_to_raise
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._exc is not None:
            raise self._exc
        return self._response


class _FakeAnthropicClient:
    def __init__(self, response=None, exc_to_raise=None):
        self.messages = _FakeAnthropicMessages(response, exc_to_raise)


def test_anthropic_provider_returns_text_and_usage():
    client = _FakeAnthropicClient(
        response=_FakeAnthropicMessage(
            "ok", usage=_FakeAnthropicUsage(100, 50)))
    p = AnthropicProvider(api_key="sk-test", client=client)
    out = p.complete(
        model="claude-sonnet-4-6",
        messages=[
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "user prompt"},
        ],
        temperature=0.0,
    )
    assert out.content == "ok"
    assert out.usage.prompt_tokens == 100
    assert out.usage.completion_tokens == 50


def test_anthropic_provider_marks_system_prompt_as_cacheable():
    """The whole point of this provider: the system prompt must carry
    cache_control so subsequent chunks within 5 min are cheap."""
    client = _FakeAnthropicClient(
        response=_FakeAnthropicMessage("ok", usage=_FakeAnthropicUsage(10, 5)))
    p = AnthropicProvider(api_key="sk-test", client=client)
    p.complete(
        model="claude-sonnet-4-6",
        messages=[
            {"role": "system", "content": "long system prompt"},
            {"role": "user", "content": "user"},
        ],
        temperature=0.0,
    )
    sent = client.messages.calls[0]
    assert isinstance(sent["system"], list)
    assert sent["system"][0]["text"] == "long system prompt"
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_provider_can_disable_caching():
    client = _FakeAnthropicClient(
        response=_FakeAnthropicMessage("ok", usage=_FakeAnthropicUsage(10, 5)))
    p = AnthropicProvider(api_key="sk-test", client=client,
                           cache_system_prompt=False)
    p.complete(
        model="claude-sonnet-4-6",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "user"},
        ],
        temperature=0.0,
    )
    sent = client.messages.calls[0]
    assert sent["system"] == "sys"  # plain string, no cache block


def test_anthropic_provider_extracts_system_separate_from_messages():
    """Anthropic's API takes ``system`` as a top-level kwarg, not as
    a role inside the messages list."""
    client = _FakeAnthropicClient(
        response=_FakeAnthropicMessage("ok", usage=_FakeAnthropicUsage(10, 5)))
    p = AnthropicProvider(api_key="sk-test", client=client)
    p.complete(
        model="claude-sonnet-4-6",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
        ],
        temperature=0.0,
    )
    sent = client.messages.calls[0]
    assert sent["messages"] == [{"role": "user", "content": "u"}]


def test_anthropic_provider_reports_cache_tokens_in_usage():
    client = _FakeAnthropicClient(
        response=_FakeAnthropicMessage(
            "ok",
            usage=_FakeAnthropicUsage(
                input_tokens=10, output_tokens=5,
                cache_creation_input_tokens=800,
                cache_read_input_tokens=0,
            ),
        ),
    )
    p = AnthropicProvider(api_key="sk-test", client=client)
    out = p.complete(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "x"}],
        temperature=0.0,
    )
    assert out.usage.prompt_tokens == 10
    assert out.usage.completion_tokens == 5
    assert out.usage.cache_creation_tokens == 800
    assert out.usage.cache_read_tokens == 0


def test_anthropic_provider_ignores_response_format():
    """Anthropic has no response_format param; we pass JSON instructions
    via the system prompt instead. The provider should not raise."""
    client = _FakeAnthropicClient(
        response=_FakeAnthropicMessage("ok", usage=_FakeAnthropicUsage(1, 1)))
    p = AnthropicProvider(api_key="sk-test", client=client)
    p.complete(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "x"}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    sent = client.messages.calls[0]
    assert "response_format" not in sent


def _make_anthropic_status_error(status_code):
    """Build a real APIStatusError. The SDK's constructor pulls
    ``response.request`` so we need a proper httpx Response."""
    import httpx
    from anthropic import APIStatusError
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code, headers={}, request=request)
    return APIStatusError(f"status {status_code}", response=response, body={})


def test_anthropic_provider_translates_429_to_transient():
    client = _FakeAnthropicClient(exc_to_raise=_make_anthropic_status_error(429))
    p = AnthropicProvider(api_key="sk-test", client=client)
    with pytest.raises(ProviderTransientError) as exc:
        p.complete(model="x", messages=[{"role": "user", "content": "x"}],
                   temperature=0.0)
    assert exc.value.status_code == 429


def test_anthropic_provider_translates_401_to_permanent():
    client = _FakeAnthropicClient(exc_to_raise=_make_anthropic_status_error(401))
    p = AnthropicProvider(api_key="sk-test", client=client)
    with pytest.raises(ProviderPermanentError) as exc:
        p.complete(model="x", messages=[{"role": "user", "content": "x"}],
                   temperature=0.0)
    assert exc.value.status_code == 401


def test_anthropic_provider_requires_api_key_when_no_client():
    with pytest.raises(RuntimeError, match="api_key is required"):
        AnthropicProvider(api_key=None)


def test_build_provider_anthropic_requires_api_key():
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        build_provider("anthropic", env={})


def test_build_provider_anthropic_returns_anthropic_provider():
    p = build_provider("anthropic", env={"ANTHROPIC_API_KEY": "sk-test"})
    assert p.name == "anthropic"
    assert isinstance(p, AnthropicProvider)


def test_anthropic_provider_rejects_multiple_system_messages():
    client = _FakeAnthropicClient(
        response=_FakeAnthropicMessage("ok", usage=_FakeAnthropicUsage(1, 1)))
    p = AnthropicProvider(api_key="sk-test", client=client)
    with pytest.raises(ValueError, match="multiple system messages"):
        p.complete(
            model="x",
            messages=[
                {"role": "system", "content": "a"},
                {"role": "system", "content": "b"},
            ],
            temperature=0.0,
        )
