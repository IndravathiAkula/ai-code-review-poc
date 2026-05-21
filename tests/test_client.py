import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from reviewer.client import _looks_like_github_token, build_client


def test_looks_like_github_token_accepts_known_prefixes():
    assert _looks_like_github_token("ghp_aaaaaaaaaaaaaaaaaaaa")
    assert _looks_like_github_token("github_pat_aaaaaaaaaaaa")
    assert _looks_like_github_token("ghs_aaaaaaaaaaaaaaaaaaaa")
    assert _looks_like_github_token("gho_aaaaaaaaaaaaaaaaaaaa")
    assert _looks_like_github_token("ghu_aaaaaaaaaaaaaaaaaaaa")
    assert _looks_like_github_token("ghr_aaaaaaaaaaaaaaaaaaaa")


def test_looks_like_github_token_rejects_unknown_prefix():
    assert not _looks_like_github_token("not-a-token")
    assert not _looks_like_github_token("sk-openai-live")
    assert not _looks_like_github_token("")


def test_build_client_raises_when_token_missing(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="GITHUB_TOKEN"):
        build_client()


def test_build_client_warns_on_suspicious_token_format(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_TOKEN", "definitely-not-a-pat")
    build_client()
    err = capsys.readouterr().err
    assert "does not have a recognised GitHub prefix" in err


def test_build_client_does_not_warn_on_valid_token(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_" + "x" * 36)
    build_client()
    err = capsys.readouterr().err
    assert "does not have a recognised GitHub prefix" not in err


def test_build_client_prefers_model_api_token_over_github_token(monkeypatch):
    """When both are set, MODEL_API_TOKEN wins so the GitHub App
    installation token (which may lack models:read) is not used for
    inference."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghs_app_token_for_posting")
    monkeypatch.setenv("MODEL_API_TOKEN", "ghp_" + "x" * 36)
    client = build_client()
    # The Azure SDK's credential holds the actual token in its `key` attr.
    cred = client._config.credential
    assert cred.key == "ghp_" + "x" * 36


def test_build_client_falls_back_to_github_token_when_model_api_token_missing(monkeypatch):
    monkeypatch.delenv("MODEL_API_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_" + "y" * 36)
    client = build_client()
    cred = client._config.credential
    assert cred.key == "ghp_" + "y" * 36


def test_build_client_explicit_token_arg_beats_env_vars(monkeypatch):
    """A caller passing token= explicitly should override any env state."""
    monkeypatch.setenv("MODEL_API_TOKEN", "from-env")
    monkeypatch.setenv("GITHUB_TOKEN", "also-from-env")
    client = build_client(token="explicit_caller_token")
    cred = client._config.credential
    assert cred.key == "explicit_caller_token"
