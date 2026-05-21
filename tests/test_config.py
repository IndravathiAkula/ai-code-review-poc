import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reviewer.config import (
    CONFIG_FILENAME, ReviewConfig, effective_config, load_config,
)


# ---------- load_config ----------

def test_load_config_returns_empty_dict_when_missing(tmp_path):
    assert load_config(tmp_path / "missing.yml") == {}


def test_load_config_parses_yaml(tmp_path):
    p = tmp_path / CONFIG_FILENAME
    p.write_text(
        "model: openai/gpt-4o\n"
        "min_confidence: 0.8\n"
        "block_patterns:\n  - vendor/**\n  - legacy/*\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg["model"] == "openai/gpt-4o"
    assert cfg["min_confidence"] == 0.8
    assert cfg["block_patterns"] == ["vendor/**", "legacy/*"]


def test_load_config_warns_and_drops_unknown_keys(tmp_path, capsys):
    p = tmp_path / CONFIG_FILENAME
    p.write_text("model: x\nbogus_key: 1\n", encoding="utf-8")
    cfg = load_config(p)
    assert "bogus_key" not in cfg
    err = capsys.readouterr().err
    assert "unknown config keys" in err
    assert "bogus_key" in err


def test_load_config_warns_when_top_level_is_not_a_mapping(tmp_path, capsys):
    p = tmp_path / CONFIG_FILENAME
    p.write_text("- just\n- a\n- list\n", encoding="utf-8")
    assert load_config(p) == {}
    assert "must be a mapping" in capsys.readouterr().err


def test_load_config_handles_empty_file(tmp_path):
    p = tmp_path / CONFIG_FILENAME
    p.write_text("", encoding="utf-8")
    assert load_config(p) == {}


# ---------- effective_config ----------

def test_effective_config_returns_defaults_when_nothing_set():
    cfg = effective_config(env={})
    expected = ReviewConfig()
    assert cfg.model == expected.model
    assert cfg.min_confidence == expected.min_confidence
    assert cfg.min_severity == expected.min_severity
    assert cfg.concurrency == expected.concurrency


def test_effective_config_env_overrides_default():
    cfg = effective_config(env={
        "MODEL": "openai/gpt-4o",
        "MIN_CONFIDENCE": "0.85",
        "REVIEWER_CONCURRENCY": "8",
        "REVIEWER_POST_SUMMARY": "false",
    })
    assert cfg.model == "openai/gpt-4o"
    assert cfg.min_confidence == 0.85
    assert cfg.concurrency == 8
    assert cfg.post_summary is False


def test_effective_config_file_overrides_env():
    """Repo-level .ai-review.yml has the final say over workflow env vars."""
    cfg = effective_config(
        env={"MODEL": "from-env", "MIN_CONFIDENCE": "0.5"},
        config_file={"model": "from-config", "min_confidence": 0.9},
    )
    assert cfg.model == "from-config"
    assert cfg.min_confidence == 0.9


def test_effective_config_env_fills_fields_not_in_config():
    cfg = effective_config(
        env={"MODEL": "from-env", "MIN_SEVERITY": "high"},
        config_file={"model": "from-config"},  # only sets model
    )
    assert cfg.model == "from-config"
    assert cfg.min_severity == "high"  # env wins where config is silent


def test_effective_config_block_patterns_from_env_string():
    cfg = effective_config(env={"REVIEWER_BLOCK_PATTERNS": "a,b , c"})
    assert cfg.block_patterns == ("a", "b", "c")


def test_effective_config_block_patterns_from_yaml_list():
    cfg = effective_config(config_file={"block_patterns": ["a", "b"]})
    assert cfg.block_patterns == ("a", "b")


def test_effective_config_post_summary_truthy_strings():
    for raw in ("true", "True", "1", "yes", "on"):
        cfg = effective_config(env={"REVIEWER_POST_SUMMARY": raw})
        assert cfg.post_summary is True, f"failed on {raw!r}"
    for raw in ("false", "False", "0", "no", "off"):
        cfg = effective_config(env={"REVIEWER_POST_SUMMARY": raw})
        assert cfg.post_summary is False, f"failed on {raw!r}"


def test_effective_config_invalid_value_keeps_previous_value(capsys):
    cfg = effective_config(env={"MIN_CONFIDENCE": "not-a-float"})
    assert cfg.min_confidence == ReviewConfig().min_confidence
    assert "MIN_CONFIDENCE" in capsys.readouterr().err


def test_effective_config_max_files_per_pr_from_yaml():
    cfg = effective_config(config_file={"max_files_per_pr": 50})
    assert cfg.max_files_per_pr == 50


def test_effective_config_max_tokens_per_pr_from_env_string():
    cfg = effective_config(env={"REVIEWER_MAX_TOKENS_PER_PR": "200000"})
    assert cfg.max_tokens_per_pr == 200000


def test_effective_config_caps_default_to_unlimited():
    cfg = effective_config(env={})
    assert cfg.max_files_per_pr == 0
    assert cfg.max_tokens_per_pr == 0


def test_effective_config_models_by_language_from_yaml():
    cfg = effective_config(config_file={
        "models_by_language": {"python": "openai/gpt-4o",
                               "typescript": "openai/gpt-4o-mini"},
    })
    assert cfg.models_by_language == {
        "python": "openai/gpt-4o",
        "typescript": "openai/gpt-4o-mini",
    }


def test_effective_config_skip_labels_default_includes_skip_ai_review():
    cfg = effective_config(env={})
    assert "skip-ai-review" in cfg.skip_labels


def test_effective_config_skip_labels_from_yaml_list():
    cfg = effective_config(config_file={
        "skip_labels": ["wip", "do-not-review"],
    })
    assert cfg.skip_labels == ("wip", "do-not-review")


def test_effective_config_skip_labels_from_env_string():
    cfg = effective_config(env={
        "REVIEWER_SKIP_LABELS": "wip, dependencies,generated-code",
    })
    assert cfg.skip_labels == ("wip", "dependencies", "generated-code")


def test_effective_config_skip_labels_yaml_empty_disables_gating():
    """Setting skip_labels to [] explicitly is how consumers opt OUT
    of label gating even though the default is non-empty."""
    cfg = effective_config(config_file={"skip_labels": []})
    assert cfg.skip_labels == ()
