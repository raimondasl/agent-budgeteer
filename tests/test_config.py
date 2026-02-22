"""Tests for budgeteer.config."""

import json

import pytest

from budgeteer.config import BudgeteerConfig
from budgeteer.models import ModelTier


class TestBudgeteerConfig:
    def test_defaults(self):
        cfg = BudgeteerConfig()
        assert cfg.default_model == "gpt-4o-mini"
        assert cfg.default_max_tokens == 1024
        assert cfg.default_temperature == 0.7
        assert cfg.model_tiers == []
        assert cfg.default_run_budget is None

    def test_from_dict_minimal(self):
        cfg = BudgeteerConfig.from_dict({"default_model": "claude-sonnet"})
        assert cfg.default_model == "claude-sonnet"
        assert cfg.default_max_tokens == 1024  # keeps defaults

    def test_from_dict_with_tiers(self):
        data = {
            "default_model": "gpt-4o",
            "model_tiers": [
                {
                    "name": "gpt-4o-mini",
                    "cost_per_prompt_token": 0.00015,
                    "cost_per_completion_token": 0.0006,
                    "max_context_window": 128000,
                    "tier": "cheap",
                },
                {
                    "name": "gpt-4o",
                    "cost_per_prompt_token": 0.005,
                    "cost_per_completion_token": 0.015,
                    "max_context_window": 128000,
                    "tier": "premium",
                },
            ],
        }
        cfg = BudgeteerConfig.from_dict(data)
        assert len(cfg.model_tiers) == 2
        assert cfg.model_tiers[0].tier == "cheap"

    def test_from_dict_with_run_budget(self):
        data = {
            "default_run_budget": {
                "run_id": "default",
                "hard_usd_cap": 1.0,
                "hard_token_cap": 50000,
            },
        }
        cfg = BudgeteerConfig.from_dict(data)
        assert cfg.default_run_budget is not None
        assert cfg.default_run_budget.hard_usd_cap == 1.0

    def test_from_json_file(self, tmp_path):
        data = {"default_model": "test-model", "default_max_tokens": 512}
        path = tmp_path / "config.json"
        path.write_text(json.dumps(data))

        cfg = BudgeteerConfig.from_file(path)
        assert cfg.default_model == "test-model"
        assert cfg.default_max_tokens == 512

    def test_from_yaml_file(self, tmp_path):
        pytest.importorskip("yaml")
        import yaml

        data = {"default_model": "yaml-model", "default_temperature": 0.5}
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(data))

        cfg = BudgeteerConfig.from_file(path)
        assert cfg.default_model == "yaml-model"
        assert cfg.default_temperature == 0.5

    def test_yaml_import_error(self, tmp_path, monkeypatch):
        path = tmp_path / "config.yaml"
        path.write_text("default_model: test")

        # Simulate pyyaml not installed
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "yaml":
                raise ImportError("no yaml")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="pyyaml is required"):
            BudgeteerConfig.from_file(path)
