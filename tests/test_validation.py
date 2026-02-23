"""Tests for input validation (Milestone 12)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from budgeteer.config import BudgeteerConfig
from budgeteer.models import BudgetAccount, BudgetScope, ModelTier, RunBudget


# ------------------------------------------------------------------
# BudgeteerConfig.validate()
# ------------------------------------------------------------------


class TestConfigValidation:
    """Tests for BudgeteerConfig.validate()."""

    def test_default_config_is_valid(self):
        """Default config passes validation."""
        BudgeteerConfig().validate()

    def test_invalid_default_max_tokens(self):
        cfg = BudgeteerConfig(default_max_tokens=0)
        with pytest.raises(ValueError, match="default_max_tokens"):
            cfg.validate()

    def test_negative_default_max_tokens(self):
        cfg = BudgeteerConfig(default_max_tokens=-1)
        with pytest.raises(ValueError, match="default_max_tokens"):
            cfg.validate()

    def test_invalid_calibration_alpha_zero(self):
        cfg = BudgeteerConfig(calibration_alpha=0)
        with pytest.raises(ValueError, match="calibration_alpha"):
            cfg.validate()

    def test_invalid_calibration_alpha_above_one(self):
        cfg = BudgeteerConfig(calibration_alpha=1.5)
        with pytest.raises(ValueError, match="calibration_alpha"):
            cfg.validate()

    def test_valid_calibration_alpha_one(self):
        """Alpha = 1 is valid (edge case)."""
        cfg = BudgeteerConfig(calibration_alpha=1.0)
        cfg.validate()  # should not raise

    def test_negative_temperature(self):
        cfg = BudgeteerConfig(default_temperature=-0.1)
        with pytest.raises(ValueError, match="default_temperature"):
            cfg.validate()

    def test_zero_temperature_valid(self):
        """Temperature = 0 is valid (greedy decoding)."""
        cfg = BudgeteerConfig(default_temperature=0.0)
        cfg.validate()  # should not raise

    def test_negative_max_retries(self):
        cfg = BudgeteerConfig(max_retries=-1)
        with pytest.raises(ValueError, match="max_retries"):
            cfg.validate()

    def test_negative_retry_delay(self):
        cfg = BudgeteerConfig(retry_delay_ms=-100)
        with pytest.raises(ValueError, match="retry_delay_ms"):
            cfg.validate()

    def test_negative_roi_lambda_latency(self):
        cfg = BudgeteerConfig(roi_lambda_latency=-0.01)
        with pytest.raises(ValueError, match="roi_lambda_latency"):
            cfg.validate()

    def test_zero_roi_recommend_threshold(self):
        cfg = BudgeteerConfig(roi_recommend_threshold=0)
        with pytest.raises(ValueError, match="roi_recommend_threshold"):
            cfg.validate()

    def test_invalid_roi_budget_floor_zero(self):
        cfg = BudgeteerConfig(roi_budget_floor=0)
        with pytest.raises(ValueError, match="roi_budget_floor"):
            cfg.validate()

    def test_invalid_roi_budget_floor_above_one(self):
        cfg = BudgeteerConfig(roi_budget_floor=1.5)
        with pytest.raises(ValueError, match="roi_budget_floor"):
            cfg.validate()

    def test_invalid_roi_clarify_ambiguity_threshold(self):
        cfg = BudgeteerConfig(roi_clarify_ambiguity_threshold=0)
        with pytest.raises(ValueError, match="roi_clarify_ambiguity_threshold"):
            cfg.validate()

    def test_duplicate_model_tier_names(self):
        tiers = [
            ModelTier(name="gpt-4", cost_per_prompt_token=0.01, cost_per_completion_token=0.03, max_context_window=8192),
            ModelTier(name="gpt-4", cost_per_prompt_token=0.02, cost_per_completion_token=0.06, max_context_window=4096),
        ]
        cfg = BudgeteerConfig(model_tiers=tiers)
        with pytest.raises(ValueError, match="duplicate model tier name"):
            cfg.validate()

    def test_negative_cost_per_prompt_token_in_tier(self):
        """Negative prompt cost is caught at ModelTier construction."""
        with pytest.raises(ValueError, match="cost_per_prompt_token"):
            ModelTier(name="bad", cost_per_prompt_token=-0.01, cost_per_completion_token=0.03, max_context_window=8192)

    def test_zero_context_window_in_tier(self):
        """Zero context window is caught at ModelTier construction."""
        with pytest.raises(ValueError, match="max_context_window"):
            ModelTier(name="bad", cost_per_prompt_token=0.01, cost_per_completion_token=0.03, max_context_window=0)

    def test_multiple_errors_reported(self):
        """Multiple errors should all appear in the message."""
        cfg = BudgeteerConfig(
            default_max_tokens=-1,
            calibration_alpha=0,
            default_temperature=-1,
        )
        with pytest.raises(ValueError) as exc_info:
            cfg.validate()
        msg = str(exc_info.value)
        assert "default_max_tokens" in msg
        assert "calibration_alpha" in msg
        assert "default_temperature" in msg

    def test_valid_config_with_tiers(self):
        """A fully valid config with model tiers passes."""
        tiers = [
            ModelTier(name="gpt-4o-mini", cost_per_prompt_token=0.001, cost_per_completion_token=0.002, max_context_window=16384),
            ModelTier(name="gpt-4o", cost_per_prompt_token=0.01, cost_per_completion_token=0.03, max_context_window=128000),
        ]
        cfg = BudgeteerConfig(model_tiers=tiers)
        cfg.validate()  # should not raise


class TestConfigFromDictValidation:
    """from_dict triggers validation."""

    def test_from_dict_rejects_invalid(self):
        with pytest.raises(ValueError, match="default_max_tokens"):
            BudgeteerConfig.from_dict({"default_max_tokens": 0})

    def test_from_dict_accepts_valid(self):
        cfg = BudgeteerConfig.from_dict({"default_max_tokens": 512})
        assert cfg.default_max_tokens == 512


class TestConfigFromFileValidation:
    """from_file triggers validation."""

    def test_from_file_rejects_invalid_json(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text(json.dumps({"default_max_tokens": -5}))
        with pytest.raises(ValueError, match="default_max_tokens"):
            BudgeteerConfig.from_file(f)

    def test_from_file_accepts_valid_json(self, tmp_path):
        f = tmp_path / "good.json"
        f.write_text(json.dumps({"default_max_tokens": 2048}))
        cfg = BudgeteerConfig.from_file(f)
        assert cfg.default_max_tokens == 2048


# ------------------------------------------------------------------
# RunBudget validation
# ------------------------------------------------------------------


class TestRunBudgetValidation:
    """Tests for RunBudget.__post_init__."""

    def test_default_run_budget_valid(self):
        RunBudget()  # should not raise

    def test_none_caps_valid(self):
        RunBudget(hard_usd_cap=None, hard_token_cap=None)

    def test_positive_caps_valid(self):
        RunBudget(hard_usd_cap=1.0, hard_token_cap=1000, hard_latency_cap_ms=5000, max_tool_calls=10)

    def test_zero_usd_cap_rejected(self):
        with pytest.raises(ValueError, match="hard_usd_cap"):
            RunBudget(hard_usd_cap=0)

    def test_negative_usd_cap_rejected(self):
        with pytest.raises(ValueError, match="hard_usd_cap"):
            RunBudget(hard_usd_cap=-1.0)

    def test_zero_token_cap_rejected(self):
        with pytest.raises(ValueError, match="hard_token_cap"):
            RunBudget(hard_token_cap=0)

    def test_negative_latency_cap_rejected(self):
        with pytest.raises(ValueError, match="hard_latency_cap_ms"):
            RunBudget(hard_latency_cap_ms=-100)

    def test_zero_max_tool_calls_rejected(self):
        with pytest.raises(ValueError, match="max_tool_calls"):
            RunBudget(max_tool_calls=0)


# ------------------------------------------------------------------
# BudgetAccount validation
# ------------------------------------------------------------------


class TestBudgetAccountValidation:
    """Tests for BudgetAccount.__post_init__."""

    def test_valid_account(self):
        BudgetAccount(scope=BudgetScope.USER, scope_id="user-1", limit_usd_per_day=10.0)

    def test_empty_scope_id_rejected(self):
        with pytest.raises(ValueError, match="scope_id"):
            BudgetAccount(scope=BudgetScope.USER, scope_id="")

    def test_zero_usd_limit_rejected(self):
        with pytest.raises(ValueError, match="limit_usd_per_day"):
            BudgetAccount(scope=BudgetScope.USER, scope_id="u1", limit_usd_per_day=0)

    def test_negative_tokens_limit_rejected(self):
        with pytest.raises(ValueError, match="limit_tokens_per_day"):
            BudgetAccount(scope=BudgetScope.USER, scope_id="u1", limit_tokens_per_day=-1)

    def test_negative_runs_limit_rejected(self):
        with pytest.raises(ValueError, match="limit_runs_per_day"):
            BudgetAccount(scope=BudgetScope.USER, scope_id="u1", limit_runs_per_day=-1)

    def test_none_limits_valid(self):
        BudgetAccount(scope=BudgetScope.ORG, scope_id="org-1")


# ------------------------------------------------------------------
# ModelTier validation
# ------------------------------------------------------------------


class TestModelTierValidation:
    """Tests for ModelTier.__post_init__."""

    def test_valid_tier(self):
        ModelTier(name="gpt-4", cost_per_prompt_token=0.01, cost_per_completion_token=0.03, max_context_window=8192)

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError, match="name"):
            ModelTier(name="", cost_per_prompt_token=0.01, cost_per_completion_token=0.03, max_context_window=8192)

    def test_negative_prompt_cost_rejected(self):
        with pytest.raises(ValueError, match="cost_per_prompt_token"):
            ModelTier(name="m", cost_per_prompt_token=-0.01, cost_per_completion_token=0.03, max_context_window=8192)

    def test_negative_completion_cost_rejected(self):
        with pytest.raises(ValueError, match="cost_per_completion_token"):
            ModelTier(name="m", cost_per_prompt_token=0.01, cost_per_completion_token=-0.03, max_context_window=8192)

    def test_zero_context_window_rejected(self):
        with pytest.raises(ValueError, match="max_context_window"):
            ModelTier(name="m", cost_per_prompt_token=0.01, cost_per_completion_token=0.03, max_context_window=0)

    def test_zero_costs_valid(self):
        """Zero-cost models (e.g. local) are valid."""
        ModelTier(name="local", cost_per_prompt_token=0, cost_per_completion_token=0, max_context_window=4096)


# ------------------------------------------------------------------
# SDK validation
# ------------------------------------------------------------------


class TestSDKValidation:
    """Tests for validation in Budgeteer SDK."""

    def test_sdk_validates_config(self, tmp_path):
        from budgeteer.sdk import Budgeteer

        cfg = BudgeteerConfig(
            default_max_tokens=-1,
            storage_path=str(tmp_path / "test.db"),
        )
        with pytest.raises(ValueError, match="default_max_tokens"):
            Budgeteer(config=cfg)

    def test_sdk_no_config_uses_default(self, tmp_path):
        """Passing None config should work (default is valid)."""
        from budgeteer.sdk import Budgeteer

        b = Budgeteer(config=BudgeteerConfig(storage_path=str(tmp_path / "test.db")))
        b.close()

    def test_execute_step_empty_messages(self, tmp_path):
        from budgeteer.llm_client import LLMClient
        from budgeteer.sdk import Budgeteer

        client = LLMClient(call_fn=lambda **kw: {}, model_tiers=[])
        b = Budgeteer(
            config=BudgeteerConfig(storage_path=str(tmp_path / "test.db")),
            llm_client=client,
        )
        run = b.start_run()
        with pytest.raises(ValueError, match="messages must be a non-empty list"):
            b.execute_step(run.run_id, messages=[])
        b.close()
