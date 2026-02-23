"""Tests for calibrator persistence (Milestone 13)."""

from __future__ import annotations

import json

import pytest

from budgeteer.calibrator import Calibrator, CorrectionFactors
from budgeteer.config import BudgeteerConfig
from budgeteer.models import StepMetrics
from budgeteer.sdk import Budgeteer


class TestCalibratorSaveLoad:
    """Save/load round-trip tests."""

    def test_save_load_round_trip(self, tmp_path):
        path = tmp_path / "cal.json"
        cal = Calibrator(alpha=0.3)
        # Feed some data
        cal.update("gpt-4", StepMetrics(prompt_tokens=100, cost_usd=0.01), StepMetrics(prompt_tokens=120, cost_usd=0.012))
        cal.save(path)

        cal2 = Calibrator(alpha=0.3)
        cal2.load(path)
        f1 = cal.get_factors("gpt-4")
        f2 = cal2.get_factors("gpt-4")
        assert f2.prompt_tokens == f1.prompt_tokens
        assert f2.cost_usd == f1.cost_usd
        assert f2.sample_count == f1.sample_count

    def test_save_creates_file(self, tmp_path):
        path = tmp_path / "cal.json"
        cal = Calibrator()
        cal.save(path)
        assert path.exists()

    def test_load_nonexistent_file(self, tmp_path):
        """Loading from a file that doesn't exist should be a no-op."""
        cal = Calibrator()
        cal.load(tmp_path / "does_not_exist.json")
        assert cal.models == []

    def test_load_merges_with_existing(self, tmp_path):
        path = tmp_path / "cal.json"
        cal1 = Calibrator(alpha=0.5)
        cal1.update("model-a", StepMetrics(prompt_tokens=100, cost_usd=0.01), StepMetrics(prompt_tokens=150, cost_usd=0.015))
        cal1.save(path)

        cal2 = Calibrator(alpha=0.5)
        cal2.update("model-b", StepMetrics(prompt_tokens=200, cost_usd=0.02), StepMetrics(prompt_tokens=220, cost_usd=0.022))
        cal2.load(path)

        assert "model-a" in cal2.models
        assert "model-b" in cal2.models

    def test_save_overwrites_existing(self, tmp_path):
        path = tmp_path / "cal.json"
        cal = Calibrator()
        cal.update("m1", StepMetrics(prompt_tokens=100, cost_usd=0.01), StepMetrics(prompt_tokens=120, cost_usd=0.012))
        cal.save(path)

        cal.reset()
        cal.update("m2", StepMetrics(prompt_tokens=100, cost_usd=0.01), StepMetrics(prompt_tokens=120, cost_usd=0.012))
        cal.save(path)

        data = json.loads(path.read_text())
        assert "m2" in data
        assert "m1" not in data

    def test_empty_calibrator_saves_empty_object(self, tmp_path):
        path = tmp_path / "cal.json"
        cal = Calibrator()
        cal.save(path)
        data = json.loads(path.read_text())
        assert data == {}

    def test_load_preserves_all_factor_fields(self, tmp_path):
        path = tmp_path / "cal.json"
        cal = Calibrator(alpha=0.5)
        cal.update("m", StepMetrics(prompt_tokens=100, completion_tokens=50, cost_usd=0.01, latency_ms=200),
                   StepMetrics(prompt_tokens=130, completion_tokens=60, cost_usd=0.013, latency_ms=250))
        cal.save(path)

        cal2 = Calibrator(alpha=0.5)
        cal2.load(path)
        f = cal2.get_factors("m")
        assert f.prompt_tokens != 1.0  # was updated
        assert f.completion_tokens != 1.0
        assert f.cost_usd != 1.0
        assert f.latency_ms != 1.0
        assert f.sample_count == 1

    def test_multiple_models_round_trip(self, tmp_path):
        path = tmp_path / "cal.json"
        cal = Calibrator()
        for model in ["gpt-4", "gpt-3.5", "claude-3"]:
            cal.update(model, StepMetrics(prompt_tokens=100, cost_usd=0.01), StepMetrics(prompt_tokens=120, cost_usd=0.012))
        cal.save(path)

        cal2 = Calibrator()
        cal2.load(path)
        assert set(cal2.models) == {"gpt-4", "gpt-3.5", "claude-3"}

    def test_corrupt_file_raises_value_error(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not valid json {{{", encoding="utf-8")
        cal = Calibrator()
        with pytest.raises(ValueError, match="Corrupt"):
            cal.load(path)

    def test_non_dict_file_raises_value_error(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        cal = Calibrator()
        with pytest.raises(ValueError, match="JSON object"):
            cal.load(path)


class TestCalibratorFromFile:
    """Tests for Calibrator.from_file() constructor."""

    def test_from_file_loads_factors(self, tmp_path):
        path = tmp_path / "cal.json"
        cal = Calibrator(alpha=0.4)
        cal.update("m", StepMetrics(prompt_tokens=100, cost_usd=0.01), StepMetrics(prompt_tokens=120, cost_usd=0.012))
        cal.save(path)

        cal2 = Calibrator.from_file(path, alpha=0.4)
        assert "m" in cal2.models
        assert cal2.get_factors("m").sample_count == 1

    def test_from_file_nonexistent(self, tmp_path):
        cal = Calibrator.from_file(tmp_path / "nope.json")
        assert cal.models == []

    def test_from_file_uses_alpha(self, tmp_path):
        path = tmp_path / "cal.json"
        Calibrator().save(path)  # empty file
        cal = Calibrator.from_file(path, alpha=0.9)
        # Verify alpha is used by checking update behavior
        cal.update("m", StepMetrics(prompt_tokens=100), StepMetrics(prompt_tokens=200))
        # With alpha=0.9, new factor should be heavily weighted toward 2.0
        f = cal.get_factors("m")
        assert f.prompt_tokens > 1.5  # should be close to 0.1*1.0 + 0.9*2.0 = 1.9


class TestSDKCalibratorPersistence:
    """Tests for SDK integration of calibrator persistence."""

    def test_close_saves_calibrator(self, tmp_path):
        path = tmp_path / "cal_state.json"
        cfg = BudgeteerConfig(
            storage_path=str(tmp_path / "tel.db"),
            calibration_state_path=str(path),
        )
        b = Budgeteer(config=cfg)
        # Manually update calibrator
        b.calibrator.update("m", StepMetrics(prompt_tokens=100, cost_usd=0.01), StepMetrics(prompt_tokens=120, cost_usd=0.012))
        b.close()
        assert path.exists()
        data = json.loads(path.read_text())
        assert "m" in data

    def test_init_loads_calibrator(self, tmp_path):
        path = tmp_path / "cal_state.json"
        # First: save state
        cfg = BudgeteerConfig(
            storage_path=str(tmp_path / "tel.db"),
            calibration_state_path=str(path),
        )
        b = Budgeteer(config=cfg)
        b.calibrator.update("m", StepMetrics(prompt_tokens=100, cost_usd=0.01), StepMetrics(prompt_tokens=120, cost_usd=0.012))
        b.close()

        # Second: load state
        cfg2 = BudgeteerConfig(
            storage_path=str(tmp_path / "tel2.db"),
            calibration_state_path=str(path),
        )
        b2 = Budgeteer(config=cfg2)
        assert "m" in b2.calibrator.models
        assert b2.calibrator.get_factors("m").sample_count == 1
        b2.close()

    def test_factors_accumulate_across_restarts(self, tmp_path):
        path = tmp_path / "cal_state.json"

        # Session 1: one update
        cfg = BudgeteerConfig(storage_path=str(tmp_path / "t1.db"), calibration_state_path=str(path))
        b = Budgeteer(config=cfg)
        b.calibrator.update("m", StepMetrics(prompt_tokens=100, cost_usd=0.01), StepMetrics(prompt_tokens=120, cost_usd=0.012))
        b.close()

        # Session 2: another update
        cfg2 = BudgeteerConfig(storage_path=str(tmp_path / "t2.db"), calibration_state_path=str(path))
        b2 = Budgeteer(config=cfg2)
        assert b2.calibrator.get_factors("m").sample_count == 1
        b2.calibrator.update("m", StepMetrics(prompt_tokens=100, cost_usd=0.01), StepMetrics(prompt_tokens=130, cost_usd=0.013))
        b2.close()

        # Session 3: verify accumulated
        cfg3 = BudgeteerConfig(storage_path=str(tmp_path / "t3.db"), calibration_state_path=str(path))
        b3 = Budgeteer(config=cfg3)
        assert b3.calibrator.get_factors("m").sample_count == 2
        b3.close()

    def test_no_path_configured_no_save(self, tmp_path):
        """Without calibration_state_path, close() doesn't crash."""
        cfg = BudgeteerConfig(storage_path=str(tmp_path / "tel.db"))
        b = Budgeteer(config=cfg)
        b.close()  # should not raise

    def test_context_manager_saves(self, tmp_path):
        path = tmp_path / "cal_state.json"
        cfg = BudgeteerConfig(storage_path=str(tmp_path / "tel.db"), calibration_state_path=str(path))
        with Budgeteer(config=cfg) as b:
            b.calibrator.update("m", StepMetrics(prompt_tokens=100, cost_usd=0.01), StepMetrics(prompt_tokens=120, cost_usd=0.012))
        assert path.exists()

    def test_first_run_no_file(self, tmp_path):
        """First run with no existing file should not error."""
        path = tmp_path / "cal_state.json"
        cfg = BudgeteerConfig(storage_path=str(tmp_path / "tel.db"), calibration_state_path=str(path))
        b = Budgeteer(config=cfg)
        assert b.calibrator.models == []
        b.close()
