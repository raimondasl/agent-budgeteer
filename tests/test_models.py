"""Tests for budgeteer.models."""

from budgeteer.models import (
    BudgetAccount,
    BudgetScope,
    LLMResponse,
    ModelTier,
    RunBudget,
    RunRecord,
    StepContext,
    StepDecision,
    StepMetrics,
    StepRecord,
    ToolRecord,
    ToolResult,
)


class TestBudgetScope:
    def test_values(self):
        assert BudgetScope.USER == "user"
        assert BudgetScope.ORG == "org"
        assert BudgetScope.PROJECT == "project"

    def test_from_string(self):
        assert BudgetScope("user") is BudgetScope.USER


class TestBudgetAccount:
    def test_defaults(self):
        acct = BudgetAccount(scope=BudgetScope.USER, scope_id="u1")
        assert acct.limit_usd_per_day is None
        assert acct.limit_tokens_per_day is None
        assert acct.limit_runs_per_day is None

    def test_custom(self):
        acct = BudgetAccount(
            scope=BudgetScope.ORG,
            scope_id="org1",
            limit_usd_per_day=100.0,
            limit_tokens_per_day=1_000_000,
            limit_runs_per_day=50,
        )
        assert acct.scope == BudgetScope.ORG
        assert acct.limit_usd_per_day == 100.0


class TestRunBudget:
    def test_auto_id(self):
        b1 = RunBudget()
        b2 = RunBudget()
        assert b1.run_id != b2.run_id

    def test_custom(self):
        b = RunBudget(run_id="r1", hard_usd_cap=5.0, hard_token_cap=10000)
        assert b.run_id == "r1"
        assert b.hard_usd_cap == 5.0
        assert b.hard_latency_cap_ms is None


class TestModelTier:
    def test_creation(self):
        t = ModelTier(
            name="gpt-4o-mini",
            cost_per_prompt_token=0.00015,
            cost_per_completion_token=0.0006,
            max_context_window=128000,
        )
        assert t.tier == "standard"
        assert t.name == "gpt-4o-mini"


class TestStepContext:
    def test_defaults(self):
        ctx = StepContext(run_id="r1")
        assert ctx.run_id == "r1"
        assert ctx.step_id  # auto-generated
        assert ctx.current_run_cost_usd == 0.0
        assert ctx.metadata == {}

    def test_auto_step_id_unique(self):
        c1 = StepContext(run_id="r1")
        c2 = StepContext(run_id="r1")
        assert c1.step_id != c2.step_id


class TestStepDecision:
    def test_defaults(self):
        d = StepDecision(model="gpt-4o")
        assert d.max_tokens == 1024
        assert d.temperature == 0.7
        assert d.degrade_level == 0
        assert d.degrade_reason is None
        assert d.retrieval_enabled is True


class TestStepMetrics:
    def test_defaults(self):
        m = StepMetrics()
        assert m.prompt_tokens == 0
        assert m.cost_usd == 0.0
        assert m.success is None

    def test_custom(self):
        m = StepMetrics(prompt_tokens=100, completion_tokens=50, cost_usd=0.01, success=True)
        assert m.prompt_tokens == 100
        assert m.success is True


class TestRunRecord:
    def test_auto_time(self):
        r = RunRecord(run_id="r1")
        assert r.start_time > 0
        assert r.end_time is None
        assert r.success is None


class TestStepRecord:
    def test_creation(self):
        d = StepDecision(model="gpt-4o")
        m = StepMetrics(prompt_tokens=10)
        rec = StepRecord(run_id="r1", step_id="s1", decision=d, actual=m)
        assert rec.decision.model == "gpt-4o"
        assert rec.actual.prompt_tokens == 10
        assert rec.predicted is None


class TestToolRecord:
    def test_creation(self):
        rec = ToolRecord(run_id="r1", step_id="s1", tool_name="search", duration_ms=150.0)
        assert rec.success is True
        assert rec.error is None


class TestLLMResponse:
    def test_creation(self):
        resp = LLMResponse(content="hello", model="gpt-4o", prompt_tokens=5, completion_tokens=1)
        assert resp.latency_ms == 0.0
        assert resp.raw_response is None


class TestToolResult:
    def test_success(self):
        res = ToolResult(tool_name="calc", output=42, success=True, duration_ms=5.0)
        assert res.output == 42

    def test_failure(self):
        res = ToolResult(tool_name="calc", success=False, error="boom")
        assert res.error == "boom"
