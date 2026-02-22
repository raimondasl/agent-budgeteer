import pytest

from budgeteer.config import BudgeteerConfig
from budgeteer.telemetry import TelemetryStore


@pytest.fixture()
def tmp_db(tmp_path):
    """Path to a temporary SQLite database."""
    return str(tmp_path / "test.db")


@pytest.fixture()
def config(tmp_db):
    """BudgeteerConfig pointing at a temporary database."""
    return BudgeteerConfig(storage_path=tmp_db)


@pytest.fixture()
def telemetry(tmp_db):
    """TelemetryStore backed by a temporary database."""
    store = TelemetryStore(tmp_db)
    yield store
    store.close()
