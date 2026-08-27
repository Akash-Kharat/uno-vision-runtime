import pytest
from app.services.execution_provider_manager import ExecutionProviderManager
from app.core.exceptions import AppError

def test_execution_provider_discovery():
    manager = ExecutionProviderManager()
    
    # Assert CPU is always at least present since ORT guarantees it
    assert "CPUExecutionProvider" in manager.available_providers
    
    # Assert get_status returns the correct active provider
    status = manager.get_status()
    assert status.active_provider == manager.active_provider
    assert any(p.name == "CPUExecutionProvider" for p in status.providers)

def test_execution_provider_switching():
    manager = ExecutionProviderManager()
    
    # Try to switch to CPU
    manager.set_active_provider("CPUExecutionProvider")
    assert manager.active_provider == "CPUExecutionProvider"
    
    # Try to switch to an invalid one
    with pytest.raises(AppError) as exc:
        manager.set_active_provider("UnknownExecutionProvider")
    assert exc.value.code == "PROVIDER_UNAVAILABLE"
