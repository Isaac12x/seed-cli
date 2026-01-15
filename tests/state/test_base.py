import pytest
from seed_cli.state.base import StateBackend


def test_state_backend_is_abstract():
    with pytest.raises(TypeError):
        StateBackend()  # cannot instantiate
