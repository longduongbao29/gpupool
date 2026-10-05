import pytest

from gpupool.scheduler import scoring


@pytest.fixture(autouse=True)
def _default_speed_model():
    """The decode-speed model is process-wide (the coordinator learns it): every test starts
    from the default and cannot leak a learned one into the next."""
    scoring.set_speed_model(scoring.ETA, scoring.HOP_S)
    yield
    scoring.set_speed_model(scoring.ETA, scoring.HOP_S)
