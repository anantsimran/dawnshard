import pytest


@pytest.fixture(autouse=True)
def checkpoints_in_tmp_path(monkeypatch, tmp_path):
    # fit checkpoints after every validation pass; keep those files out of app/history.
    monkeypatch.setattr("training.train.train_loop.CHECKPOINTS_PATH", tmp_path / "checkpoints")
