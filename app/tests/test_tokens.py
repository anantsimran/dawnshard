import pytest
import torch
from training.dataload.tokens import TokenWindows, lm_collate, write_tokens


def test_windows_shift_by_one_and_round_trip_large_ids(tmp_path):
    ids = list(range(70000, 70021))
    write_tokens(path=tmp_path / "t.bin", ids=ids)
    windows = TokenWindows(path=tmp_path / "t.bin", seq_len=5)
    assert len(windows) == 4
    x, y = lm_collate(rows=[windows[0], windows[1]])
    assert x.tolist() == [ids[0:5], ids[5:10]]
    assert y.tolist() == ids[1:6] + ids[6:11]
    assert x.dtype == torch.int64


def test_ids_outside_uint32_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_tokens(path=tmp_path / "t.bin", ids=[2**32])
