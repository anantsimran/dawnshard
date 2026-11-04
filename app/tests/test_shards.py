import json

import numpy as np
from training.dataload.shards import pack, write_manifest, write_shards


def test_documents_are_split_across_shards_with_eos_and_a_start_index(tmp_path):
    documents = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    entries = write_shards(documents=documents, eos_id=0, out_dir=tmp_path, prefix="d", tokens_per_shard=5)
    stream = np.concatenate([np.fromfile(tmp_path / e["name"], dtype=np.uint32) for e in entries])
    assert stream.tolist() == [1, 2, 3, 0, 4, 5, 0, 6, 7, 8, 9, 0]
    assert [e["num_tokens"] for e in entries] == [5, 5, 2]
    assert np.fromfile(tmp_path / "d_00000.idx", dtype=np.uint64).tolist() == [0, 4]
    manifest = write_manifest(out_dir=tmp_path)
    assert manifest["num_tokens"] == 12
    assert json.loads((tmp_path / "manifest.json").read_text())["shards"][0]["name"] == "d_00000.bin"


def test_pack_yields_full_windows_only_and_keeps_eos():
    windows = list(pack(stream=[1, 2, 0, 3, 4, 5, 0, 6, 7], seq_len=3))
    assert [w.tolist() for w in windows] == [[1, 2, 0, 3], [3, 4, 5, 0]]
    assert all(len(w) == 4 for w in windows)
