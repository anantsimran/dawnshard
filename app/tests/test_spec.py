import json

import pytest
from training.launch.spec import JobSpec

DIGEST = "ghcr.io/x/dawnshard@sha256:" + "a" * 64


def _spec(**kwargs) -> JobSpec:
    return JobSpec(**{"name": "job", "image": DIGEST, "module": "m", **kwargs})


def test_valid_spec_passes():
    _spec(nodes=2, gpus_per_node=1, data_uri="s3://b/d", checkpoint_uri="gs://b/c").validate()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "Bad_Name"},
        {"name": "x" * 41},
        {"image": "ghcr.io/x/dawnshard:latest"},
        {"module": ""},
        {"nodes": 0},
        {"gpus_per_node": -1},
        {"data_uri": "http://example.com/data"},
    ],
)
def test_invalid_fields_are_rejected(kwargs):
    with pytest.raises(ValueError):
        _spec(**kwargs).validate()


def test_tag_is_allowed_only_without_digest_requirement():
    _spec(image="local").validate(require_digest=False)


def test_world_size_uses_gpus_when_present():
    assert _spec(nodes=2, procs_per_node=3).world_size == 6
    assert _spec(nodes=2, gpus_per_node=4, procs_per_node=3).world_size == 8


def test_from_json(tmp_path):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps({"name": "job", "image": DIGEST, "module": "m", "args": ["--a", "1"]}))
    assert JobSpec.from_json(path=path).args == ("--a", "1")
