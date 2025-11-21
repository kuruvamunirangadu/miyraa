import os
import pytest


def _has_datasets_and_data(path="data/processed/bootstrap"):
    try:
        from datasets import load_from_disk  # noqa: F401
    except Exception:
        return False
    return os.path.exists(path)


@pytest.mark.skipif(
    not _has_datasets_and_data(), reason="datasets/bootstrap not available"
)
def test_bootstrap_schema_fields():
    from datasets import load_from_disk

    ds = load_from_disk("data/processed/bootstrap")
    if "train" in ds:
        d = ds["train"]
    else:
        d = ds

    assert len(d) > 0
    rec = d[0]
    # Basic schema checks
    assert "text" in rec and isinstance(rec["text"], str)
    assert "labels" in rec and isinstance(rec["labels"], (list, tuple))
    # labels should be a list of ints or 0/1 values
    assert all(isinstance(x, (int,)) for x in rec["labels"])
    assert "vad" in rec and isinstance(rec["vad"], dict)
    for key in ("v", "a", "d"):
        assert key in rec["vad"]
        assert isinstance(rec["vad"][key], float)
