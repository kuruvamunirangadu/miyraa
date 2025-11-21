import scripts.prepare_data as prep


def test_synthetic_dataset_schema():
    data = prep.synthetic_dataset(5)
    assert len(data) == 5
    rec = data[0]
    assert "text" in rec and isinstance(rec["text"], str)
    assert "labels" in rec and isinstance(rec["labels"], list)
    assert "vad" in rec and isinstance(rec["vad"], dict)
