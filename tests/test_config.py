from efforttts import config


def test_data_root_respects_env(tmp_path, monkeypatch):
    monkeypatch.setenv("EFFORT_DATA_DIR", str(tmp_path))
    assert config.data_root() == tmp_path
    assert config.raw_dir().parent == tmp_path
    assert config.raw_dir().is_dir()
