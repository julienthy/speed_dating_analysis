import os
import pytest
import pandas as pd
import yaml
import json
from src.utils.io import load_config, load_data, save_data, save_model, load_model, save_metrics, ensure_dir

# Fixture pour un répertoire temporaire
@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path

# Tests pour load_config
def test_load_config(tmp_dir):
    config = {"key": "value"}
    config_path = tmp_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    result = load_config(str(config_path))
    assert result == config, "La configuration n'est pas chargée correctement"

def test_load_config_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("non_existent.yaml")

# Tests pour load_data
def test_load_data_csv(tmp_dir):
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    file_path = tmp_dir / "data.csv"
    df.to_csv(file_path, index=False)
    result = load_data(str(file_path))
    pd.testing.assert_frame_equal(result, df)

def test_load_data_excel(tmp_dir):
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    file_path = tmp_dir / "data.xlsx"
    df.to_excel(file_path, index=False)
    result = load_data(str(file_path))
    pd.testing.assert_frame_equal(result, df)

def test_load_data_not_found():
    with pytest.raises(FileNotFoundError):
        load_data("non_existent.csv")

def test_load_data_unsupported_format(tmp_dir):
    file_path = tmp_dir / "data.txt"
    with open(file_path, 'w') as f:
        f.write("test content")
    with pytest.raises(ValueError):
        load_data(str(file_path))

# Tests pour save_data
def test_save_data_csv(tmp_dir):
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    file_path = tmp_dir / "output.csv"
    save_data(df, str(file_path))
    assert os.path.exists(file_path), "Le fichier CSV n'a pas été créé"
    loaded_df = pd.read_csv(file_path)
    pd.testing.assert_frame_equal(loaded_df, df)

def test_save_data_excel(tmp_dir):
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    file_path = tmp_dir / "output.xlsx"
    save_data(df, str(file_path))
    assert os.path.exists(file_path), "Le fichier Excel n'a pas été créé"

def test_save_data_unknown_extension(tmp_dir):
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    file_path = tmp_dir / "output.txt"
    save_data(df, str(file_path))
    assert os.path.exists(file_path), "Le fichier n'a pas été créé avec une extension inconnue"
    loaded_df = pd.read_csv(file_path)
    pd.testing.assert_frame_equal(loaded_df, df)

# Tests pour save_model et load_model
def test_save_and_load_model(tmp_dir):
    model = {"test": True}
    model_path = tmp_dir / "model.pkl"
    save_model(model, str(model_path))
    assert os.path.exists(model_path), "Le modèle n'a pas été sauvegardé"
    loaded_model = load_model(str(model_path))
    assert loaded_model == model, "Le modèle chargé ne correspond pas"

def test_load_model_not_found():
    with pytest.raises(FileNotFoundError):
        load_model("non_existent.pkl")

# Tests pour save_metrics
def test_save_metrics(tmp_dir):
    metrics = {"acc": 0.9}
    metrics_path = tmp_dir / "metrics.json"
    save_metrics(metrics, str(metrics_path))
    assert os.path.exists(metrics_path), "Les métriques n'ont pas été sauvegardées"
    with open(metrics_path, 'r') as f:
        loaded_metrics = json.load(f)
    assert loaded_metrics == metrics, "Les métriques chargées ne correspondent pas"

# Tests pour ensure_dir
def test_ensure_dir(tmp_dir):
    new_dir = tmp_dir / "new_dir"
    ensure_dir(str(new_dir))
    assert os.path.exists(new_dir), "Le répertoire n'a pas été créé"