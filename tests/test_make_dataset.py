import pandas as pd
import pytest
from src.data.make_dataset import *

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "iid": [1, 1, 2, 2, 3],
        "match": [1, 0, 1, 1, 0],
        "age": [25, 30, 35, 40, 45]
    })

def test_add_aggregated_column(sample_df):
    df = add_aggregated_column(sample_df, "iid", "match", "sum", "total_matches")
    assert "total_matches" in df.columns
    assert df["total_matches"].iloc[0] == 1  # Pour iid=1, sum(match)=1
    assert df["total_matches"].iloc[2] == 2  # Pour iid=2, sum(match)=2

def test_drop_rows_by_condition(sample_df):
    df = drop_rows_by_condition(sample_df, lambda row: row["age"] > 35)
    assert len(df) == 3  # Les lignes avec age=40 et 45 sont supprimées

def test_categorize_column(sample_df):
    df = categorize_column(sample_df, "age", bins=2, labels=["jeune", "vieux"])
    assert "age_cat" in df.columns
    assert df["age_cat"].iloc[0] == "jeune"
    assert df["age_cat"].iloc[4] == "vieux"

def test_merge_dataframes():
    # Création de DataFrames de test
    df1 = pd.DataFrame({"id": [1, 2, 3], "value1": [10, 20, 30]})
    df2 = pd.DataFrame({"id": [2, 3, 4], "value2": [200, 300, 400]})
    
    # Test fusion "inner"
    merged_inner = merge_dataframes(df1, df2, "id", "inner")
    assert len(merged_inner) == 2  # Seuls les id 2 et 3 sont communs
    assert list(merged_inner.columns) == ["id", "value1", "value2"]
    
    # Test fusion "left"
    merged_left = merge_dataframes(df1, df2, "id", "left")
    assert len(merged_left) == 3  # Tous les id de df1 sont conservés
    assert merged_left["value2"].isna().sum() == 1  # Une valeur NaN pour id=1

def test_rename_columns():
    df = pd.DataFrame({"old_name": [1, 2, 3]})
    df_renamed = rename_columns(df, {"old_name": "new_name"})
    assert "new_name" in df_renamed.columns
    assert "old_name" not in df_renamed.columns

def test_select_columns():
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4], "col3": [5, 6]})
    df_selected = select_columns(df, ["col1", "col3"])
    assert list(df_selected.columns) == ["col1", "col3"]
    assert len(df_selected) == 2

def test_convert_types():
    df = pd.DataFrame({"num": [1.0, 2.0], "cat": ["a", "b"]})
    df_converted = convert_types(df, {"num": "int", "cat": "category"})
    assert df_converted["num"].dtype == "int"
    assert df_converted["cat"].dtype.name == "category"