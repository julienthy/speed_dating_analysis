import pandas as pd
import pytest
from src.visualization.exploratory import explore_quantitative_data, explore_qualitative_data, explore_mixed_data

@pytest.fixture
def sample_mixed_df(tmpdir):
    df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
    return df, str(tmpdir)

def test_explore_quantitative_data(sample_mixed_df):
    df, figures_path = sample_mixed_df
    explore_quantitative_data(df, figures_path)

def test_explore_qualitative_data(sample_mixed_df):
    df, figures_path = sample_mixed_df
    explore_qualitative_data(df, figures_path)

def test_explore_mixed_data(sample_mixed_df):
    df, figures_path = sample_mixed_df
    explore_mixed_data(df, figures_path)