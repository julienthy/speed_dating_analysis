import pandas as pd
import pytest
from src.visualization.quantitative import (plot_correlation_matrix, plot_feature_distributions, 
                                       plot_feature_target_relations, analyse_multivariee_selective,
                                       plot_boxplots, diagramme_dispersion_cibles, plot_boxenplot)

@pytest.fixture
def sample_quant_df(tmpdir):
    df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [4, 3, 2, 1]})
    return df, str(tmpdir)

@pytest.fixture
def sample_target_df():
    return pd.DataFrame({'Target': [0, 1, 0, 1]})

def test_plot_correlation_matrix(sample_quant_df):
    df, figures_path = sample_quant_df
    corr_matrix = plot_correlation_matrix(df, figures_path)
    assert isinstance(corr_matrix, pd.DataFrame)

def test_plot_feature_distributions(sample_quant_df):
    df, figures_path = sample_quant_df
    plot_feature_distributions(df, figures_path)

def test_plot_feature_target_relations(sample_quant_df, sample_target_df):
    df, figures_path = sample_quant_df
    plot_feature_target_relations(df, sample_target_df, figures_path)

def test_analyse_multivariee_selective(sample_quant_df, sample_target_df):
    df, figures_path = sample_quant_df
    combined_df = pd.concat([df, sample_target_df], axis=1)
    corr_matrix = combined_df.corr()
    analyse_multivariee_selective(combined_df, sample_target_df, corr_matrix, figures_path)

def test_plot_boxplots(sample_quant_df):
    df, figures_path = sample_quant_df
    plot_boxplots(df, figures_path)

def test_diagramme_dispersion_cibles(sample_quant_df, sample_target_df):
    df, figures_path = sample_quant_df
    corr_matrix = df.corr()
    diagramme_dispersion_cibles(df, sample_target_df, corr_matrix, figures_path)

def test_plot_boxenplot(sample_quant_df):
    df, figures_path = sample_quant_df
    plot_boxenplot(df, figures_path)