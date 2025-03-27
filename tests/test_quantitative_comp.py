import pandas as pd
import pytest
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.visualization.quantitative import (
    plot_temporal_histograms,
    plot_temporal_histograms2,
    plot_scatter_comparison,
    plot_violin_comparison,
    plot_boxplots_by_decision,
    plot_correlation_heatmap
)

# Fixture pour un DataFrame de test simulé
@pytest.fixture
def sample_df(tmp_path):
    # Création d'un DataFrame avec des données similaires à speed_dating
    data = pd.DataFrame({
        "gender": [0, 1, 0, 1],
        "attr1_1": [20, 30, 25, 35],
        "attr1_2": [22, 32, 27, 37],
        "attr1_3": [25, 35, 30, 40],
        "attr": [6, 7, 5, 8],
        "sinc": [7, 6, 8, 5],
        "intel": [8, 9, 7, 6],
        "attr3_1": [7, 8, 6, 9],
        "attr_o": [6, 7, 5, 8],
        "dec": [0, 1, 0, 1]
    })
    figures_path = tmp_path / "figures"
    figures_path.mkdir()
    return data, str(figures_path)

def test_plot_temporal_histograms(sample_df):
    df, figures_path = sample_df
    plot_temporal_histograms(df, "attr1_", ["1", "2", "3"], "gender", save_path=figures_path)
    output_file = Path(figures_path) / "Evolution_attr1__by_gender.png"
    assert output_file.exists(), f"Le fichier {output_file} n'a pas été créé."

def test_plot_temporal_histograms2(sample_df):
    df, figures_path = sample_df
    plot_temporal_histograms2(df, "attr1_", ["1", "2", "3"], "gender", save_path=figures_path)
    for time in ["1", "2", "3"]:
        output_file = Path(figures_path) / f"temporal_histogram_attr1__time_{time}_by_gender.png"
        assert output_file.exists(), f"Le fichier {output_file} n'a pas été créé."

def test_plot_scatter_comparison(sample_df):
    df, figures_path = sample_df
    plot_scatter_comparison(df, "attr1_1", "attr", "gender", save_path=figures_path)
    output_file = Path(figures_path) / "Comparaison_attr1_1_attr.png"
    assert output_file.exists(), f"Le fichier {output_file} n'a pas été créé."

def test_plot_violin_comparison(sample_df):
    df, figures_path = sample_df
    plot_violin_comparison(df, ["attr3_1", "attr_o"], "gender", save_path=figures_path)
    output_file = Path(figures_path) / "Distributions_attr3_1_attr_o_by_gender.png"
    assert output_file.exists(), f"Le fichier {output_file} n'a pas été créé."
    
    # Test avec une liste invalide de variables
    with pytest.raises(ValueError, match="vars_to_compare doit contenir exactement deux variables"):
        plot_violin_comparison(df, ["attr3_1"], "gender", save_path=figures_path)

def test_plot_boxplots_by_decision(sample_df):
    df, figures_path = sample_df
    plot_boxplots_by_decision(df, ["attr", "sinc", "intel"], "dec", save_path=figures_path)
    output_file = Path(figures_path) / "Boxplots_by_dec.png"
    assert output_file.exists(), f"Le fichier {output_file} n'a pas été créé."
    
    # Test avec une seule variable
    plot_boxplots_by_decision(df, ["attr"], "dec", save_path=figures_path)
    assert output_file.exists(), f"Le fichier {output_file} n'a pas été créé après une seule variable."

def test_plot_correlation_heatmap(sample_df):
    df, figures_path = sample_df
    plot_correlation_heatmap(df, ["attr", "sinc", "intel", "dec"], save_path=figures_path)
    output_file = Path(figures_path) / "Heatmap_correlation.png"
    assert output_file.exists(), f"Le fichier {output_file} n'a pas été créé."