import pandas as pd
import pytest
from src.visualization.qualitative import (plot_bar_chart, plot_pie_chart, plot_contingency_heatmap,
                                      plot_stacked_bar, plot_countplot_with_hue, plot_mosaic)

@pytest.fixture
def sample_qual_df(tmpdir):
    df = pd.DataFrame({'Cat': ['A', 'B', 'A', 'C'], 'Group': ['X', 'Y', 'X', 'Y']})
    return df, str(tmpdir)

def test_plot_bar_chart(sample_qual_df):
    df, figures_path = sample_qual_df
    plot_bar_chart(df, 'Cat', figures_path)

def test_plot_pie_chart(sample_qual_df):
    df, figures_path = sample_qual_df
    plot_pie_chart(df, 'Cat', figures_path)

def test_plot_contingency_heatmap(sample_qual_df):
    df, figures_path = sample_qual_df
    plot_contingency_heatmap(df, 'Cat', 'Group', figures_path)

def test_plot_stacked_bar(sample_qual_df):
    df, figures_path = sample_qual_df
    plot_stacked_bar(df, 'Cat', 'Group', figures_path)

def test_plot_countplot_with_hue(sample_qual_df):
    df, figures_path = sample_qual_df
    plot_countplot_with_hue(df, 'Cat', 'Group', figures_path)

def test_plot_mosaic(sample_qual_df):
    df, figures_path = sample_qual_df
    plot_mosaic(df, ['Cat', 'Group'], figures_path)