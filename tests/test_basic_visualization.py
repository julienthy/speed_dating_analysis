import pandas as pd
import pytest
import matplotlib.pyplot as plt
from src.utils.basic_visualization import display_head, display_info, display_description, plot_missing_values

# Fixture pour un DataFrame de test
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [5, 4, 3, 2, 1],
        'C': [1.1, 2.2, 3.3, 4.4, 5.5],
        'D': ['a', 'b', 'c', 'd', 'e']
    })

# Test pour display_head
def test_display_head(sample_df, capsys):
    display_head(sample_df, n=3)
    captured = capsys.readouterr()
    assert "A  B    C  D" in captured.out
    assert "0  1  5  1.1  a" in captured.out
    assert "1  2  4  2.2  b" in captured.out
    assert "2  3  3  3.3  c" in captured.out

# Test pour display_info
def test_display_info(sample_df, capsys):
    display_info(sample_df)
    captured = capsys.readouterr()
    assert "RangeIndex: 5 entries, 0 to 4" in captured.out
    assert "Data columns (total 4 columns):" in captured.out

    # Ajustement pour être flexible avec le formatage
    assert "A" in captured.out and "5 non-null" in captured.out and "int64" in captured.out
    assert "B" in captured.out and "5 non-null" in captured.out and "int64" in captured.out
    assert "C" in captured.out and "5 non-null" in captured.out and "float64" in captured.out
    assert "D" in captured.out and "5 non-null" in captured.out and "object" in captured.out

# Test pour display_description
def test_display_description(sample_df, capsys):
    display_description(sample_df)
    captured = capsys.readouterr()
    assert "A         B         C" in captured.out
    assert "count  5.000000  5.000000  5.000000" in captured.out
    assert "mean   3.000000  3.000000  3.300000" in captured.out
    assert "std    1.581139  1.581139  1.739253" in captured.out

# Test pour plot_missing_values
def test_plot_missing_values(sample_df):
    # Vérifie que la fonction s'exécute sans erreur
    plot_missing_values(sample_df)
    plt.close()  # Ferme la figure pour éviter les interférences