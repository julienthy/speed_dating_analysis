import pandas as pd
import pytest
from src.visualization.reporting import generate_summary_report

@pytest.fixture
def sample_df(tmpdir):
    df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
    return df, str(tmpdir)

def test_generate_summary_report(sample_df, capsys):
    df, figures_path = sample_df
    generate_summary_report(df, figures_path)
    captured = capsys.readouterr()
    assert "Statistiques descriptives :" in captured.out
    assert "Informations structurelles :" in captured.out