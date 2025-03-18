import pandas as pd
from src.visualization.quantitative import plot_correlation_matrix
from src.visualization.qualitative import plot_bar_chart, plot_pie_chart

def generate_summary_report(df: pd.DataFrame, figures_path: str) -> None:
    """Génère un rapport résumé avec statistiques et visualisations."""
    print("Statistiques descriptives :")
    print(df.describe(include='all'))
    print("\nInformations structurelles :")
    print(df.info())
    
    numeric_cols = df.select_dtypes(include=['float', 'int']).columns
    object_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if not numeric_cols.empty:
        plot_correlation_matrix(df[numeric_cols], figures_path)
    for col in object_cols:
        plot_bar_chart(df, col, figures_path)
        plot_pie_chart(df, col, figures_path)