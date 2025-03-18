import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.visualization.quantitative import plot_correlation_matrix, plot_feature_distributions, plot_boxplots
from src.visualization.qualitative import plot_bar_chart, plot_pie_chart, plot_contingency_heatmap
from pathlib import Path

def explore_quantitative_data(df: pd.DataFrame, figures_path: str) -> None:
    """Explore les données quantitatives."""
    numeric_cols = df.select_dtypes(include=['float', 'int']).columns
    if numeric_cols.empty:
        print("Aucune donnée quantitative à explorer.")
        return
    X = df[numeric_cols]
    plot_correlation_matrix(X, figures_path)
    plot_feature_distributions(X, figures_path)
    plot_boxplots(X, figures_path)

def explore_qualitative_data(df: pd.DataFrame, figures_path: str) -> None:
    """Explore les données qualitatives."""
    object_cols = df.select_dtypes(include=['object', 'category']).columns
    if object_cols.empty:
        print("Aucune donnée qualitative à explorer.")
        return
    for col in object_cols:
        plot_bar_chart(df, col, figures_path)
        plot_pie_chart(df, col, figures_path)
    if len(object_cols) >= 2:
        plot_contingency_heatmap(df, object_cols[0], object_cols[1], figures_path)

def explore_mixed_data(df: pd.DataFrame, figures_path: str) -> None:
    """Analyse exploratoire mixte pour données quantitatives et qualitatives."""
    numeric_cols = df.select_dtypes(include=['float', 'int']).columns
    object_cols = df.select_dtypes(include=['object', 'category']).columns
    if not numeric_cols.empty and not object_cols.empty:
        for num_col in numeric_cols:
            for obj_col in object_cols:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=obj_col, y=num_col, data=df, palette="Set3")
                plt.title(f"Distribution de {num_col} par {obj_col}")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(Path(figures_path) / f"boxplot_{num_col}_by_{obj_col}.png")
                plt.clf()
                plt.close()