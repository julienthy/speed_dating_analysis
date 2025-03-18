"""
Module pour la visualisation de données qualitatives.
Contient des fonctions de visualisation de données qualitatives (diagrammes en barres, camemberts, heatmap...).
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic
from pathlib import Path
from typing import List

def plot_bar_chart(df: pd.DataFrame, column: str, figures_path: str, figsize: tuple = (10, 6)) -> None:
    """Trace un diagramme en barres pour une colonne qualitative."""
    plt.figure(figsize=figsize)
    df[column].value_counts().plot(kind='bar', color='skyblue')
    plt.title(f"Répartition de {column}")
    plt.xlabel(column)
    plt.ylabel("Fréquence")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path(figures_path) / f"bar_{column}.png")
    plt.clf()
    plt.close()

def plot_pie_chart(df: pd.DataFrame, column: str, figures_path: str, figsize: tuple = (8, 8)) -> None:
    """Trace un diagramme circulaire pour une colonne qualitative."""
    plt.figure(figsize=figsize)
    df[column].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title(f"Répartition de {column}")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(Path(figures_path) / f"pie_{column}.png")
    plt.clf()
    plt.close()

def plot_contingency_heatmap(df: pd.DataFrame, col1: str, col2: str, figures_path: str, figsize: tuple = (10, 8)) -> None:
    """Trace un heatmap pour le tableau de contingence entre deux colonnes qualitatives."""
    contingency_table = pd.crosstab(df[col1], df[col2])
    plt.figure(figsize=figsize)
    sns.heatmap(contingency_table, annot=True, cmap="YlGnBu", fmt="d")
    plt.title(f"Tableau de contingence entre {col1} et {col2}")
    plt.tight_layout()
    plt.savefig(Path(figures_path) / f"heatmap_{col1}_vs_{col2}.png")
    plt.clf()
    plt.close()

def plot_stacked_bar(df: pd.DataFrame, col1: str, col2: str, figures_path: str, figsize: tuple = (10, 6)) -> None:
    """Trace un diagramme en barres empilées pour deux variables qualitatives."""
    crosstab = pd.crosstab(df[col1], df[col2], normalize='index') * 100
    crosstab.plot(kind='bar', stacked=True, figsize=figsize)
    plt.title(f"Répartition de {col2} par {col1} (en %)")
    plt.xlabel(col1)
    plt.ylabel("Pourcentage")
    plt.legend(title=col2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(Path(figures_path) / f"stacked_bar_{col1}_vs_{col2}.png")
    plt.clf()
    plt.close()

def plot_countplot_with_hue(df: pd.DataFrame, x_col: str, hue_col: str, figures_path: str, figsize: tuple = (10, 6)) -> None:
    """Trace un countplot avec une variable de teinte (hue)."""
    plt.figure(figsize=figsize)
    sns.countplot(x=x_col, hue=hue_col, data=df, palette="Set2")
    plt.title(f"Comptage de {x_col} par {hue_col}")
    plt.xlabel(x_col)
    plt.ylabel("Nombre")
    plt.xticks(rotation=45)
    plt.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(Path(figures_path) / f"countplot_{x_col}_by_{hue_col}.png")
    plt.clf()
    plt.close()

def plot_mosaic(df: pd.DataFrame, columns: List[str], figures_path: str, figsize: tuple = (10, 8)) -> None:
    """Trace un diagramme en mosaïque pour plusieurs variables qualitatives."""
    plt.figure(figsize=figsize)
    mosaic(df, columns, gap=0.01)
    plt.title(f"Diagramme en mosaïque pour {', '.join(columns)}")
    plt.tight_layout()
    plt.savefig(Path(figures_path) / f"mosaic_{'_'.join(columns)}.png")
    plt.clf()
    plt.close()