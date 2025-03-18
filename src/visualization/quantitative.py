"""
Module pour la visualisation de données quantitatives.
Contient des fonctions de visualisation de données quantitatives (matrice de corrélation, distribution, boxplot...).
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

import logging

# Configuration globale du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_correlation_matrix(df: pd.DataFrame, figures_path: str) -> pd.DataFrame:
    """Trace la matrice de corrélation."""
    corr_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", 
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
    plt.title("Matrice de corrélation", pad=20)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path(figures_path)/f"correlation_matrix.png")
    plt.clf()
    plt.close()

    return corr_matrix


def plot_feature_distributions(X: pd.DataFrame, figures_path: str) -> None:
    """Trace les distributions des features."""
    for col in X.columns:
        # Double visualisation
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogramme + KDE
        sns.histplot(X[col], kde=True, ax=axs[0], color='skyblue')
        axs[0].set_title(f"Distribution de {col}")
        
        # QQ-Plot
        stats.probplot(X[col], plot=axs[1])
        axs[1].set_title(f"QQ-Plot de {col}")
        
        plt.tight_layout()
        plt.savefig(Path(figures_path)/f"{col}_distribution.png", bbox_inches='tight')
        plt.clf()
        plt.close()


def plot_feature_target_relations(X: pd.DataFrame, y: pd.DataFrame, figures_path: str, classification_threshold: int = 10) -> None:
    """Trace les relations features-target."""
    for target_col in y.columns:
        target_type = 'classification' if (y[target_col].nunique() <= classification_threshold) else 'regression'
        
        for feature_col in X.columns:
            plt.figure(figsize=(10, 6))
            
            if target_type == 'classification':
                # Violin plot pour classification
                sns.violinplot(x=y[target_col], y=X[feature_col], hue=y[target_col],
                              palette="viridis", legend=False, cut=0)
                plt.title(f"Distribution de {feature_col} par classe de {target_col}")
            else:
                # Regression plot avec intervalle de confiance
                sns.regplot(x=X[feature_col], y=y[target_col], 
                           scatter_kws={'alpha':0.3, 'color':'slategray'},
                           line_kws={'color':'crimson'}, ci=95)
                plt.title(f"Relation {feature_col} vs {target_col}")
            
            plt.tight_layout()
            plt.savefig(Path(figures_path)/f"relation_{feature_col}_vs_{target_col}.png", dpi=120)
            plt.clf()
            plt.close()


def analyse_multivariee_selective(df: pd.DataFrame, y: pd.DataFrame, corr_matrix: pd.DataFrame, figures_path: str) -> None:
    if len(y.columns) == 1:  # Pour éviter les visualisations trop complexes
        target_col = y.columns[0]
        top_features = corr_matrix[target_col].abs().sort_values(ascending=False).index[1:4]
        
        # Pairplot ciblé
        sns.pairplot(df[top_features.tolist() + [target_col]], 
                    diag_kind='kde',
                    plot_kws={'alpha':0.5, 'edgecolor':'none'},
                    diag_kws={'fill':True})
        plt.suptitle("Relations clés avec la target", y=1.02)
        plt.savefig(Path(figures_path)/f"key_relationships.png", bbox_inches='tight')
        plt.clf()
        plt.close()


def plot_boxplots(X: pd.DataFrame, figures_path: str):
    for col in X.columns:
        plt.figure(figsize=(10, 4))
        sns.boxplot(x=col, data=X)
        plt.title(f"Distribution de {col}")
        plt.savefig(Path(figures_path)/f"{col}_boxplot.png")
        plt.clf()
        plt.close()


def diagramme_dispersion_cibles(df: pd.DataFrame, y: pd.DataFrame, corr_matrix: pd.DataFrame, figures_path: str, classification_threshold: int=10) -> None:
    for target_col in y.columns:
        try:
            # Sélection des 2 features les plus corrélées avec cette target
            top_features = corr_matrix[target_col].abs().sort_values(ascending=False).index[1:3]
            
            if len(top_features) >= 2:
                plt.figure(figsize=(10, 6))
                
                # Détermination du type de palette
                unique_classes = y[target_col].nunique()
                palette = "viridis" if (unique_classes <= classification_threshold) or (y[target_col].dtype == 'object') else "plasma"
                
                # Création du scatter plot
                scatter = sns.scatterplot(x=top_features[0], y=top_features[1], hue=target_col,
                                        data=df, palette=palette, alpha=0.7)
                
                # Optimisation de la légende
                handles, labels = scatter.get_legend_handles_labels()
                plt.legend(handles=handles[1:], title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left')
                
                # Personnalisation du titre
                plt.title(f"Interaction {top_features[0]} et {top_features[1]}\nColorée par {target_col}", pad=20)
                plt.xlabel(top_features[0], fontweight='bold')
                plt.ylabel(top_features[1], fontweight='bold')
                
                plt.savefig(Path(figures_path)/f"scatter_{target_col}.png", bbox_inches='tight', dpi=120)
                plt.clf()
                plt.close()
                
        except Exception as e:
            logging.error(f"Erreur avec la target {target_col} : {str(e)}")
            continue


def plot_boxenplot(X: pd.DataFrame, figures_path: str) -> None:
    plt.figure(figsize=(12, 6))
    sns.boxenplot(data=X, palette="Set3", orient="h")
    plt.title("Distribution des features avec détection d'outliers (boxenplot)")
    plt.tight_layout()
    plt.savefig(Path(figures_path)/f"outliers_detection.png")
    plt.clf()
    plt.close()