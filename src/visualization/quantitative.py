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


def plot_temporal_histograms(df: pd.DataFrame, var_base: str, times: list, groupby_col: str, figsize: tuple = (12, 6), save_path: str = None) -> None:
    """
    Trace des histogrammes superposés pour une variable à travers plusieurs points temporels, stratifiés par une variable catégorielle.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        var_base (str): Nom de base de la variable (ex. "attr1_" pour attr1_1, attr1_2, attr1_3).
        times (list): Liste des suffixes temporels (ex. ["1", "2", "3"]).
        groupby_col (str): Colonne de regroupement (ex. "gender").
        figsize (tuple): Taille de la figure (largeur, hauteur).
        save_path (str, optional): Chemin pour sauvegarder la figure.
    """
    plt.figure(figsize=figsize)
    for time in times:
        var = f"{var_base}{time}"
        if var not in df.columns:
            continue
        for group in df[groupby_col].unique():
            subset = df[df[groupby_col] == group]
            sns.histplot(subset[var].dropna(), label=f"{group} - Temps {time}", kde=True, stat="density", alpha=0.5)
    plt.title(f"Évolution de {var_base} par {groupby_col} au fil du temps")
    plt.xlabel(var_base)
    plt.ylabel("Densité")
    plt.legend()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(save_path)/f"Evolution_{var_base}_by_{groupby_col}")

    plt.clf()
    plt.close()


def plot_temporal_histograms2(df: pd.DataFrame, var_base: str, times: list, groupby_col: str, figsize: tuple = (12, 6), save_path: str = None) -> None:
    """
    Trace des histogrammes séparés pour une variable à chaque point temporel, stratifiés par une variable catégorielle.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        var_base (str): Nom de base de la variable (ex. "attr1_" pour attr1_1, attr1_2, attr1_3).
        times (list): Liste des suffixes temporels (ex. ["1", "2", "3"]).
        groupby_col (str): Colonne de regroupement (ex. "gender").
        figsize (tuple): Taille de la figure par histogramme (largeur, hauteur).
        save_path (str, optional): Chemin de base pour sauvegarder les figures (une par temps).
    """
    for time in times:
        var = f"{var_base}{time}"
        if var not in df.columns:
            continue
        plt.figure(figsize=figsize)
        for group in df[groupby_col].unique():
            subset = df[df[groupby_col] == group]
            sns.histplot(subset[var].dropna(), label=f"{group}", kde=True, stat="density", alpha=0.5)
        plt.title(f"Distribution de {var_base} au temps {time} par {groupby_col}")
        plt.xlabel(var_base)
        plt.ylabel("Densité")
        plt.legend()
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(save_path)/f"temporal_histogram_{var_base}_time_{time}_by_{groupby_col}")

        plt.clf()
        plt.close()


def plot_scatter_comparison(df: pd.DataFrame, x_var: str, y_var: str, hue_col: str = None, figsize: tuple = (8, 6), save_path: str = None) -> None:
    """
    Trace un diagramme de dispersion pour comparer deux variables, avec coloration optionnelle par une variable catégorielle.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        x_var (str): Variable sur l’axe X (ex. "attr1_1").
        y_var (str): Variable sur l’axe Y (ex. "attr").
        hue_col (str, optional): Colonne pour la coloration (ex. "gender").
        figsize (tuple): Taille de la figure.
        save_path (str, optional): Chemin pour sauvegarder la figure.
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(data=df, x=x_var, y=y_var, hue=hue_col, alpha=0.7)
    plt.title(f"Comparaison entre {x_var} et {y_var}")
    plt.xlabel(x_var)
    plt.ylabel(y_var)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(save_path)/f"Comparaison_{x_var}_{y_var}")

    plt.clf()
    plt.close()


def plot_violin_comparison(df: pd.DataFrame, vars_to_compare: list, groupby_col: str, figsize: tuple = (10, 6), save_path: str = None) -> None:
    """
    Trace des diagrammes en violon pour comparer les distributions de deux variables, stratifiées par une variable catégorielle.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        vars_to_compare (list): Liste de deux variables à comparer (ex. ["attr3_1", "attr_o"]).
        groupby_col (str): Colonne de regroupement (ex. "gender").
        figsize (tuple): Taille de la figure.
        save_path (str, optional): Chemin pour sauvegarder la figure.
    """
    if len(vars_to_compare) != 2:
        raise ValueError("vars_to_compare doit contenir exactement deux variables.")
    
    df_long = df.melt(id_vars=[groupby_col], value_vars=vars_to_compare, var_name="Variable", value_name="Valeur")
    plt.figure(figsize=figsize)
    sns.violinplot(data=df_long.dropna(), x="Variable", y="Valeur", hue=groupby_col, split=True, inner="quart")
    plt.title(f"Comparaison des distributions de {vars_to_compare[0]} et {vars_to_compare[1]} par {groupby_col}")
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(save_path)/f"Distributions_{vars_to_compare[0]}_{vars_to_compare[1]}_by_{groupby_col}")

    plt.clf()
    plt.close()


def plot_boxplots_by_decision(df: pd.DataFrame, vars_list: list, decision_col: str, figsize: tuple = (12, 8), save_path: str = None) -> None:
    """
    Trace des boxplots pour des variables quantitatives stratifiées par la décision.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        vars_list (list): Liste des variables à visualiser (ex. ["attr", "sinc"]).
        decision_col (str): Colonne de décision (ex. "dec").
        figsize (tuple): Taille de la figure.
        save_path (str, optional): Chemin pour sauvegarder la figure.
    """
    n_vars = len(vars_list)
    fig, axes = plt.subplots(nrows=1, ncols=n_vars, figsize=figsize)
    if n_vars == 1:
        axes = [axes]  # Pour gérer le cas d’une seule variable
    for i, var in enumerate(vars_list):
        sns.boxplot(x=decision_col, y=var, data=df.dropna(subset=[var, decision_col]), ax=axes[i])
        axes[i].set_title(f"{var} par {decision_col}")
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(save_path)/f"Boxplots_by_{decision_col}")
    
    plt.clf()
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, vars_list: list, figsize: tuple = (10, 8), save_path: str = None) -> None:
    """
    Trace une heatmap de corrélation pour un ensemble de variables quantitatives.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        vars_list (list): Liste des variables à inclure (ex. ["attr", "sinc", "dec"]).
        figsize (tuple): Taille de la figure.
        save_path (str, optional): Chemin pour sauvegarder la figure.
    """
    corr_matrix = df[vars_list].corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Heatmap de corrélation")
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(save_path)/f"Heatmap_correlation")

    plt.clf()
    plt.close()