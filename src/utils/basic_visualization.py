import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno

def display_head(df: pd.DataFrame, n: int = 5) -> None:
    """
    Affiche les n premières lignes du DataFrame.
    
    Args:
        df (pd.DataFrame): Le DataFrame à afficher.
        n (int): Nombre de lignes à afficher (par défaut 5).
    """
    print(df.head(n))

def display_info(df: pd.DataFrame) -> None:
    """
    Affiche les informations structurelles du DataFrame.
    
    Args:
        df (pd.DataFrame): Le DataFrame à analyser.
    """
    print(df.info())

def display_description(df: pd.DataFrame) -> None:
    """
    Affiche les statistiques descriptives du DataFrame.
    
    Args:
        df (pd.DataFrame): Le DataFrame à analyser.
    """
    print(df.describe())

def plot_missing_values(df: pd.DataFrame, figsize: tuple = (10, 6)) -> None:
    """
    Affiche un heatmap des valeurs manquantes.
    
    Args:
        df (pd.DataFrame): Le DataFrame à analyser.
        figsize (tuple): Taille de la figure (par défaut (10, 6)).
    """
    plt.figure(figsize=figsize)
    msno.matrix(df)
    plt.title("Heatmap des valeurs manquantes")
    plt.show()