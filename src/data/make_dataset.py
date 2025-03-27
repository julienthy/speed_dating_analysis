import pandas as pd
from typing import Union

def add_aggregated_column(df: pd.DataFrame, groupby_col: str, agg_col: str, agg_func: str, new_col_name: str) -> pd.DataFrame:
    """
    Ajoute une colonne au DataFrame avec les valeurs agrégées d'une colonne selon une variable de regroupement.
    
    Args:
        df (pd.DataFrame): DataFrame d'origine.
        groupby_col (str): Colonne de regroupement (ex. "iid").
        agg_col (str): Colonne à agréger (ex. "match").
        agg_func (str): Fonction d'agrégation (ex. "sum", "mean", "count").
        new_col_name (str): Nom de la nouvelle colonne à ajouter.
    
    Returns:
        pd.DataFrame: DataFrame avec la nouvelle colonne ajoutée.
    """
    df[new_col_name] = df.groupby(groupby_col)[agg_col].transform(agg_func)
    return df


def drop_rows_by_condition(df: pd.DataFrame, condition: callable) -> pd.DataFrame:
    """
    Supprime les lignes du DataFrame qui satisfont à la condition spécifiée.
    
    Args:
        df (pd.DataFrame): DataFrame d'origine.
        condition (callable): Fonction qui prend une ligne et renvoie True si la ligne doit être supprimée.
    
    Returns:
        pd.DataFrame: DataFrame filtré.
    """
    return df[~df.apply(condition, axis=1)]


def categorize_column(df: pd.DataFrame, col: str, bins: Union[int, list], labels: list = None, new_col_name: str = None) -> pd.DataFrame:
    """
    Catégorise une colonne quantitative en ajoutant une nouvelle colonne catégorielle.
    
    Args:
        df (pd.DataFrame): DataFrame d'origine.
        col (str): Colonne à catégoriser.
        bins (int or list): Nombre de catégories ou liste des seuils.
        labels (list, optional): Noms des catégories.
        new_col_name (str, optional): Nom de la nouvelle colonne (par défaut, col + "_cat").
    
    Returns:
        pd.DataFrame: DataFrame avec la nouvelle colonne catégorielle.
    """
    if new_col_name is None:
        new_col_name = f"{col}_cat"
    df[new_col_name] = pd.cut(df[col], bins=bins, labels=labels)
    return df


def merge_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, key: str, how: str = "inner") -> pd.DataFrame:
    """
    Fusionne deux DataFrames sur une clé commune.
    
    Args:
        df1 (pd.DataFrame): Premier DataFrame.
        df2 (pd.DataFrame): Deuxième DataFrame.
        key (str): Colonne de fusion.
        how (str): Type de fusion ("inner", "left", "right", "outer"). Par défaut : "inner".
    
    Returns:
        pd.DataFrame: DataFrame fusionné.
    """
    return pd.merge(df1, df2, on=key, how=how)


def rename_columns(df: pd.DataFrame, rename_dict: dict) -> pd.DataFrame:
    """
    Renomme les colonnes du DataFrame selon un dictionnaire de correspondance.
    
    Args:
        df (pd.DataFrame): DataFrame d'origine.
        rename_dict (dict): Dictionnaire avec les anciens noms comme clés et les nouveaux comme valeurs.
    
    Returns:
        pd.DataFrame: DataFrame avec les colonnes renommées.
    """
    return df.rename(columns=rename_dict)


def select_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Sélectionne un sous-ensemble de colonnes dans le DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame d'origine.
        columns (list): Liste des colonnes à conserver.
    
    Returns:
        pd.DataFrame: DataFrame avec uniquement les colonnes spécifiées.
    """
    return df[columns]


def convert_types(df: pd.DataFrame, type_dict: dict) -> pd.DataFrame:
    """
    Convertit les types de colonnes du DataFrame selon un dictionnaire.
    
    Args:
        df (pd.DataFrame): DataFrame d'origine.
        type_dict (dict): Dictionnaire avec les noms de colonnes comme clés et les types comme valeurs.
    
    Returns:
        pd.DataFrame: DataFrame avec les types convertis.
    """
    return df.astype(type_dict)