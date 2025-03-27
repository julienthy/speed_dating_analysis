"""
Module pour la gestion des entrées/sorties du projet.
Contient des fonctions pour charger et sauvegarder des données et des modèles.
"""
import os
import yaml
import json
import pickle
import pandas as pd
from typing import Dict, Any
import logging

# Configuration globale du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str = "../config.yaml") -> Dict[str, Any]:
    """
    Charge le fichier de configuration YAML.
    
    Args:
        config_path: Chemin vers le fichier de configuration.
        
    Returns:
        Dict contenant la configuration.
        
    Raises:
        FileNotFoundError: Si le fichier de configuration n'existe pas.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration chargée depuis {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Fichier de configuration {config_path} non trouvé.")
        raise FileNotFoundError(f"Fichier de configuration {config_path} non trouvé.")

def load_data(file_path: str, encoding: str = "utf-8") -> pd.DataFrame:
    """
    Charge les données depuis un fichier CSV.
    
    Args:
        file_path: Chemin vers le fichier CSV.
        
    Returns:
        DataFrame contenant les données.
        
    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
    """
    if not os.path.exists(file_path):
        logger.error(f"Fichier de données {file_path} non trouvé.")
        raise FileNotFoundError(f"Fichier de données {file_path} non trouvé.")
    
    # Déterminer l'extension du fichier
    _, ext = os.path.splitext(file_path)
    
    if ext.lower() == '.csv':
        df = pd.read_csv(file_path, encoding=encoding)
    elif ext.lower() in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path)
    else:
        logger.error(f"Format de fichier non supporté: {ext}")
        raise ValueError(f"Format de fichier non supporté: {ext}")
    
    logger.info(f"Données chargées depuis {file_path}: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df

def save_data(df: pd.DataFrame, file_path: str, index: bool = False) -> None:
    """
    Sauvegarde un DataFrame dans un fichier CSV.
    
    Args:
        df: DataFrame à sauvegarder.
        file_path: Chemin pour sauvegarder le fichier.
        index: Si True, sauvegarde les index.
    """
    # Créer le répertoire si nécessaire
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Déterminer l'extension du fichier
    _, ext = os.path.splitext(file_path)
    
    if ext.lower() == '.csv':
        df.to_csv(file_path, index=index)
    elif ext.lower() in ['.xls', '.xlsx']:
        df.to_excel(file_path, index=index)
    else:
        logger.warning(f"Extension inconnue, sauvegarde par défaut en CSV: {file_path}")
        df.to_csv(file_path, index=index)
    
    logger.info(f"Données sauvegardées dans {file_path}")

def save_model(model: Any, model_path: str) -> None:
    """
    Sauvegarde un modèle en utilisant pickle.
    
    Args:
        model: Modèle à sauvegarder.
        model_path: Chemin pour sauvegarder le modèle.
    """
    # Créer le répertoire si nécessaire
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Modèle sauvegardé dans {model_path}")

def load_model(model_path: str) -> Any:
    """
    Charge un modèle sauvegardé avec pickle.
    
    Args:
        model_path: Chemin vers le modèle sauvegardé.
        
    Returns:
        Le modèle chargé.
        
    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
    """
    if not os.path.exists(model_path):
        logger.error(f"Fichier de modèle {model_path} non trouvé.")
        raise FileNotFoundError(f"Fichier de modèle {model_path} non trouvé.")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"Modèle chargé depuis {model_path}")
    return model

def save_metrics(metrics: Dict[str, Any], metrics_path: str) -> None:
    """
    Sauvegarde les métriques d'évaluation au format JSON.
    
    Args:
        metrics: Dictionnaire de métriques.
        metrics_path: Chemin pour sauvegarder les métriques.
    """
    # Créer le répertoire si nécessaire
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Métriques sauvegardées dans {metrics_path}")

def ensure_dir(directory: str) -> None:
    """
    Crée un répertoire s'il n'existe pas.
    
    Args:
        directory: Chemin du répertoire à créer.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Répertoire créé: {directory}")