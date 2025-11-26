"""
Sistema de Inspeção de Qualidade de Frutas
===========================================

Sistema completo de inspeção de qualidade de frutas usando:
- Visão Computacional Clássica (CV)
- Machine Learning Tradicional (SVM, Random Forest)

Módulos:
--------
- feature_extractor: Extração de features de cor, textura e forma
- dataset_loader: Carregamento e preparação de datasets
- classifier: Treinamento e avaliação de modelos ML
- visualizer: Visualização de resultados
- inspector: Sistema de inspeção para novas imagens
- pipeline: Pipeline completo orquestrado

Autor: Sistema de Inspeção de Frutas
Versão: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Fruit Inspection System"

from .feature_extractor import FeatureExtractor
from .dataset_loader import DatasetLoader
from .classifier import FruitClassifier
from .visualizer import ResultVisualizer
from .inspector import FruitInspector
from .pipeline import SimpleFruitInspectionPipeline

__all__ = [
    'FeatureExtractor',
    'DatasetLoader',
    'FruitClassifier',
    'ResultVisualizer',
    'FruitInspector',
    'SimpleFruitInspectionPipeline'
]
