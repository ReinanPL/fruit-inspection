"""
Módulo de Carregamento de Dataset
===================================

Este módulo carrega imagens de frutas de uma estrutura de diretórios
e extrai features de todas as imagens usando o FeatureExtractor.

Estrutura esperada do dataset:
-------------------------------
dataset/
├── fresh_apple/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── rotten_apple/
│   ├── img1.jpg
│   └── ...
└── ...

O nome da pasta é usado como label (classe) da fruta.
"""

import os
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm


class DatasetLoader:
    """
    Carrega dataset de imagens e extrai features.
    
    Parâmetros
    ----------
    dataset_path : str ou Path
        Caminho para o diretório raiz do dataset
    feature_extractor : FeatureExtractor
        Instância do extrator de features
    
    Atributos
    ---------
    dataset_path : Path
        Caminho para o dataset
    feature_extractor : FeatureExtractor
        Extrator de features configurado
    
    Exemplos
    --------
    >>> from feature_extractor import FeatureExtractor
    >>> extractor = FeatureExtractor()
    >>> loader = DatasetLoader('/path/to/dataset', extractor)
    >>> X, y, paths, classes = loader.load_dataset()
    """
    
    def __init__(self, dataset_path, feature_extractor):
        self.dataset_path = Path(dataset_path)
        self.feature_extractor = feature_extractor
    
    def load_dataset(self, max_images_per_class=None):
        """
        Carrega todas as imagens do dataset e extrai features.
        
        Parâmetros
        ----------
        max_images_per_class : int, opcional
            Número máximo de imagens por classe (útil para testes rápidos)
            Se None, carrega todas as imagens
        
        Retorna
        -------
        X : numpy.ndarray
            Array de features (n_samples, n_features)
        y : numpy.ndarray
            Array de labels (n_samples,)
        image_paths : list
            Lista com caminhos das imagens
        classes : list
            Lista com nomes das classes
        
        Exemplos
        --------
        >>> X, y, paths, classes = loader.load_dataset(max_images_per_class=100)
        >>> print(f"Carregadas {len(X)} imagens de {len(classes)} classes")
        """
        print("\n📂 Carregando dataset e extraindo features...")
        print("="*70)
        
        X = []  # Features
        y = []  # Labels
        image_paths = []
        
        # Encontrar todas as classes (subdiretórios)
        classes = [d.name for d in self.dataset_path.iterdir() if d.is_dir()]
        classes.sort()
        
        print(f"✅ Encontradas {len(classes)} classes: {', '.join(classes[:5])}...")
        
        total_images = 0
        
        # Para cada classe
        for class_name in classes:
            class_path = self.dataset_path / class_name
            
            # Pegar imagens da classe
            images = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
            
            # Limitar se necessário (para testes rápidos)
            if max_images_per_class:
                images = images[:max_images_per_class]
            
            print(f"   📸 {class_name}: {len(images)} imagens")
            
            # Extrair features de cada imagem
            for img_path in tqdm(images, desc=f"   Processando {class_name}", leave=False):
                try:
                    # Carregar imagem
                    img = cv2.imread(str(img_path))
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Extrair features
                    features = self.feature_extractor.extract_all_features(img_rgb)
                    
                    X.append(features)
                    y.append(class_name)
                    image_paths.append(str(img_path))
                    total_images += 1
                
                except Exception as e:
                    print(f"      ⚠️ Erro ao processar {img_path.name}: {e}")
        
        print(f"\n✅ Total de imagens processadas: {total_images}")
        print(f"✅ Dimensão das features: {len(X[0])} features por imagem")
        print("="*70)
        
        return np.array(X), np.array(y), image_paths, classes
    
    def create_dataframe(self, X, y, image_paths):
        """
        Cria DataFrame com features e labels.
        
        Parâmetros
        ----------
        X : numpy.ndarray
            Array de features
        y : numpy.ndarray
            Array de labels
        image_paths : list
            Lista com caminhos das imagens
        
        Retorna
        -------
        pandas.DataFrame
            DataFrame com colunas:
            - feature_0 até feature_264: Features extraídas
            - label: Nome da classe
            - image_path: Caminho da imagem
            - is_rotten: Boolean indicando se é podre
        
        Exemplos
        --------
        >>> df = loader.create_dataframe(X, y, image_paths)
        >>> print(df.head())
        >>> print(f"Frutas podres: {df['is_rotten'].sum()}")
        """
        # Criar nomes das features
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Criar DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['label'] = y
        df['image_path'] = image_paths
        
        # Identificar frutas podres (baseado no nome da classe)
        df['is_rotten'] = df['label'].apply(
            lambda x: 'rotten' in x.lower() or 'podre' in x.lower()
        )
        
        return df
