"""
Módulo de Classificação
========================

Este módulo implementa o treinamento e avaliação de modelos de Machine Learning
para classificação de qualidade de frutas.

Modelos Suportados:
-------------------
1. SVM (Support Vector Machine): Kernel RBF
2. Random Forest: Ensemble de árvores de decisão

Ambos suportam GridSearch para otimização de hiperparâmetros.
"""

import numpy as np
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class FruitClassifier:
    """
    Treina e avalia classificadores de ML para inspeção de frutas.
    
    Parâmetros
    ----------
    class_names : list
        Lista com nomes das classes
    
    Atributos
    ---------
    class_names : list
        Nomes das classes
    label_encoder : LabelEncoder
        Codificador de labels
    scaler : StandardScaler
        Normalizador de features
    model : sklearn estimator
        Modelo treinado
    model_name : str
        Nome do modelo ('SVM' ou 'Random Forest')
    
    Exemplos
    --------
    >>> classifier = FruitClassifier(class_names=['fresh_apple', 'rotten_apple'])
    >>> X_train, X_test, y_train, y_test = classifier.prepare_data(X, y)
    >>> classifier.train_svm(X_train, y_train)
    >>> y_pred, y_proba, acc, report = classifier.evaluate(X_test, y_test)
    """
    
    def __init__(self, class_names):
        self.class_names = class_names
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = None
        self.model_name = None
    
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """
        Prepara dados para treinamento.
        
        Processo:
        ---------
        1. Codifica labels (texto -> números)
        2. Divide em treino e teste (stratified)
        3. Normaliza features (StandardScaler)
        
        Parâmetros
        ----------
        X : numpy.ndarray
            Features (n_samples, n_features)
        y : numpy.ndarray
            Labels (n_samples,)
        test_size : float
            Proporção do conjunto de teste (padrão: 0.2)
        random_state : int
            Seed para reprodutibilidade (padrão: 42)
        
        Retorna
        -------
        X_train_scaled : numpy.ndarray
            Features de treino normalizadas
        X_test_scaled : numpy.ndarray
            Features de teste normalizadas
        y_train : numpy.ndarray
            Labels de treino codificadas
        y_test : numpy.ndarray
            Labels de teste codificadas
        """
        print("\n🔧 Preparando dados...")
        
        # Codificar labels (texto -> números)
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split treino/teste (stratified mantém proporção das classes)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y_encoded
        )
        
        # Normalizar features (média=0, desvio=1)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✅ Treino: {len(X_train)} amostras")
        print(f"✅ Teste: {len(X_test)} amostras")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_svm(self, X_train, y_train):
        """
        Treina SVM (Support Vector Machine).
        
        Hiperparâmetros Fixos:
        -----------------------
        - kernel: 'rbf' (Radial Basis Function)
        - C: 10 (penalização)
        - gamma: 'scale'
        
        Parâmetros
        ----------
        X_train : numpy.ndarray
            Features de treino
        y_train : numpy.ndarray
            Labels de treino
        
        Retorna
        -------
        sklearn.svm.SVC
            Modelo SVM treinado
        """
        print("\n🚀 Treinando SVM...")
        start_time = time.time()
        
        # SVM com parâmetros fixos (sem Grid Search)
        svm = SVC(kernel='rbf', C=10, gamma='scale', 
                 random_state=42, probability=True)
        
        svm.fit(X_train, y_train)
        
        elapsed_time = time.time() - start_time
        
        self.model = svm
        self.model_name = "SVM"
        print(f"✅ SVM treinado em {elapsed_time:.2f} segundos!")
        
        return self.model
    

    
    def evaluate(self, X_test, y_test):
        """
        Avalia o modelo treinado.
        
        Métricas Calculadas:
        --------------------
        - Acurácia geral
        - Precisão, Recall e F1-Score por classe
        - Matriz de confusão
        
        Parâmetros
        ----------
        X_test : numpy.ndarray
            Features de teste
        y_test : numpy.ndarray
            Labels de teste
        
        Retorna
        -------
        y_pred : numpy.ndarray
            Predições (labels codificadas)
        y_pred_proba : numpy.ndarray
            Probabilidades por classe
        accuracy : float
            Acurácia geral
        report : dict
            Relatório completo de classificação
        """
        print(f"\n📊 Avaliando {self.model_name}...")
        print("="*70)
        
        # Predições
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Acurácia
        accuracy = accuracy_score(y_test, y_pred)
        print(f"✅ Acurácia: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Relatório detalhado
        class_names_decoded = self.label_encoder.classes_
        report = classification_report(
            y_test, y_pred,
            target_names=class_names_decoded,
            output_dict=True
        )
        
        print("\n📈 Métricas por classe:")
        print("-"*70)
        for class_name in class_names_decoded:
            if class_name in report:
                metrics = report[class_name]
                print(f"{class_name:30} | "
                      f"Precisão: {metrics['precision']:.3f} | "
                      f"Recall: {metrics['recall']:.3f} | "
                      f"F1: {metrics['f1-score']:.3f}")
        
        print("="*70)
        
        return y_pred, y_pred_proba, accuracy, report
