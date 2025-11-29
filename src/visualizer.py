"""
Módulo de Visualização
=======================

Este módulo cria visualizações dos resultados da classificação,
incluindo matrizes de confusão, métricas por classe, distribuição
de confiança e comparação de modelos.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class ResultVisualizer:
    """
    Visualiza resultados da classificação de frutas.
    
    Parâmetros
    ----------
    classifier : FruitClassifier
        Instância do classificador treinado
    
    Atributos
    ---------
    classifier : FruitClassifier
        Classificador usado para visualização
    
    Exemplos
    --------
    >>> visualizer = ResultVisualizer(classifier)
    >>> visualizer.plot_confusion_matrix(y_test, y_pred)
    >>> visualizer.plot_classification_metrics(report)
    """
    
    def __init__(self, classifier):
        self.classifier = classifier
    
    def plot_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        class_names = self.classifier.label_encoder.classes_

        # Matriz normalizada
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title(f'Matriz de Confusão Normalizada - {self.classifier.model_name}')
        plt.ylabel('Real')
        plt.xlabel('Predito')
        plt.show()
    

    
    def plot_classification_metrics(self, report):
        """
        Plota métricas de classificação por classe.
        
        Mostra 3 gráficos de barras:
        - Precisão: % de predições corretas
        - Recall: % de casos detectados
        - F1-Score: média harmônica de Precisão e Recall
        
        Parâmetros
        ----------
        report : dict
            Relatório de classificação (sklearn.classification_report)
        """
        # Preparar dados
        class_names = self.classifier.label_encoder.classes_
        metrics_data = []
        
        for class_name in class_names:
            if class_name in report:
                metrics_data.append({
                    'Classe': class_name,
                    'Precisão': report[class_name]['precision'],
                    'Recall': report[class_name]['recall'],
                    'F1-Score': report[class_name]['f1-score']
                })
        
        df = pd.DataFrame(metrics_data)
        
        # Plotar 3 gráficos
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Precisão
        df_sorted = df.sort_values('Precisão', ascending=True)
        colors = ['green' if 'fresh' in c.lower() else 'red' 
                 for c in df_sorted['Classe']]
        axes[0].barh(df_sorted['Classe'], df_sorted['Precisão'], 
                    color=colors, alpha=0.7)
        axes[0].set_xlabel('Precisão')
        axes[0].set_title('Precisão por Classe')
        axes[0].set_xlim([0, 1])
        axes[0].axvline(df['Precisão'].mean(), color='blue', linestyle='--',
                       label=f'Média: {df["Precisão"].mean():.3f}')
        axes[0].legend()
        axes[0].grid(axis='x', alpha=0.3)
        
        # Recall
        df_sorted = df.sort_values('Recall', ascending=True)
        colors = ['green' if 'fresh' in c.lower() else 'red' 
                 for c in df_sorted['Classe']]
        axes[1].barh(df_sorted['Classe'], df_sorted['Recall'], 
                    color=colors, alpha=0.7)
        axes[1].set_xlabel('Recall')
        axes[1].set_title('Recall por Classe')
        axes[1].set_xlim([0, 1])
        axes[1].axvline(df['Recall'].mean(), color='blue', linestyle='--',
                       label=f'Média: {df["Recall"].mean():.3f}')
        axes[1].legend()
        axes[1].grid(axis='x', alpha=0.3)
        
        # F1-Score
        df_sorted = df.sort_values('F1-Score', ascending=True)
        colors = ['green' if 'fresh' in c.lower() else 'red' 
                 for c in df_sorted['Classe']]
        axes[2].barh(df_sorted['Classe'], df_sorted['F1-Score'], 
                    color=colors, alpha=0.7)
        axes[2].set_xlabel('F1-Score')
        axes[2].set_title('F1-Score por Classe')
        axes[2].set_xlim([0, 1])
        axes[2].axvline(df['F1-Score'].mean(), color='blue', linestyle='--',
                       label=f'Média: {df["F1-Score"].mean():.3f}')
        axes[2].legend()
        axes[2].grid(axis='x', alpha=0.3)
        
        plt.suptitle(f'Métricas de Classificação - {self.classifier.model_name}',
                    fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
    

    
    def plot_confidence_distribution(self, y_pred_proba):
        """
        Plota distribuição de confiança das predições.
        
        Mostra um histograma e boxplot da confiança máxima das predições.
        Alta confiança indica que o modelo está seguro das suas predições.
        
        Parâmetros
        ----------
        y_pred_proba : numpy.ndarray
            Probabilidades preditas (n_samples, n_classes)
        """
        max_confidences = np.max(y_pred_proba, axis=1)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histograma
        axes[0].hist(max_confidences, bins=30, color='steelblue',
                    edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(max_confidences), color='red', linestyle='--',
                       linewidth=2, label=f'Média: {np.mean(max_confidences):.3f}')
        axes[0].set_xlabel('Confiança Máxima')
        axes[0].set_ylabel('Frequência')
        axes[0].set_title('Distribuição de Confiança das Predições')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Boxplot
        axes[1].boxplot(max_confidences, vert=True)
        axes[1].set_ylabel('Confiança')
        axes[1].set_title('Boxplot de Confiança')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.suptitle(f'{self.classifier.model_name} - Análise de Confiança',
                    fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
        
        # Estatísticas
        print("\n📊 Estatísticas de Confiança:")
        print(f"   Média: {np.mean(max_confidences):.4f}")
        print(f"   Mediana: {np.median(max_confidences):.4f}")
        print(f"   Desvio Padrão: {np.std(max_confidences):.4f}")
        print(f"   Mínimo: {np.min(max_confidences):.4f}")
        print(f"   Máximo: {np.max(max_confidences):.4f}")
    

