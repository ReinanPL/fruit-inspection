"""
Módulo de Inspeção
==================

Este módulo implementa o sistema de inspeção para predizer a qualidade
de novas frutas usando o modelo treinado.

Funcionalidades:
----------------
- Predição em imagem única
- Predição em lote (batch)
- Visualização detalhada dos resultados
- Relatórios de inspeção
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


class FruitInspector:
    """
    Sistema de inspeção para predição de qualidade de frutas.
    
    Parâmetros
    ----------
    classifier : FruitClassifier
        Classificador treinado
    feature_extractor : FeatureExtractor
        Extrator de features
    
    Atributos
    ---------
    classifier : FruitClassifier
        Classificador usado para predições
    feature_extractor : FeatureExtractor
        Extrator de features
    
    Exemplos
    --------
    >>> inspector = FruitInspector(classifier, feature_extractor)
    >>> result = inspector.predict_image('nova_fruta.jpg')
    >>> print(f"Classe: {result['class']}, Confiança: {result['confidence']:.2%}")
    """
    
    def __init__(self, classifier, feature_extractor):
        self.classifier = classifier
        self.feature_extractor = feature_extractor
    
    def predict_image(self, image_path, show_details=True):
        """
        Prediz a qualidade de uma fruta em uma imagem.
        
        Parâmetros
        ----------
        image_path : str
            Caminho para a imagem
        show_details : bool
            Se True, mostra visualização detalhada
        
        Retorna
        -------
        dict
            Dicionário com:
            - 'class': Classe predita
            - 'confidence': Confiança da predição (0-1)
            - 'is_rotten': Boolean indicando se é podre
            - 'all_probabilities': Probabilidades de todas as classes
            - 'class_names': Nomes de todas as classes
        """
        # Carregar imagem
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Extrair features
        features = self.feature_extractor.extract_all_features(img_rgb)
        features_scaled = self.classifier.scaler.transform([features])
        
        # Predição
        pred_encoded = self.classifier.model.predict(features_scaled)[0]
        pred_proba = self.classifier.model.predict_proba(features_scaled)[0]
        
        # Decodificar
        predicted_class = self.classifier.label_encoder.inverse_transform([pred_encoded])[0]
        confidence = np.max(pred_proba)
        
        is_rotten = 'rotten' in predicted_class.lower() or 'podre' in predicted_class.lower()
        
        result = {
            'class': predicted_class,
            'confidence': confidence,
            'is_rotten': is_rotten,
            'all_probabilities': pred_proba,
            'class_names': self.classifier.label_encoder.classes_
        }
        
        if show_details:
            self._visualize_prediction(img_rgb, result, image_path)
        
        return result
    
    def _visualize_prediction(self, img, result, image_path):
        """
        Visualiza resultado da predição.
        
        Mostra:
        - Imagem original
        - Detecção de bordas
        - Detecção de defeitos
        - Top 5 predições com probabilidades
        """
        # Redimensionar para visualização
        img_display = cv2.resize(img, (300, 300))
        
        # Criar visualização das features
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_resized = cv2.resize(gray, (256, 256))
        
        edges = cv2.Canny(gray_resized, 50, 150)
        _, dark = cv2.threshold(gray_resized, 60, 255, cv2.THRESH_BINARY_INV)
        
        fig = plt.figure(figsize=(16, 5))
        
        # Grid layout
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
        
        # Imagem original
        ax1 = fig.add_subplot(gs[:, 0])
        ax1.imshow(img_display)
        ax1.set_title('Imagem Original', fontsize=12)
        ax1.axis('off')
        
        # Detecção de bordas
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(edges, cmap='gray')
        ax2.set_title('Bordas (Canny)', fontsize=10)
        ax2.axis('off')
        
        # Imagem Pré-processada (Fundo Removido)
        ax5 = fig.add_subplot(gs[1, 0])
        try:
            img_preprocessed = self.feature_extractor.preprocessor.preprocess(img)
            ax5.imshow(img_preprocessed)
            ax5.set_title('Pré-processada (Sem Fundo)', fontsize=10)
        except Exception:
            ax5.text(0.5, 0.5, "N/A", ha='center', va='center')
            ax5.set_title('Pré-processada', fontsize=10)
        ax5.axis('off')

        # Escala de cinza
        ax5_gray = fig.add_subplot(gs[1, 1])
        ax5_gray.imshow(gray, cmap='gray')
        ax5_gray.set_title('Escala de Cinza', fontsize=10)
        ax5_gray.axis('off')
        
        # Top 5 probabilidades
        ax4 = fig.add_subplot(gs[:, 2:])
        top_5_idx = np.argsort(result['all_probabilities'])[-5:][::-1]
        top_5_classes = [result['class_names'][i] for i in top_5_idx]
        top_5_probs = result['all_probabilities'][top_5_idx]
        
        colors = ['red' if result['class_names'][i] == result['class'] else 'gray' 
                 for i in top_5_idx]
        
        y_pos = np.arange(len(top_5_classes))
        ax4.barh(y_pos, top_5_probs, color=colors, alpha=0.7)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(top_5_classes, fontsize=10)
        ax4.set_xlabel('Probabilidade')
        ax4.set_title('Top 5 Predições', fontsize=12)
        ax4.set_xlim([0, 1])
        ax4.grid(axis='x', alpha=0.3)
        
        # Título com resultado
        status = "⚠️ PODRE" if result['is_rotten'] else "✅ FRESCA"
        color = 'red' if result['is_rotten'] else 'green'
        
        fig.suptitle(f'Resultado: {status} - {result["class"]}\n'
                    f'Confiança: {result["confidence"]*100:.2f}% | '
                    f'Modelo: {self.classifier.model_name}',
                    fontsize=14, color=color, weight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def batch_inspect(self, image_paths, threshold=0.7):
        """
        Inspeciona um lote de frutas.
        
        Parâmetros
        ----------
        image_paths : list
            Lista de caminhos para imagens
        threshold : float
            Threshold de confiança mínima (padrão: 0.7)
        
        Retorna
        -------
        list
            Lista de dicionários com resultados para cada imagem
        """
        print(f"\n🔍 Inspecionando {len(image_paths)} frutas...")
        print("="*70)
        
        results = []
        
        for img_path in tqdm(image_paths, desc="Processando"):
            result = self.predict_image(img_path, show_details=False)
            results.append(result)
        
        # Estatísticas
        total = len(results)
        rotten = sum([1 for r in results if r['is_rotten']])
        fresh = total - rotten
        low_confidence = sum([1 for r in results if r['confidence'] < threshold])
        
        print("\n" + "="*70)
        print("📊 RELATÓRIO DE INSPEÇÃO DO LOTE")
        print("="*70)
        print(f"\n✅ Frutas Frescas: {fresh} ({fresh/total*100:.1f}%)")
        print(f"❌ Frutas Podres: {rotten} ({rotten/total*100:.1f}%)")
        print(f"⚠️  Baixa Confiança (< {threshold*100:.0f}%): {low_confidence}")
        print(f"\n📈 Confiança Média: {np.mean([r['confidence'] for r in results])*100:.2f}%")
        print(f"📊 Modelo usado: {self.classifier.model_name}")
        print("="*70)
        
        # Visualização
        self._plot_batch_summary(results)
        
        return results
    
    def _plot_batch_summary(self, results):
        """
        Plota resumo do lote inspecionado.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Gráfico Pizza
        labels = ['Frescas', 'Podres']
        sizes = [
            sum([1 for r in results if not r['is_rotten']]),
            sum([1 for r in results if r['is_rotten']])
        ]
        colors = ['green', 'red']
        explode = (0.1, 0)
        
        axes[0].pie(sizes, explode=explode, labels=labels, colors=colors,
                   autopct='%1.1f%%', startangle=90, textprops={'fontsize': 14})
        axes[0].set_title('Distribuição de Qualidade', fontsize=14)
        
        # Histograma de confiança
        confidences = [r['confidence'] for r in results]
        axes[1].hist(confidences, bins=20, color='steelblue',
                    edgecolor='black', alpha=0.7)
        axes[1].axvline(np.mean(confidences), color='red', linestyle='--',
                       linewidth=2, label=f'Média: {np.mean(confidences):.3f}')
        axes[1].set_xlabel('Confiança')
        axes[1].set_ylabel('Frequência')
        axes[1].set_title('Distribuição de Confiança', fontsize=14)
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.suptitle(f'Resumo da Inspeção - {self.classifier.model_name}',
                    fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
