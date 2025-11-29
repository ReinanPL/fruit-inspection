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

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import gridspec
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
        # Pré-processamento
        img_resized = cv2.resize(img, self.feature_extractor.img_size)
        img_preprocessed = self.feature_extractor.preprocessor.preprocess(img.copy())
        img_for_features = img_preprocessed
        
        # Features de Cor
        hsv = cv2.cvtColor(img_for_features, cv2.COLOR_RGB2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv)

        hist_r = cv2.calcHist([img_for_features], [0], None, [256], [0, 256]).flatten()
        hist_g = cv2.calcHist([img_for_features], [1], None, [256], [0, 256]).flatten()
        hist_b = cv2.calcHist([img_for_features], [2], None, [256], [0, 256]).flatten()

        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256]).flatten()

        # Features de Textura
        gray = cv2.cvtColor(img_for_features, cv2.COLOR_RGB2GRAY)
        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-6)

        # Features de Forma
        edges = cv2.Canny(gray, 50, 150)
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        _, dark_regions = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 11, 2)

        # Visualização
        status = "FRUTA FRESCA" if not result['is_rotten'] else "FRUTA PODRE"
        status_color = 'green' if not result['is_rotten'] else 'red'

        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(4, 5, figure=fig, height_ratios=[1, 1, 1, 1.2],
                               width_ratios=[1, 1, 1, 1, 1.5],
                               wspace=0.1, hspace=0.3)

        # Original
        ax_orig_title = fig.add_subplot(gs[0, 0])
        ax_orig_title.axis('off')
        ax_orig_title.text(0.5, 0.9, f"Original: {os.path.basename(image_path)}",
                           fontsize=12, ha='center', va='top', fontweight='bold')
        ax_orig_title.imshow(img_resized)
        if img_preprocessed is not None:
            ax_orig_title.text(0.5, 0.05, "Pré-processada", ha='center', va='bottom')

        # Linha 1: Canais
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(h_channel, cmap='hsv')
        ax1.set_title('Canal H (Matiz)')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.imshow(s_channel, cmap='gray')
        ax2.set_title('Canal S (Saturação)')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 3])
        ax3.imshow(v_channel, cmap='gray')
        ax3.set_title('Canal V (Brilho)')
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[0, 4])
        ax4.plot(hist_r, color='red', alpha=0.7, label='R')
        ax4.plot(hist_g, color='green', alpha=0.7, label='G')
        ax4.plot(hist_b, color='blue', alpha=0.7, label='B')
        ax4.set_title('Histograma RGB')
        ax4.set_xlim([0, 256])
        ax4.legend(fontsize=8)
        
        # Linha 2: Textura
        ax5 = fig.add_subplot(gs[1, 0])
        if img_preprocessed is not None:
            ax5.imshow(img_preprocessed)
            ax5.set_title('Sem Fundo')
        else:
            ax5.text(0.5, 0.5, "N/A", ha='center', va='center')
        ax5.axis('off')

        ax5_gray = fig.add_subplot(gs[1, 1])
        ax5_gray.imshow(gray, cmap='gray')
        ax5_gray.set_title('Escala de Cinza')
        ax5_gray.axis('off')
        
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.imshow(lbp, cmap='gray')
        ax6.set_title('LBP (Textura)')
        ax6.axis('off')
        
        ax7 = fig.add_subplot(gs[1, 3])
        ax7.imshow(gradient_magnitude, cmap='hot')
        ax7.set_title('Gradiente')
        ax7.axis('off')
        
        ax8 = fig.add_subplot(gs[1, 4])
        ax8.plot(hist_h, color='purple', alpha=0.7, label='H')
        ax8.plot(hist_s, color='orange', alpha=0.7, label='S')
        ax8.plot(hist_v, color='gray', alpha=0.7, label='V')
        ax8.set_title('Histograma HSV')
        ax8.legend(fontsize=8)
        
        # Linha 3: Defeitos
        ax9 = fig.add_subplot(gs[2, 0])
        ax9.imshow(edges, cmap='gray')
        ax9.set_title('Bordas')
        ax9.axis('off')
        
        ax10 = fig.add_subplot(gs[2, 1])
        ax10.imshow(dark_regions, cmap='hot')
        ax10.set_title('Regiões Escuras')
        ax10.axis('off')
        
        ax11 = fig.add_subplot(gs[2, 2])
        ax11.imshow(adaptive_thresh, cmap='gray')
        ax11.set_title('Threshold Adaptativo')
        ax11.axis('off')
        
        ax12 = fig.add_subplot(gs[2, 3])
        ax12.bar(range(len(lbp_hist)), lbp_hist, color='steelblue', alpha=0.7)
        ax12.set_title('Histograma LBP')
        
        # Texto Features
        ax13 = fig.add_subplot(gs[2, 4])
        ax13.axis('off')
        feature_text = f"""
FEATURES:
---------
COR: 204
TEXTURA: 54
FORMA: 7
DEFEITOS: 6
---------
TOTAL: 271 features
        """
        ax13.text(0.05, 0.95, feature_text, transform=ax13.transAxes,
                 fontfamily='monospace', verticalalignment='top')
        
        # Linha 4: Resultado
        ax14 = fig.add_subplot(gs[3, :4])
        top_5_idx = np.argsort(result['all_probabilities'])[-5:][::-1]
        top_5_classes = [result['class_names'][i] for i in top_5_idx]
        top_5_probs = result['all_probabilities'][top_5_idx]
        
        colors = ['red' if result['class_names'][i] == result['class'] else 'lightgray'
                 for i in top_5_idx]
        
        y_pos = np.arange(len(top_5_classes))
        bars = ax14.barh(y_pos, top_5_probs, color=colors, alpha=0.8)
        ax14.set_yticks(y_pos)
        ax14.set_yticklabels(top_5_classes)
        ax14.set_xlabel('Probabilidade')
        ax14.set_title('Top Predições')
        ax14.set_xlim([0, 1])
        
        for i, (bar, prob) in enumerate(zip(bars, top_5_probs)):
            ax14.text(prob + 0.02, bar.get_y() + bar.get_height()/2,
                     f'{prob*100:.1f}%', va='center')
        
        ax15 = fig.add_subplot(gs[3, 4])
        ax15.axis('off')
        result_text = f"""
{status}

Classe:
{result['class']}

Confiança:
{result['confidence']*100:.2f}%
        """
        ax15.text(0.5, 0.5, result_text, transform=ax15.transAxes,
                 ha='center', va='center', fontweight='bold',
                 color=status_color)
        
        fig.suptitle('Inspeção de Qualidade', fontsize=16, fontweight='bold')
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
