# Sistema de inspeção de frutas (frescas x podres)

!pip install -q opencv-python-headless scikit-image scikit-learn

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from pathlib import Path
import cv2
import warnings
import time
from tqdm import tqdm

# Configurações
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops


# MÓDULO 0: PRÉ-PROCESSAMENTO DE IMAGENS
class ImagePreprocessor:
    
    def __init__(self, img_size=(256, 256)):
        self.img_size = img_size
    
    def preprocess(self, image):
        # 1. Redimensionar
        img = cv2.resize(image, self.img_size)
        
        # 2. Remover ruído (Gaussian Blur)
        img_denoised = cv2.GaussianBlur(img, (5, 5), 0)
        
        # 3. Equalizar histograma (CLAHE)
        img_lab = cv2.cvtColor(img_denoised, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_lab[:,:,0] = clahe.apply(img_lab[:,:,0])
        img_equalized = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        
        # 4. Segmentar fundo (HSV Threshold)
        img_segmented = self._segment_background(img_equalized)
        
        return img_segmented
    
    def _segment_background(self, image):
        # Converter para HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Threshold de saturação (frutas > 30)
        _, mask = cv2.threshold(hsv[:,:,1], 30, 255, cv2.THRESH_BINARY)
        
        # Limpar ruído
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Maior contorno
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
        # Aplicar máscara (Fundo Branco)
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) // 255
        # Onde é fundo (0), vira branco (255). Onde é fruta (1), mantém original.
        img_segmented = image * mask_3ch + (255 * (1 - mask_3ch)).astype(np.uint8)
        
        return img_segmented


# MÓDULO 1: EXTRAÇÃO DE FEATURES
class FeatureExtractor:

    def __init__(self, img_size=(256, 256)):
        self.img_size = img_size
        self.preprocessor = ImagePreprocessor(img_size)

    def extract_color_features(self, image):

        # Redimensionar
        img = cv2.resize(image, self.img_size)

        # Histogramas RGB (3 canais × 32 bins = 96 features)
        hist_r = cv2.calcHist([img], [0], None, [32], [0, 256]).flatten()
        hist_g = cv2.calcHist([img], [1], None, [32], [0, 256]).flatten()
        hist_b = cv2.calcHist([img], [2], None, [32], [0, 256]).flatten()

        # Converter para HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Histogramas HSV (3 canais × 32 bins = 96 features)
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()

        # Estatísticas de cor (média e desvio padrão por canal = 12 features)
        color_stats = np.array([
            np.mean(img[:,:,0]), np.std(img[:,:,0]),  # R
            np.mean(img[:,:,1]), np.std(img[:,:,1]),  # G
            np.mean(img[:,:,2]), np.std(img[:,:,2]),  # B
            np.mean(hsv[:,:,0]), np.std(hsv[:,:,0]),  # H
            np.mean(hsv[:,:,1]), np.std(hsv[:,:,1]),  # S
            np.mean(hsv[:,:,2]), np.std(hsv[:,:,2])   # V
        ])

        # Concatenar todas as features de cor (96+96+12 = 204 features)
        color_features = np.concatenate([
            hist_r, hist_g, hist_b,
            hist_h, hist_s, hist_v,
            color_stats
        ])

        return color_features

    def extract_texture_features(self, image):

        # Redimensionar e converter para escala de cinza
        img = cv2.resize(image, self.img_size)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 1. Local Binary Pattern (LBP)
        # Parâmetros: radius=3, n_points=24
        lbp = local_binary_pattern(gray, P=24, R=3, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26))
        lbp_hist = lbp_hist.astype(float)
        lbp_hist /= (lbp_hist.sum() + 1e-6)  # Normalizar (26 features)

        # 2. Gray Level Co-occurrence Matrix (GLCM)
        # Calcular GLCM em 4 direções
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

        glcm = graycomatrix(gray, distances=distances, angles=angles,
                           levels=256, symmetric=True, normed=True)

        # Propriedades de Haralick (5 × 4 direções = 20 features)
        glcm_props = []
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
            glcm_props.extend(graycoprops(glcm, prop)[0])

        glcm_features = np.array(glcm_props)

        # 3. Estatísticas de textura básicas (8 features)
        texture_stats = np.array([
            np.mean(gray),           # Média
            np.std(gray),            # Desvio padrão
            np.min(gray),            # Mínimo
            np.max(gray),            # Máximo
            np.median(gray),         # Mediana
            np.percentile(gray, 25), # 1º quartil
            np.percentile(gray, 75), # 3º quartil
            np.var(gray)             # Variância
        ])

        # Concatenar todas as features de textura (26+20+8 = 54 features)
        texture_features = np.concatenate([
            lbp_hist,
            glcm_features,
            texture_stats
        ])

        return texture_features

    def extract_shape_features(self, image):

        # Redimensionar
        img = cv2.resize(image, self.img_size)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Detecção de bordas
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Detecção de regiões escuras (possíveis defeitos)
        _, dark_regions = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        dark_percentage = np.sum(dark_regions > 0) / dark_regions.size

        # Detecção de manchas (usando threshold adaptativo)
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 11, 2)
        defect_score = 1 - (np.sum(adaptive_thresh > 0) / adaptive_thresh.size)

        # Gradiente (variação de intensidade)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_mean = np.mean(gradient_magnitude)
        gradient_std = np.std(gradient_magnitude)

        # Features de forma (7 features)
        shape_features = np.array([
            edge_density,
            dark_percentage,
            defect_score,
            gradient_mean,
            gradient_std,
            np.mean(edges),
            np.std(edges)
        ])

        return shape_features

    def extract_defect_features(self, image):
        
        img = cv2.resize(image, self.img_size)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # 1. Detecção de manchas circulares (Hough Circles)
        # Frutas podres têm manchas escuras circulares
        circles = cv2.HoughCircles(
            gray, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=20,
            param1=50, 
            param2=30, 
            minRadius=5, 
            maxRadius=50
        )
        num_spots = len(circles[0]) if circles is not None else 0
        
        # 2. Área de defeitos (threshold adaptativo)
        adaptive = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        defect_area_ratio = 1 - (np.sum(adaptive > 0) / adaptive.size)
        
        # 3. Análise de simetria (comparar metades esquerda/direita)
        h, w = gray.shape
        left_half = gray[:, :w//2]
        right_half = cv2.flip(gray[:, w//2:], 1)  # Espelhar
        
        # Redimensionar para mesmo tamanho
        if left_half.shape != right_half.shape:
            right_half = cv2.resize(right_half, (left_half.shape[1], left_half.shape[0]))
        
        # Diferença absoluta entre metades
        symmetry_score = 1 - (np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255)
        
        # 4. Uniformidade de cor (desvio padrão de saturação)
        saturation_std = np.std(hsv[:,:,1])
        saturation_uniformity = 1 / (1 + saturation_std / 100)  # Normalizar
        
        # 5. Variação local de cor (detecta manchas)
        # Calcular variância em janelas 16x16 usando convolução
        kernel_size = 16
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
        sat_channel = hsv[:,:,1].astype(np.float32)
        local_mean = cv2.filter2D(sat_channel, -1, kernel)
        local_sq_mean = cv2.filter2D(sat_channel**2, -1, kernel)
        local_variance = local_sq_mean - local_mean**2
        avg_local_variance = np.mean(local_variance)
        
        # 6. Contagem de regiões conectadas escuras
        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        num_dark_regions = num_labels - 1  # Subtrair o fundo
        
        # Vetor de features de defeitos (6 features)
        defect_features = np.array([
            num_spots,              # Quantidade de manchas circulares
            defect_area_ratio,      # Proporção de área com defeitos
            symmetry_score,         # Simetria (frutas podres perdem simetria)
            saturation_uniformity,  # Uniformidade de saturação
            avg_local_variance,     # Variação local de cor
            num_dark_regions        # Número de regiões escuras
        ])
        
        return defect_features

    def extract_all_features(self, image):

        # Pré-processamento obrigatório
        image = self.preprocessor.preprocess(image)

        color_feat = self.extract_color_features(image)      # 204 features
        texture_feat = self.extract_texture_features(image)  # 54 features
        shape_feat = self.extract_shape_features(image)      # 7 features
        defect_feat = self.extract_defect_features(image)    # 6 features

        # Total: 271 features
        all_features = np.concatenate([color_feat, texture_feat, shape_feat, defect_feat])

        return all_features

    def visualize_features(self, image_path):

        # Carregar imagem
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img_rgb, self.img_size)

        # Converter para escala de cinza
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # LBP
        lbp = local_binary_pattern(gray, P=24, R=3, method='uniform')

        # Detecção de bordas
        edges = cv2.Canny(gray, 50, 150)

        # Regiões escuras
        _, dark = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)

        # Threshold adaptativo
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)

        # Visualizar
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))

        # Linha 1: Imagem original e canais
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('Imagem Original', fontsize=12)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(hsv[:,:,0], cmap='hsv')
        axes[0, 1].set_title('Canal H (Matiz)', fontsize=12)
        axes[0, 1].axis('off')

        axes[0, 2].imshow(hsv[:,:,1], cmap='gray')
        axes[0, 2].set_title('Canal S (Saturação)', fontsize=12)
        axes[0, 2].axis('off')

        # Linha 2: Textura
        axes[1, 0].imshow(gray, cmap='gray')
        axes[1, 0].set_title('Escala de Cinza', fontsize=12)
        axes[1, 0].axis('off')

        axes[1, 1].imshow(lbp, cmap='gray')
        axes[1, 1].set_title('LBP (Textura)', fontsize=12)
        axes[1, 1].axis('off')

        axes[1, 2].imshow(edges, cmap='gray')
        axes[1, 2].set_title('Detecção de Bordas (Canny)', fontsize=12)
        axes[1, 2].axis('off')

        # Linha 3: Defeitos
        # Combinar regiões escuras com bordas para dar contexto
        dark_with_edges = dark.copy()
        dark_with_edges[edges > 0] = 0 # Bordas pretas
        axes[2, 0].imshow(dark_with_edges, cmap='gray')
        axes[2, 0].set_title('Regiões Escuras (Defeitos)', fontsize=12)
        axes[2, 0].axis('off')

        axes[2, 1].imshow(adaptive, cmap='gray')
        axes[2, 1].set_title('Threshold Adaptativo', fontsize=12)
        axes[2, 1].axis('off')

        # Histograma de cores
        for i, color in enumerate(['r', 'g', 'b']):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            axes[2, 2].plot(hist, color=color, alpha=0.7, label=color.upper())
        axes[2, 2].set_title('Histograma RGB', fontsize=12)
        axes[2, 2].legend()
        axes[2, 2].set_xlim([0, 256])

        plt.suptitle('Extração de Features - Visão Computacional Clássica',
                    fontsize=16, y=0.995)
        plt.tight_layout()
        plt.show()



# MÓDULO 2: CARREGAMENTO E PREPARAÇÃO DOS DADOS
class DatasetLoader:
    def __init__(self, dataset_path, feature_extractor):
        self.dataset_path = Path(dataset_path)
        self.feature_extractor = feature_extractor

    def load_dataset(self, max_images_per_class=None):
        print("Carregando dataset...")
        
        X = []
        y = []
        image_paths = []

        classes = [d.name for d in self.dataset_path.iterdir() if d.is_dir()]
        classes.sort()
        
        print(f"Classes encontradas: {classes}")

        total_images = 0

        for class_name in classes:
            class_path = self.dataset_path / class_name
            images = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))

            if max_images_per_class:
                images = images[:max_images_per_class]

            print(f"Processando {class_name}: {len(images)} imagens")

            for img_path in tqdm(images, desc=f"Extraindo features de {class_name}", leave=False):
                try:
                    img = cv2.imread(str(img_path))
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    features = self.feature_extractor.extract_all_features(img_rgb)
                    X.append(features)
                    y.append(class_name)
                    image_paths.append(str(img_path))
                    total_images += 1

                except Exception as e:
                    print(f"Erro em {img_path.name}: {e}")

        print(f"Total de imagens processadas: {total_images}")
        return np.array(X), np.array(y), image_paths, classes

    def create_dataframe(self, X, y, image_paths):
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        df = pd.DataFrame(X, columns=feature_names)
        df['label'] = y
        df['image_path'] = image_paths
        df['is_rotten'] = df['label'].apply(lambda x: 'rotten' in x.lower() or 'podre' in x.lower())
        
        return df


# MÓDULO 3: TREINAMENTO DO MODELO (SVM)
class FruitClassifier:
    def __init__(self, class_names):
        self.class_names = class_names
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = None
        self.model_name = "SVM"

    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        print("Preparando dados (Split e Normalização)...")

        # Codificar labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Split treino/teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )

        # Normalizar features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"Treino: {len(X_train)} amostras")
        print(f"Teste: {len(X_test)} amostras")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_svm(self, X_train, y_train):
        print("Treinando SVM...")
        start_time = time.time()

        # Parâmetros fixos e robustos
        svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42, probability=True)
        svm.fit(X_train, y_train)

        elapsed_time = time.time() - start_time
        self.model = svm
        print(f"SVM treinado em {elapsed_time:.2f} segundos")

        return self.model

    def evaluate(self, X_test, y_test):
        print(f"Avaliando {self.model_name}...")
        
        # Predições
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        # Acurácia
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Acurácia: {accuracy:.4f}")

        # Relatório detalhado
        class_names_decoded = self.label_encoder.classes_
        report = classification_report(y_test, y_pred,
                                      target_names=class_names_decoded,
                                      output_dict=True)

        return y_pred, y_pred_proba, accuracy, report

print("Módulos 1-3 carregados! Próximo: Visualização e Pipeline Completo")


# MÓDULO 4: SISTEMA DE INSPEÇÃO (PREDIÇÃO EM NOVAS IMAGENS)
class FruitInspector:
    def __init__(self, classifier, feature_extractor):
        self.classifier = classifier
        self.feature_extractor = feature_extractor

    def predict_image(self, image_path, show_details=True):
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        features = self.feature_extractor.extract_all_features(img_rgb)
        features_scaled = self.classifier.scaler.transform([features])

        pred_encoded = self.classifier.model.predict(features_scaled)[0]
        pred_proba = self.classifier.model.predict_proba(features_scaled)[0]

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

        # Criar máscara para ignorar o fundo branco (255)
        gray_for_mask = cv2.cvtColor(img_for_features, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray_for_mask, 254, 255, cv2.THRESH_BINARY_INV)

        # Features de Textura
        gray = cv2.cvtColor(img_for_features, cv2.COLOR_RGB2GRAY)
        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method="uniform")

        # Features de Forma
        edges = cv2.Canny(gray, 50, 150)
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        _, dark_regions = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 11, 2)

        # Visualização
        status = "FRUTA FRESCA" if not result['is_rotten'] else "FRUTA PODRE"
        status_color = 'green' if not result['is_rotten'] else 'red'

        fig = plt.figure(figsize=(16, 14))
        # Layout: 4 linhas x 4 colunas (Histogramas removidos)
        gs = gridspec.GridSpec(4, 4, figure=fig, height_ratios=[1, 1, 1, 1],
                               width_ratios=[1, 1, 1, 1.5],
                               wspace=0.2, hspace=0.4)

        # --- LINHA 1: VISÃO GERAL & RESULTADOS ---
        # Original
        ax_orig = fig.add_subplot(gs[0, 0])
        ax_orig.imshow(img_resized)
        ax_orig.set_title(f"Original\n{os.path.basename(image_path)}", fontsize=11, fontweight='bold')
        ax_orig.axis('off')

        # Pré-processada
        ax_pre = fig.add_subplot(gs[0, 1])
        if img_preprocessed is not None:
            ax_pre.imshow(img_preprocessed)
            ax_pre.set_title("Pré-processada\n(Fundo Branco)", fontsize=11)
        ax_pre.axis('off')

        # Texto Resultado
        ax_res_text = fig.add_subplot(gs[0, 2])
        ax_res_text.axis('off')
        result_text = f"{status}\n\nClasse: {result['class']}\nConfiança: {result['confidence']*100:.1f}%"
        ax_res_text.text(0.5, 0.5, result_text, ha='center', va='center', 
                        fontweight='bold', color=status_color, fontsize=14)

        # Gráfico de Probabilidades
        ax_probs = fig.add_subplot(gs[0, 3])
        top_5_idx = np.argsort(result['all_probabilities'])[-5:][::-1]
        top_5_classes = [result['class_names'][i] for i in top_5_idx]
        top_5_probs = result['all_probabilities'][top_5_idx]
        colors = ['red' if result['class_names'][i] == result['class'] else 'lightgray' for i in top_5_idx]
        
        y_pos = np.arange(len(top_5_classes))
        bars = ax_probs.barh(y_pos, top_5_probs, color=colors, alpha=0.8)
        ax_probs.set_yticks(y_pos)
        ax_probs.set_yticklabels(top_5_classes)
        ax_probs.set_title('Top Predições', fontsize=11)
        ax_probs.set_xlim([0, 1])
        for i, (bar, prob) in enumerate(zip(bars, top_5_probs)):
            ax_probs.text(prob + 0.02, bar.get_y() + bar.get_height()/2, f'{prob*100:.1f}%', va='center')

        # --- LINHA 2: MÓDULO 1 - COR ---
        # H Channel
        ax_h = fig.add_subplot(gs[1, 0])
        h_normalized = h_channel / 180.0
        h_rgba = plt.cm.hsv(h_normalized)
        mask_norm = mask / 255.0
        mask_norm = mask_norm[:, :, np.newaxis]
        h_final = h_rgba * mask_norm + (1 - mask_norm)
        ax_h.imshow(h_final)
        ax_h.set_title('Canal H (Matiz)', fontsize=10)
        ax_h.axis('off')

        # S Channel
        ax_s = fig.add_subplot(gs[1, 1])
        ax_s.imshow(s_channel, cmap='gray_r')
        ax_s.set_title('Canal S (Saturação)', fontsize=10)
        ax_s.axis('off')

        # V Channel
        ax_v = fig.add_subplot(gs[1, 2])
        ax_v.imshow(v_channel, cmap='gray')
        ax_v.set_title('Canal V (Brilho)', fontsize=10)
        ax_v.axis('off')
        
        # Info Cor
        ax_cor_info = fig.add_subplot(gs[1, 3])
        ax_cor_info.axis('off')
        cor_text = "MÓDULO 1: COR\n\n- Matiz (H): Cor predominante\n- Saturação (S): Intensidade\n- Brilho (V): Luminosidade"
        ax_cor_info.text(0.5, 0.5, cor_text, ha='center', va='center', fontsize=10, color='#444')

        # --- LINHA 3: MÓDULO 2 - TEXTURA ---
        # Grayscale
        ax_gray = fig.add_subplot(gs[2, 0])
        ax_gray.imshow(gray, cmap='gray')
        ax_gray.set_title('Escala de Cinza', fontsize=10)
        ax_gray.axis('off')

        # LBP
        ax_lbp = fig.add_subplot(gs[2, 1])
        ax_lbp.imshow(lbp, cmap='gray')
        ax_lbp.set_title('LBP (Textura)', fontsize=10)
        ax_lbp.axis('off')

        # Textura Info
        ax_tex_info = fig.add_subplot(gs[2, 2:])
        ax_tex_info.axis('off')
        tex_text = "MÓDULO 2: TEXTURA\n\n- LBP: Padrões Locais\n- GLCM: Contraste/Energia"
        ax_tex_info.text(0.5, 0.5, tex_text, ha='center', va='center', fontsize=10, color='#444')

        # --- LINHA 4: MÓDULO 3 (FORMA) & 4 (DEFEITOS) ---
        # Bordas
        ax_edges = fig.add_subplot(gs[3, 0])
        ax_edges.imshow(255 - edges, cmap='gray')
        ax_edges.set_title('Bordas', fontsize=10)
        ax_edges.axis('off')

        # Gradiente
        ax_grad = fig.add_subplot(gs[3, 1])
        ax_grad.imshow(gradient_magnitude, cmap='gray_r')
        ax_grad.set_title('Gradiente', fontsize=10)
        ax_grad.axis('off')

        # Regiões Escuras
        ax_dark = fig.add_subplot(gs[3, 2])
        dark_with_edges = dark_regions.copy()
        dark_with_edges[edges > 0] = 0 
        ax_dark.imshow(dark_with_edges, cmap='gray')
        ax_dark.set_title('Regiões Escuras', fontsize=10)
        ax_dark.axis('off')

        # Threshold Adaptativo
        ax_adapt = fig.add_subplot(gs[3, 3])
        ax_adapt.imshow(adaptive_thresh, cmap='gray')
        ax_adapt.set_title('Threshold Adaptativo', fontsize=10)
        ax_adapt.axis('off')

        # Títulos das Seções
        plt.figtext(0.1, 0.72, "MÓDULO 1: COR", fontsize=12, fontweight='bold', color='darkblue')
        plt.figtext(0.1, 0.52, "MÓDULO 2: TEXTURA", fontsize=12, fontweight='bold', color='darkblue')
        plt.figtext(0.1, 0.32, "MÓDULOS 3 & 4: FORMA E DEFEITOS", fontsize=12, fontweight='bold', color='darkblue')

        fig.suptitle('Inspeção de Qualidade - Análise por Módulos', fontsize=16, fontweight='bold')
        plt.show()

    def batch_inspect(self, image_paths, threshold=0.7):
        print(f"Inspecionando {len(image_paths)} frutas...")

        results = []
        for img_path in tqdm(image_paths, desc="Processando"):
            result = self.predict_image(img_path, show_details=False)
            results.append(result)

        total = len(results)
        rotten = sum([1 for r in results if r['is_rotten']])
        fresh = total - rotten
        low_confidence = sum([1 for r in results if r['confidence'] < threshold])

        print("\nRELATÓRIO DO LOTE")
        print(f"Frescas: {fresh} ({fresh/total*100:.1f}%)")
        print(f"Podres: {rotten} ({rotten/total*100:.1f}%)")
        print(f"Baixa Confiança: {low_confidence}")
        print(f"Confiança Média: {np.mean([r['confidence'] for r in results])*100:.2f}%")

        self._plot_batch_summary(results)
        return results

    def _plot_batch_summary(self, results):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        labels = ['Frescas', 'Podres']
        sizes = [
            sum([1 for r in results if not r['is_rotten']]),
            sum([1 for r in results if r['is_rotten']])
        ]
        colors = ['green', 'red']
        
        axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        axes[0].set_title('Distribuição de Qualidade')

        confidences = [r['confidence'] for r in results]
        axes[1].hist(confidences, bins=20, color='steelblue', alpha=0.7)
        axes[1].set_xlabel('Confiança')
        axes[1].set_ylabel('Frequência')
        axes[1].set_title('Distribuição de Confiança')
        
        plt.tight_layout()
        plt.show()


# MÓDULO 5: VISUALIZAÇÃO DE RESULTADOS
class ResultVisualizer:
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

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Precisão
        df_sorted = df.sort_values('Precisão', ascending=True)
        axes[0].barh(df_sorted['Classe'], df_sorted['Precisão'], color='skyblue')
        axes[0].set_title('Precisão por Classe')
        axes[0].set_xlim([0, 1])

        # Recall
        df_sorted = df.sort_values('Recall', ascending=True)
        axes[1].barh(df_sorted['Classe'], df_sorted['Recall'], color='lightgreen')
        axes[1].set_title('Recall por Classe')
        axes[1].set_xlim([0, 1])

        # F1-Score
        df_sorted = df.sort_values('F1-Score', ascending=True)
        axes[2].barh(df_sorted['Classe'], df_sorted['F1-Score'], color='salmon')
        axes[2].set_title('F1-Score por Classe')
        axes[2].set_xlim([0, 1])

        plt.tight_layout()
        plt.show()


# MÓDULO 6: PIPELINE COMPLETO
class SimpleFruitInspectionPipeline:
    
    def __init__(self, dataset_path, img_size=(256, 256)):
        self.dataset_path = dataset_path
        self.img_size = img_size

        self.feature_extractor = FeatureExtractor(img_size)
        self.loader = DatasetLoader(dataset_path, self.feature_extractor)
        self.classifier = None
        self.visualizer = None
        self.inspector = None

    def run_complete_pipeline(self, max_images_per_class=None):
        print("INICIANDO PIPELINE COMPLETO")

        # 1. Carregar dados
        X, y, image_paths, classes = self.loader.load_dataset(max_images_per_class)

        # 2. Treinar SVM
        classifier_svm = FruitClassifier(classes)
        X_train, X_test, y_train, y_test = classifier_svm.prepare_data(X, y)

        classifier_svm.train_svm(X_train, y_train)
        y_pred, y_pred_proba, accuracy, report = classifier_svm.evaluate(X_test, y_test)

        self.classifier = classifier_svm

        # 3. Sistema de Inspeção
        self.inspector = FruitInspector(self.classifier, self.feature_extractor)

        # 4. Visualizar Resultados
        visualizer_svm = ResultVisualizer(classifier_svm)
        visualizer_svm.plot_classification_metrics(report)
        self.visualizer = visualizer_svm

        print("\nSistema pronto!")
        print("Use: inspector.predict_image('imagem.jpg')")


# EXEMPLO DE USO
dataset_path = "/content/drive/MyDrive/dataset-menor"

# Criar pipeline
pipeline = SimpleFruitInspectionPipeline(
    dataset_path=dataset_path,
    img_size=(256, 256)
)

# Executar
pipeline.run_complete_pipeline(
    max_images_per_class=None
)

# Teste
# result = pipeline.inspector.predict_image("caminho/para/imagem.jpg")
