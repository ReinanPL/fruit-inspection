"""
Módulo de Extração de Features
================================

Este módulo é responsável por extrair características (features) das imagens de frutas
usando técnicas de Visão Computacional Clássica.

Features Extraídas:
-------------------
1. COR (204 features):
   - Histogramas RGB (96 features)
   - Histogramas HSV (96 features)
   - Estatísticas de cor (12 features)

2. TEXTURA (54 features):
   - Local Binary Pattern - LBP (26 features)
   - Gray Level Co-occurrence Matrix - GLCM (20 features)
   - Estatísticas de textura (8 features)

3. FORMA (7 features):
   - Densidade de bordas
   - Percentual de regiões escuras
   - Score de defeitos
   - Gradientes

4. DEFEITOS (6 features):
   - Manchas circulares
   - Área de defeitos
   - Simetria
   - Uniformidade de saturação
   - Variação local de cor
   - Regiões escuras conectadas

Total: 271 features por imagem
"""

import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import matplotlib.pyplot as plt


class ImagePreprocessor:
    """
    Realiza pré-processamento das imagens das frutas.
    
    Técnicas aplicadas:
    1. Redimensionamento
    2. Remoção de ruído (Gaussian Blur)
    3. Equalização de histograma (CLAHE)
    4. Segmentação de fundo (Background Removal)
    """
    
    def __init__(self, img_size=(256, 256)):
        self.img_size = img_size
    
    def preprocess(self, image):
        """
        Aplica pipeline de pré-processamento.
        
        Parâmetros
        ----------
        image : numpy.ndarray
            Imagem RGB original
            
        Retorna
        -------
        numpy.ndarray
            Imagem processada (fundo removido, realçada)
        """
        # 1. Redimensionar
        img_resized = cv2.resize(image, self.img_size)
        
        # 2. Remover ruído
        img_blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)
        
        # 3. Segmentação de fundo (Máscara)
        img_segmented = self._segment_background(img_blurred)
        
        # 4. Melhorar contraste (CLAHE) na imagem segmentada
        img_enhanced = self._apply_clahe(img_segmented)
        
        return img_enhanced
    
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
        
        # Aplicar máscara
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) // 255
        img_segmented = image * mask_3ch
        
        return img_segmented

    def _apply_clahe(self, image):
        # Converter para LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Aplicar CLAHE no canal L (Luminosidade)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # Reconstruir
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        
        # Garantir que o fundo continue preto
        mask = (image.sum(axis=2) > 0).astype(np.uint8)
        mask_3ch = cv2.merge([mask, mask, mask])
        final = final * mask_3ch
        
        return final


class FeatureExtractor:
    """
    Extrai features de cor, textura e forma usando técnicas de CV clássica.
    
    Parâmetros
    ----------
    img_size : tuple
        Tamanho para redimensionar as imagens (largura, altura)
        Padrão: (256, 256)
    
    Atributos
    ---------
    img_size : tuple
        Tamanho padrão das imagens
    
    Exemplos
    --------
    >>> extractor = FeatureExtractor(img_size=(256, 256))
    >>> image = cv2.imread('fruta.jpg')
    >>> image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    >>> features = extractor.extract_all_features(image_rgb)
    >>> print(f"Features extraídas: {len(features)}")
    """
    
    def __init__(self, img_size=(256, 256)):
        self.img_size = img_size
        self.preprocessor = ImagePreprocessor(img_size)
    
    def extract_color_features(self, image):
        """
        Extrai features de cor usando histogramas RGB e HSV.
        """
        # Pré-processamento OBRIGATÓRIO
        img = self.preprocessor.preprocess(image)
        
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
        """
        Extrai features de textura usando LBP e GLCM.
        
        Técnicas Usadas:
        ----------------
        1. LBP (Local Binary Pattern): Analisa padrões de textura local
        2. GLCM (Gray Level Co-occurrence Matrix): Analisa relações espaciais
        3. Estatísticas básicas: Média, desvio padrão, quartis, etc.
        
        Parâmetros
        ----------
        image : numpy.ndarray
            Imagem RGB
        
        Retorna
        -------
        numpy.ndarray
            Array com 54 features de textura
        """
        # Pré-processamento OBRIGATÓRIO
        img = self.preprocessor.preprocess(image)
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
        """
        Extrai features de forma e detecta possíveis defeitos.
        
        Detecções:
        ----------
        1. Bordas (Canny): Detecta contornos e irregularidades
        2. Regiões escuras: Identifica manchas e podridão
        3. Threshold adaptativo: Detecta variações locais
        4. Gradientes: Mede mudanças de intensidade
        
        Parâmetros
        ----------
        image : numpy.ndarray
            Imagem RGB
        
        Retorna
        -------
        numpy.ndarray
            Array com 7 features de forma
        """
        # Pré-processamento OBRIGATÓRIO
        img = self.preprocessor.preprocess(image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Detecção de bordas (Canny)
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
        """
        Extrai features específicas para detecção de defeitos.
        """
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
        """
        Extrai TODAS as features de uma imagem.
        
        Parâmetros
        ----------
        image : numpy.ndarray
            Imagem RGB
        
        Retorna
        -------
        numpy.ndarray
            Array com 271 features:
            - 204 features de cor
            - 54 features de textura
            - 7 features de forma
            - 6 features de defeitos
        """
        color_feat = self.extract_color_features(image)      # 204 features
        texture_feat = self.extract_texture_features(image)  # 54 features
        shape_feat = self.extract_shape_features(image)      # 7 features
        defect_feat = self.extract_defect_features(image)    # 6 features
        
        # Total: 271 features
        all_features = np.concatenate([color_feat, texture_feat, shape_feat, defect_feat])
        
        return all_features
    
    def visualize_features(self, image_path):
        """
        Visualiza as features extraídas de uma imagem.
        
        Mostra:
        -------
        - Imagem original
        - Canais HSV
        - Escala de cinza
        - LBP (textura)
        - Bordas detectadas
        - Regiões escuras (defeitos)
        - Threshold adaptativo
        - Histograma RGB
        
        Parâmetros
        ----------
        image_path : str
            Caminho para a imagem
        """
        # Carregar imagem
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img_rgb, self.img_size)
        
        # Converter para escala de cinza e HSV
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # LBP
        lbp = local_binary_pattern(gray, P=24, R=3, method='uniform')
        
        # Detecção de bordas
        edges = cv2.Canny(gray, 50, 150)
        
        # Regiões escuras
        _, dark = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        
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
        axes[2, 0].imshow(dark, cmap='hot')
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
