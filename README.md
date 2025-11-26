# 🍎 Sistema de Inspeção de Qualidade de Frutas

Sistema completo de inspeção automatizada de qualidade de frutas usando Visão Computacional Clássica e Machine Learning Tradicional.

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Instalação](#instalação)
- [Uso](#uso)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Módulos](#módulos)
- [Exemplos](#exemplos)
- [Tecnologias](#tecnologias)

## 🎯 Visão Geral

Este projeto implementa um sistema completo de inspeção de qualidade de frutas seguindo uma abordagem em duas fases:

### Fase 1: Processamento de Imagem (Visão Computacional Clássica)
- Extração de 265 features por imagem
- **Features de Cor** (204): Histogramas RGB e HSV, estatísticas de cor
- **Features de Textura** (54): LBP (Local Binary Pattern), GLCM (Gray Level Co-occurrence Matrix)
- **Features de Forma** (7): Detecção de bordas, defeitos, gradientes

### Fase 2: Classificação (Machine Learning)
- **SVM** (Support Vector Machine): Kernel RBF com parâmetros otimizados
- **Classificação Binária**: Fresca vs. Podre
- **Probabilidade**: Confiança da predição

## 🏗️ Arquitetura

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Imagem de     │────▶│  Extração de     │────▶│  Classificação  │
│     Fruta       │     │    Features      │     │    ML (SVM)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                              │                           │
                              ▼                           ▼
                        265 features              Fresca / Podre
                        (cor, textura,            (com confiança)
                         forma)
```

## 📦 Instalação

### Requisitos
- Python 3.7+
- pip

### Passo a Passo

1. **Clone o repositório** (ou baixe os arquivos)
```bash
cd fruit-inspection
```

2. **Instale as dependências**
```bash
pip install -r requirements.txt
```

3. **Organize seu dataset**
```
dataset/
├── fresh_apple/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── rotten_apple/
│   ├── img1.jpg
│   └── ...
├── fresh_banana/
└── ...
```

## 🚀 Uso

### Uso Básico

```python
from src.pipeline import SimpleFruitInspectionPipeline

# Criar pipeline
pipeline = SimpleFruitInspectionPipeline(
    dataset_path='/path/to/dataset',
    img_size=(256, 256)
)

# Executar pipeline completo
pipeline.run_complete_pipeline(
    max_images_per_class=None  # Use todas as imagens
)

# Inspecionar nova fruta
result = pipeline.inspector.predict_image('nova_fruta.jpg')
print(f"Classe: {result['class']}")
print(f"Confiança: {result['confidence']:.2%}")
print(f"É podre? {result['is_rotten']}")

# Inspecionar lote
results = pipeline.inspector.batch_inspect([
    'fruta1.jpg',
    'fruta2.jpg',
    'fruta3.jpg'
])
```

### Uso Avançado (Modular)

```python
from src.feature_extractor import FeatureExtractor
from src.dataset_loader import DatasetLoader
from src.classifier import FruitClassifier
from src.visualizer import ResultVisualizer
from src.inspector import FruitInspector

# 1. Extração de features
extractor = FeatureExtractor(img_size=(256, 256))
loader = DatasetLoader('/path/to/dataset', extractor)
X, y, paths, classes = loader.load_dataset()

# 2. Treinamento
classifier = FruitClassifier(classes)
X_train, X_test, y_train, y_test = classifier.prepare_data(X, y)
classifier.train_svm(X_train, y_train)

# 3. Avaliação
y_pred, y_proba, acc, report = classifier.evaluate(X_test, y_test)

# 4. Visualização
visualizer = ResultVisualizer(classifier)
visualizer.plot_classification_metrics(report)

# 5. Inspeção
inspector = FruitInspector(classifier, extractor)
result = inspector.predict_image('nova_fruta.jpg')
```

## 📁 Estrutura do Projeto

```
fruit-inspection/
│
├── src/                          # Código-fonte modular
│   ├── __init__.py              # Inicialização do pacote
│   ├── feature_extractor.py    # Extração de features (CV)
│   ├── dataset_loader.py       # Carregamento de dados
│   ├── classifier.py           # Modelo ML (SVM)
│   ├── visualizer.py           # Visualizações
│   ├── inspector.py            # Sistema de inspeção
│   └── pipeline.py             # Pipeline completo
│
├── config/                      # Configurações
│   └── config.yaml             # Parâmetros do sistema
│
├── notebooks/                   # Notebooks de exemplo
│   └── example_usage.ipynb     # Exemplo de uso
│
├── requirements.txt             # Dependências Python
├── README.md                    # Este arquivo
└── colab_notebook.ipynb        # Notebook para Google Colab
```

## 🧩 Módulos

### 1. `feature_extractor.py`
**Extração de Features de Visão Computacional**

- `extract_color_features()`: Histogramas RGB/HSV + estatísticas
- `extract_texture_features()`: LBP + GLCM + estatísticas
- `extract_shape_features()`: Bordas, defeitos, gradientes
- `visualize_features()`: Visualiza extração de features

**Features extraídas: 265 total**
- 204 features de cor
- 54 features de textura
- 7 features de forma

### 2. `dataset_loader.py`
**Carregamento e Preparação de Dados**

- `load_dataset()`: Carrega imagens e extrai features
- `create_dataframe()`: Cria DataFrame com features

### 3. `classifier.py`
**Treinamento de Modelos ML**

- `prepare_data()`: Prepara dados (split, normalização)
- `train_svm()`: Treina SVM com kernel RBF
- `evaluate()`: Avalia modelo (acurácia, F1, etc.)

### 4. `visualizer.py`
**Visualização de Resultados**

- `plot_classification_metrics()`: Precisão, Recall, F1
- `plot_confidence_distribution()`: Distribuição de confiança

### 5. `inspector.py`
**Sistema de Inspeção**

- `predict_image()`: Prediz qualidade de uma fruta
- `batch_inspect()`: Inspeciona lote de frutas

### 6. `pipeline.py`
**Pipeline Completo**

- `run_complete_pipeline()`: Executa fluxo completo
  - Carrega dataset
  - Extrai features
  - Treina modelos
  - Avalia e visualiza
  - Cria sistema de inspeção

## 💡 Exemplos

### Exemplo 1: Treinamento Rápido (para testes)

```python
pipeline = SimpleFruitInspectionPipeline('/path/to/dataset')
pipeline.run_complete_pipeline(
    max_images_per_class=50      # Limita a 50 imagens por classe
)
```

### Exemplo 2: Treinamento Completo

```python
pipeline = SimpleFruitInspectionPipeline('/path/to/dataset')
pipeline.run_complete_pipeline()
```

### Exemplo 3: Visualizar Features de Uma Imagem

```python
from src.feature_extractor import FeatureExtractor

extractor = FeatureExtractor()
extractor.visualize_features('minha_fruta.jpg')
```



## 🛠️ Tecnologias

### Visão Computacional
- **OpenCV**: Processamento de imagens
- **scikit-image**: Features de textura (LBP, GLCM)
- **PIL**: Carregamento de imagens

### Machine Learning
- **scikit-learn**: SVM, métricas
- **imbalanced-learn**: Tratamento de desbalanceamento

### Visualização
- **matplotlib**: Gráficos e visualizações
- **seaborn**: Visualizações estatísticas

### Utilitários
- **numpy**: Operações numéricas
- **pandas**: Manipulação de dados
- **tqdm**: Barras de progresso

## 📊 Resultados Esperados

### Métricas Típicas
- **Acurácia**: 85-95% (depende do dataset)
- **Precisão**: 80-95% por classe
- **Recall**: 80-95% por classe
- **F1-Score**: 80-95% por classe

### Tempo de Execução (exemplo com 1000 imagens)
- Extração de features: ~5-10 minutos
- Treinamento SVM: ~2-5 minutos
- Predição: ~1 segundo por imagem

## 🤝 Contribuições

Este projeto foi desenvolvido para fins educacionais. Sugestões e melhorias são bem-vindas!

## 📝 Licença

Este projeto é fornecido para fins educacionais.

## 👥 Autores

Desenvolvido como trabalho acadêmico para inspeção automatizada de qualidade de frutas.

---

**Dúvidas?** Consulte os notebooks de exemplo em `notebooks/` ou o arquivo Colab `colab_notebook.ipynb`.
