"""
Módulo de Pipeline Completo
============================

Este módulo orquestra todo o fluxo de trabalho do sistema de inspeção:
1. Carregamento de dados
2. Extração de features
3. Treinamento de modelos
4. Avaliação e visualização
5. Criação do sistema de inspeção

É o ponto de entrada principal para usar o sistema completo.
"""

from .feature_extractor import FeatureExtractor
from .dataset_loader import DatasetLoader
from .classifier import FruitClassifier
from .visualizer import ResultVisualizer
from .inspector import FruitInspector


class SimpleFruitInspectionPipeline:
    """
    Pipeline completo de inspeção de qualidade de frutas.
    
    Combina todas as etapas do sistema em um fluxo único e simplificado.
    
    Parâmetros
    ----------
    dataset_path : str
        Caminho para o diretório do dataset
    img_size : tuple
        Tamanho para redimensionar imagens (padrão: (256, 256))
    
    Atributos
    ---------
    feature_extractor : FeatureExtractor
        Extrator de features configurado
    loader : DatasetLoader
        Carregador de dataset
    classifier : FruitClassifier
        Classificador treinado (disponível após treinamento)
    visualizer : ResultVisualizer
        Visualizador de resultados
    inspector : FruitInspector
        Sistema de inspeção (disponível após treinamento)
    
    Exemplos
    --------
    >>> pipeline = SimpleFruitInspectionPipeline(
    ...     dataset_path='/path/to/dataset',
    ...     img_size=(256, 256)
    ... )
    >>> pipeline.run_complete_pipeline(model_type='rf')
    >>> result = pipeline.inspector.predict_image('nova_fruta.jpg')
    """
    
    def __init__(self, dataset_path, img_size=(256, 256)):
        self.dataset_path = dataset_path
        self.img_size = img_size
        
        # Inicializar componentes
        self.feature_extractor = FeatureExtractor(img_size)
        self.loader = DatasetLoader(dataset_path, self.feature_extractor)
        self.classifier = None
        self.visualizer = None
        self.inspector = None
    
    def run_complete_pipeline(self, max_images_per_class=None):
        """
        Executa o pipeline completo.
        
        Parâmetros
        ----------
        max_images_per_class : int, opcional
            Número máximo de imagens por classe
            Se None, usa todas as imagens
            Use valores menores (ex: 50) para testes rápidos
        
        Retorna
        -------
        self
            Retorna a própria instância do pipeline
        
        Exemplos
        --------
        >>> # Execução rápida para testes
        >>> pipeline.run_complete_pipeline(max_images_per_class=50)
        
        >>> # Execução completa
        >>> pipeline.run_complete_pipeline()
        """
        print("\n" + "="*70)
        print("🚀 PIPELINE COMPLETO - CV CLÁSSICA + ML TRADICIONAL")
        print("="*70)
        
        # PASSO 1: Carregar dataset e extrair features
        print("\n📦 PASSO 1: Carregando dataset e extraindo features...")
        X, y, image_paths, classes = self.loader.load_dataset(max_images_per_class)
        
        # PASSO 2: Preparar dados e treinar modelos
        print("\n🔧 PASSO 2: Preparando dados e treinando modelos...")
        
        print("\n" + "="*70)
        print("🎯 TREINANDO SVM")
        print("="*70)
        
        classifier_svm = FruitClassifier(classes)
        X_train, X_test, y_train, y_test = classifier_svm.prepare_data(X, y)
        
        classifier_svm.train_svm(X_train, y_train)
        y_pred, y_pred_proba, accuracy, report = classifier_svm.evaluate(X_test, y_test)
        
        # Visualizar resultados SVM
        print("\n📊 Visualizando resultados SVM...")
        visualizer_svm = ResultVisualizer(classifier_svm)
        # visualizer_svm.plot_confusion_matrix(y_test, y_pred) # Removido a pedido
        visualizer_svm.plot_classification_metrics(report)
        visualizer_svm.plot_confidence_distribution(y_pred_proba)
        
        self.classifier = classifier_svm
        self.visualizer = visualizer_svm
        
        # PASSO 3: Criar sistema de inspeção
        print("\n" + "="*70)
        print("🔍 CRIANDO SISTEMA DE INSPEÇÃO")
        print("="*70)
        
        self.inspector = FruitInspector(self.classifier, self.feature_extractor)
        
        print("\n✅ Sistema de inspeção criado e pronto para uso!")
        print("\n📝 Use: pipeline.inspector.predict_image('caminho/imagem.jpg')")
        print("📝 Use: pipeline.inspector.batch_inspect([img1, img2, ...])")
        
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETO FINALIZADO!")
        print("="*70)
