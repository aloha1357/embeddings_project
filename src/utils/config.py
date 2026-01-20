"""
Configuration management for the embeddings project
Centralized configuration following software engineering best practices
"""
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass
class CorpusConfig:
    """Configuration for corpus data"""
    url: str = "https://data.statmt.org/news-commentary/v18/training/news-commentary-v18.de-en.tsv.gz"
    raw_file_name: str = "news-commentary-v18.de-en.tsv.gz"
    
    @property
    def raw_file_path(self) -> Path:
        return DATA_DIR / self.raw_file_name


@dataclass
class DictionaryConfig:
    """Configuration for bilingual dictionary"""
    url: str = "https://dl.fbaipublicfiles.com/arrival/dictionaries/en-de.txt"
    file_name: str = "en-de.txt"
    
    @property
    def file_path(self) -> Path:
        return DATA_DIR / self.file_name


@dataclass
class Word2VecConfig:
    """Configuration for Word2Vec model training"""
    vector_size: int = 100
    window: int = 5
    min_count: int = 5
    sg: int = 1  # 1 for skipgram, 0 for CBOW
    workers: int = 4
    epochs: int = 5
    
    @property
    def en_model_path(self) -> Path:
        return MODELS_DIR / "word2vec_en.model"
    
    @property
    def de_model_path(self) -> Path:
        return MODELS_DIR / "word2vec_de.model"
    
    @property
    def en_lemma_model_path(self) -> Path:
        return MODELS_DIR / "word2vec_en_lemma.model"
    
    @property
    def de_lemma_model_path(self) -> Path:
        return MODELS_DIR / "word2vec_de_lemma.model"


@dataclass
class ProjectionConfig:
    """Configuration for cross-lingual projection"""
    top_n_words: int = 7000  # Most frequent words to consider
    train_pairs: int = 5000  # Training pairs for projection matrix
    test_pairs: int = 1200   # Test pairs for evaluation
    top_k_eval: int = 5      # Top-K for evaluation
    
    @property
    def projection_matrix_path(self) -> Path:
        return MODELS_DIR / "projection_matrix.npy"


@dataclass
class VisualizationConfig:
    """Configuration for visualization"""
    n_components: int = 2
    figsize: tuple = (12, 8)
    dpi: int = 100


# Singleton instances
corpus_config = CorpusConfig()
dictionary_config = DictionaryConfig()
word2vec_config = Word2VecConfig()
projection_config = ProjectionConfig()
visualization_config = VisualizationConfig()


# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)
