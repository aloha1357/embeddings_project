"""
Word2VecTrainer: 负责训练和管理 Word2Vec 模型
遵循单一职责原则 (SRP)
"""
from pathlib import Path
from typing import List, Optional, Set, Tuple
import numpy as np
from gensim.models import Word2Vec
from src.utils.config import Word2VecConfig


class Word2VecTrainer:
    """
    训练和管理 Word2Vec 模型
    
    职责:
    - 训练 Word2Vec 模型
    - 保存和加载模型
    - 获取词向量
    - 查询相似词
    - 计算词类比
    
    遵循 SOLID 原则:
    - SRP: 只负责 Word2Vec 模型相关操作
    - OCP: 通过 config 扩展参数
    - LSP: 可以子类化支持其他嵌入模型
    - DIP: 依赖 Word2VecConfig 抽象
    
    Examples:
        >>> config = Word2VecConfig(vector_size=100, window=5)
        >>> trainer = Word2VecTrainer(config)
        >>> model = trainer.train(data)
        >>> similar = trainer.get_most_similar('king', topn=5)
    """
    
    def __init__(self, config: Word2VecConfig):
        """
        初始化 Word2VecTrainer
        
        Args:
            config: Word2Vec 配置对象
        """
        self._config = config
        self._model: Optional[Word2Vec] = None
    
    def train(self, data: List[List[str]]) -> Word2Vec:
        """
        训练 Word2Vec 模型
        
        Args:
            data: 训练数据，格式为 List[List[str]]（句子列表）
        
        Returns:
            训练好的 Word2Vec 模型
        
        Examples:
            >>> trainer = Word2VecTrainer(config)
            >>> data = [['hello', 'world'], ['test', 'data']]
            >>> model = trainer.train(data)
        """
        self._model = Word2Vec(
            sentences=data,
            vector_size=self._config.vector_size,
            window=self._config.window,
            min_count=self._config.min_count,
            sg=self._config.sg,  # 1 = skipgram, 0 = CBOW
            workers=self._config.workers,
            epochs=self._config.epochs
        )
        
        return self._model
    
    def save_model(self, path: Path) -> None:
        """
        保存模型到文件
        
        Args:
            path: 保存路径
        
        Raises:
            ValueError: 如果模型未训练
        
        Examples:
            >>> trainer.save_model(Path('model.model'))
        """
        if self._model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # 确保目录存在
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        self._model.save(str(path))
    
    def load_model(self, path: Path) -> None:
        """
        从文件加载模型
        
        Args:
            path: 模型文件路径
        
        Raises:
            FileNotFoundError: 如果文件不存在
        
        Examples:
            >>> trainer.load_model(Path('model.model'))
        """
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        self._model = Word2Vec.load(str(path))
    
    def get_vector(self, word: str) -> np.ndarray:
        """
        获取词向量
        
        Args:
            word: 单词
        
        Returns:
            词向量 (numpy array)
        
        Raises:
            ValueError: 如果模型未训练
            KeyError: 如果词不在词汇表中
        
        Examples:
            >>> vector = trainer.get_vector('king')
            >>> print(vector.shape)
            (100,)
        """
        if self._model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self._model.wv[word]
    
    def get_most_similar(
        self, 
        word: str, 
        topn: int = 10
    ) -> List[Tuple[str, float]]:
        """
        获取最相似的词
        
        Args:
            word: 查询词
            topn: 返回的相似词数量
        
        Returns:
            相似词列表，格式为 [(word, similarity_score), ...]
        
        Raises:
            ValueError: 如果模型未训练
            KeyError: 如果词不在词汇表中
        
        Examples:
            >>> similar = trainer.get_most_similar('king', topn=5)
            >>> for word, score in similar:
            ...     print(f"{word}: {score:.3f}")
        """
        if self._model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self._model.wv.most_similar(word, topn=topn)
    
    def compute_analogy(
        self,
        positive: List[str],
        negative: List[str],
        topn: int = 10
    ) -> List[Tuple[str, float]]:
        """
        计算词类比
        
        例如: woman + king - man ≈ queen
        
        Args:
            positive: 正向词列表（加法）
            negative: 负向词列表（减法）
            topn: 返回结果数量
        
        Returns:
            结果词列表，格式为 [(word, similarity_score), ...]
        
        Raises:
            ValueError: 如果模型未训练
            KeyError: 如果词不在词汇表中
        
        Examples:
            >>> # woman + king - man
            >>> results = trainer.compute_analogy(
            ...     positive=['woman', 'king'],
            ...     negative=['man'],
            ...     topn=5
            ... )
            >>> print(results[0])  # 可能是 ('queen', 0.85)
        """
        if self._model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self._model.wv.most_similar(
            positive=positive,
            negative=negative,
            topn=topn
        )
    
    @property
    def vocabulary(self) -> Set[str]:
        """
        获取模型词汇表
        
        Returns:
            词汇表（set）
        
        Raises:
            ValueError: 如果模型未训练
        
        Examples:
            >>> vocab = trainer.vocabulary
            >>> 'king' in vocab
            True
        """
        if self._model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return set(self._model.wv.index_to_key)
    
    @property
    def model(self) -> Optional[Word2Vec]:
        """
        获取底层的 Word2Vec 模型
        
        Returns:
            Word2Vec 模型或 None
        """
        return self._model
    
    def __repr__(self) -> str:
        """字符串表示"""
        if self._model is None:
            return f"Word2VecTrainer(trained=False)"
        else:
            vocab_size = len(self.vocabulary)
            return f"Word2VecTrainer(trained=True, vocab_size={vocab_size}, vector_size={self._config.vector_size})"
