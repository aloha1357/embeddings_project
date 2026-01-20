"""
BilingualDictionary: 管理双语词典和创建词对
遵循单一职责原则 (SRP)
"""
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import random


class BilingualDictionary:
    """
    双语词典管理
    
    职责:
    - 加载双语词典
    - 查询翻译
    - 创建词对（过滤词汇表）
    - 分割训练/测试集
    
    遵循 SOLID 原则:
    - SRP: 只负责词典管理
    - OCP: 通过参数扩展功能
    - DIP: 依赖抽象接口
    
    Examples:
        >>> bilingual_dict = BilingualDictionary()
        >>> bilingual_dict.load_dictionary(Path('en-de.txt'))
        >>> translation = bilingual_dict.get_translation('apple')
        >>> print(translation)
        'Apfel'
    """
    
    def __init__(self):
        """初始化双语词典"""
        self._dict_data: Optional[Dict[str, List[str]]] = None
    
    def load_dictionary(self, file_path: Path) -> None:
        """
        加载双语词典
        
        格式: 每行为 "source_word\ttarget_word"
        支持一个源词有多个翻译
        
        Args:
            file_path: 词典文件路径
        
        Raises:
            FileNotFoundError: 如果文件不存在
        
        Examples:
            >>> bilingual_dict.load_dictionary(Path('en-de.txt'))
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Dictionary file not found: {file_path}")
        
        self._dict_data = {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) != 2:
                    continue
                
                source_word, target_word = parts
                source_word = source_word.lower()
                target_word = target_word.lower()
                
                # 支持多个翻译
                if source_word not in self._dict_data:
                    self._dict_data[source_word] = []
                
                if target_word not in self._dict_data[source_word]:
                    self._dict_data[source_word].append(target_word)
    
    def get_translation(self, word: str) -> Optional[str]:
        """
        获取词的翻译
        
        如果有多个翻译，返回第一个
        
        Args:
            word: 源语言词
        
        Returns:
            翻译或 None（如果不存在）
        
        Examples:
            >>> translation = bilingual_dict.get_translation('apple')
            >>> print(translation)
            'apfel'
        """
        if self._dict_data is None:
            raise ValueError("Dictionary not loaded. Call load_dictionary() first.")
        
        word = word.lower()
        
        if word not in self._dict_data:
            return None
        
        # 返回第一个翻译
        translations = self._dict_data[word]
        return translations[0] if translations else None
    
    def get_all_translations(self, word: str) -> List[str]:
        """
        获取词的所有翻译
        
        Args:
            word: 源语言词
        
        Returns:
            翻译列表（可能为空）
        
        Examples:
            >>> translations = bilingual_dict.get_all_translations('apple')
            >>> print(translations)
            ['apfel', 'äpfel']
        """
        if self._dict_data is None:
            raise ValueError("Dictionary not loaded. Call load_dictionary() first.")
        
        word = word.lower()
        return self._dict_data.get(word, [])
    
    def create_word_pairs(
        self,
        source_words: List[str],
        target_vocab: Set[str]
    ) -> List[Tuple[str, str]]:
        """
        创建词对，只保留目标词在目标词汇表中的配对
        
        Args:
            source_words: 源语言词列表（通常是按频率排序的）
            target_vocab: 目标语言词汇表（用于过滤）
        
        Returns:
            词对列表 [(source_word, target_word), ...]
        
        Examples:
            >>> word_pairs = bilingual_dict.create_word_pairs(
            ...     source_words=['apple', 'car', 'house'],
            ...     target_vocab={'apfel', 'auto', 'haus'}
            ... )
            >>> print(len(word_pairs))
            3
        """
        if self._dict_data is None:
            raise ValueError("Dictionary not loaded. Call load_dictionary() first.")
        
        word_pairs = []
        
        for source_word in source_words:
            source_word_lower = source_word.lower()
            
            # 查找翻译
            if source_word_lower not in self._dict_data:
                continue
            
            translations = self._dict_data[source_word_lower]
            
            # 查找在目标词汇表中的翻译
            for translation in translations:
                if translation in target_vocab:
                    word_pairs.append((source_word_lower, translation))
                    break  # 只取第一个有效翻译
        
        return word_pairs
    
    def split_train_test(
        self,
        word_pairs: List[Tuple[str, str]],
        train_size: int
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        分割词对为训练集和测试集
        
        前 train_size 个作为训练，其余作为测试
        
        Args:
            word_pairs: 词对列表
            train_size: 训练集大小
        
        Returns:
            (训练集, 测试集) 元组
        
        Examples:
            >>> train, test = bilingual_dict.split_train_test(
            ...     word_pairs=all_pairs,
            ...     train_size=5000
            ... )
            >>> print(len(train), len(test))
            5000 1200
        """
        # 确保不超过总数
        train_size = min(train_size, len(word_pairs))
        
        train_pairs = word_pairs[:train_size]
        test_pairs = word_pairs[train_size:]
        
        return train_pairs, test_pairs
    
    def download_dictionary(self, url: str, save_path: Path) -> Path:
        """
        下载词典文件
        
        Args:
            url: 下载 URL
            save_path: 保存路径
        
        Returns:
            保存的文件路径
        
        Examples:
            >>> path = bilingual_dict.download_dictionary(
            ...     url='https://example.com/en-de.txt',
            ...     save_path=Path('data/en-de.txt')
            ... )
        """
        import requests
        from tqdm import tqdm
        
        # 确保目录存在
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 下载文件
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as f:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading') as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                # 没有 content-length，直接下载
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        return save_path
    
    @property
    def vocabulary_size(self) -> int:
        """
        获取词典中的词汇数量
        
        Returns:
            词汇数量
        """
        if self._dict_data is None:
            return 0
        return len(self._dict_data)
    
    def __repr__(self) -> str:
        """字符串表示"""
        if self._dict_data is None:
            return "BilingualDictionary(loaded=False)"
        else:
            return f"BilingualDictionary(loaded=True, size={self.vocabulary_size})"
