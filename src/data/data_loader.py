"""
DataLoader: 负责加载和预处理语料库数据
遵循单一职责原则 (SRP)
"""
from pathlib import Path
from typing import List, Tuple, Optional
import re
from nltk.tokenize import word_tokenize


class DataLoader:
    """
    加载和预处理语料库数据
    
    职责:
    - 下载语料库
    - 加载平行语料库
    - Tokenization
    - 预处理（lowercase, lemmatization）
    
    遵循 SOLID 原则:
    - SRP: 只负责数据加载和预处理
    - OCP: 通过参数扩展功能
    - DIP: 依赖 nltk 接口而非具体实现
    """
    
    def __init__(self, lemmatize: bool = False):
        """
        初始化 DataLoader
        
        Args:
            lemmatize: 是否启用词形还原（Sprint 9 实现）
        """
        self._lemmatize = lemmatize
        self._lemmatizer = None
        
        # Sprint 9 会实现 lemmatization
        if lemmatize:
            from nltk.stem import WordNetLemmatizer
            self._lemmatizer = WordNetLemmatizer()
    
    def tokenize(self, text: str) -> List[str]:
        """
        对文本进行 tokenization
        
        Args:
            text: 输入文本
        
        Returns:
            token 列表（小写）
        
        Examples:
            >>> loader = DataLoader()
            >>> loader.tokenize("Hello, world!")
            ['hello', ',', 'world', '!']
        """
        if not text or text.strip() == "":
            return []
        
        # 使用 NLTK 的 word_tokenize
        tokens = word_tokenize(text.lower())
        
        # 保留所有 tokens（包括标点和数字）
        return tokens
    
    def preprocess(
        self, 
        sentences: List[str], 
        lowercase: bool = True, 
        lemmatize: bool = False
    ) -> List[List[str]]:
        """
        预处理句子列表
        
        Args:
            sentences: 句子列表
            lowercase: 是否转小写
            lemmatize: 是否词形还原（Sprint 9 实现）
        
        Returns:
            处理后的 token 列表的列表
        
        Examples:
            >>> loader = DataLoader()
            >>> loader.preprocess(["Hello World"], lowercase=True)
            [['hello', 'world']]
        """
        result = []
        
        for sentence in sentences:
            if lowercase:
                tokens = self.tokenize(sentence)
            else:
                tokens = word_tokenize(sentence)
            
            # Sprint 9: 实现 lemmatization
            if lemmatize and self._lemmatizer:
                tokens = [self._lemmatizer.lemmatize(token) for token in tokens]
            
            result.append(tokens)
        
        return result
    
    def load_parallel_corpus(
        self, 
        file_path: Path
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """
        加载平行语料库（TSV格式：英语\\t德语）
        
        Args:
            file_path: TSV 文件路径
        
        Returns:
            (英语数据, 德语数据) 元组，每个都是 token 列表的列表
        
        Examples:
            >>> loader = DataLoader()
            >>> en, de = loader.load_parallel_corpus(Path("corpus.tsv"))
        
        Raises:
            FileNotFoundError: 文件不存在
        """
        en_data = []
        de_data = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                
                # 只处理格式正确的行（包含恰好一个 tab）
                if len(parts) == 2:
                    en_text, de_text = parts
                    en_data.append(self.tokenize(en_text))
                    de_data.append(self.tokenize(de_text))
        
        return en_data, de_data
    
    def download_corpus(self, url: str, save_path: Path) -> Path:
        """
        下载语料库文件
        
        Args:
            url: 下载 URL
            save_path: 保存路径
        
        Returns:
            保存的文件路径
        
        Note:
            这个方法会在实际使用时实现，现在先提供接口
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
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return save_path
