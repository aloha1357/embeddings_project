"""
Test suite for DataLoader class
Following TDD: Write tests first, then implement
"""
import pytest
from pathlib import Path
from src.data.data_loader import DataLoader


class TestDataLoader:
    """Test suite for DataLoader class"""
    
    def test_tokenize_simple_sentence(self):
        """测试基本的 tokenization 功能"""
        loader = DataLoader()
        text = "Hello, world! This is a test."
        tokens = loader.tokenize(text)
        
        # 验证返回类型
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        
        # 验证已转换为小写
        assert all(token.islower() or not token.isalpha() for token in tokens)
        
        # 验证包含预期的词
        assert 'hello' in tokens
        assert 'world' in tokens
        assert 'test' in tokens
    
    def test_tokenize_empty_string(self):
        """测试空字符串的边界情况"""
        loader = DataLoader()
        tokens = loader.tokenize("")
        assert tokens == []
    
    def test_tokenize_with_numbers(self):
        """测试包含数字的文本"""
        loader = DataLoader()
        text = "There are 123 items in 2024"
        tokens = loader.tokenize(text)
        
        assert '123' in tokens
        assert '2024' in tokens
    
    def test_preprocess_lowercase(self):
        """测试 lowercase 预处理"""
        loader = DataLoader()
        sentences = ["Hello World", "TESTING Python"]
        result = loader.preprocess(sentences, lowercase=True, lemmatize=False)
        
        # 验证结构
        assert len(result) == 2
        assert all(isinstance(sent, list) for sent in result)
        
        # 验证所有词都是小写
        for sentence in result:
            for word in sentence:
                if word.isalpha():
                    assert word.islower(), f"Word '{word}' is not lowercase"
    
    def test_preprocess_maintains_sentence_structure(self):
        """测试预处理保持句子结构"""
        loader = DataLoader()
        sentences = ["First sentence here.", "Second sentence here."]
        result = loader.preprocess(sentences, lowercase=True, lemmatize=False)
        
        assert len(result) == 2
        assert len(result[0]) > 0
        assert len(result[1]) > 0
    
    def test_load_parallel_corpus(self, tmp_path):
        """测试加载平行语料库"""
        # 创建临时测试文件
        test_file = tmp_path / "test_corpus.tsv"
        test_content = "Hello world\tHallo Welt\nGood morning\tGuten Morgen\n"
        test_file.write_text(test_content, encoding='utf-8')
        
        loader = DataLoader()
        en_data, de_data = loader.load_parallel_corpus(test_file)
        
        # 验证数据结构
        assert len(en_data) == 2
        assert len(de_data) == 2
        
        # 验证每行都是 token 列表
        assert isinstance(en_data[0], list)
        assert isinstance(de_data[0], list)
        
        # 验证内容
        assert 'hello' in en_data[0]
        assert 'hallo' in de_data[0]
    
    def test_load_parallel_corpus_handles_malformed_lines(self, tmp_path):
        """测试处理格式错误的行"""
        test_file = tmp_path / "test_corpus.tsv"
        # 包含一些格式错误的行
        test_content = "Hello world\tHallo Welt\nBad line without tab\nGood line\tGute Zeile\n"
        test_file.write_text(test_content, encoding='utf-8')
        
        loader = DataLoader()
        en_data, de_data = loader.load_parallel_corpus(test_file)
        
        # 应该只加载格式正确的行
        assert len(en_data) == 2
        assert len(de_data) == 2


class TestDataLoaderWithLemmatization:
    """测试带词形还原的功能（Sprint 9 会实现）"""
    
    @pytest.mark.skip(reason="Lemmatization feature will be implemented in Sprint 9")
    def test_preprocess_with_lemmatization(self):
        """测试词形还原功能"""
        loader = DataLoader(lemmatize=True)
        sentences = ["running runs ran", "better best good"]
        result = loader.preprocess(sentences, lowercase=True, lemmatize=True)
        
        # 这个测试现在会跳过，Sprint 9 时实现
        pass
