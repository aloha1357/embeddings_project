"""
Test suite for BilingualDictionary class
Following TDD: Write tests first, then implement
Sprint 4: Task 4 - Prepare bilingual dictionary and word pairs
"""
import pytest
from pathlib import Path
from src.data.bilingual_dict import BilingualDictionary
from src.models.word2vec_trainer import Word2VecTrainer
from src.utils.config import Word2VecConfig


@pytest.fixture
def sample_dict_file(tmp_path):
    """创建示例词典文件"""
    dict_file = tmp_path / "en-de.txt"
    content = """apple	Apfel
apple	Äpfel
car	Auto
car	Wagen
house	Haus
book	Buch
table	Tisch
chair	Stuhl
window	Fenster
door	Tür
computer	Computer
phone	Telefon
water	Wasser
"""
    dict_file.write_text(content, encoding='utf-8')
    return dict_file


@pytest.fixture
def sample_vocabularies():
    """创建示例词汇表"""
    en_vocab = {
        'apple', 'car', 'house', 'book', 'table', 
        'chair', 'window', 'door', 'unknown_word'
    }
    de_vocab = {
        'apfel', 'auto', 'haus', 'buch', 'tisch',
        'stuhl', 'fenster', 'tür', 'wasser'
    }
    return en_vocab, de_vocab


class TestBilingualDictionary:
    """Test suite for BilingualDictionary class"""
    
    def test_initialization(self):
        """测试初始化"""
        bilingual_dict = BilingualDictionary()
        assert bilingual_dict._dict_data is None
    
    def test_load_dictionary(self, sample_dict_file):
        """测试加载词典"""
        bilingual_dict = BilingualDictionary()
        bilingual_dict.load_dictionary(sample_dict_file)
        
        # 验证词典已加载
        assert bilingual_dict._dict_data is not None
        assert len(bilingual_dict._dict_data) > 0
        
        # 验证数据结构
        assert 'apple' in bilingual_dict._dict_data
        assert isinstance(bilingual_dict._dict_data['apple'], list)
    
    def test_load_dictionary_multiple_translations(self, sample_dict_file):
        """测试处理多个翻译"""
        bilingual_dict = BilingualDictionary()
        bilingual_dict.load_dictionary(sample_dict_file)
        
        # apple 有两个翻译
        assert len(bilingual_dict._dict_data['apple']) >= 1
        assert 'apfel' in bilingual_dict._dict_data['apple']
    
    def test_get_translation(self, sample_dict_file):
        """测试获取翻译"""
        bilingual_dict = BilingualDictionary()
        bilingual_dict.load_dictionary(sample_dict_file)
        
        translation = bilingual_dict.get_translation('apple')
        assert translation is not None
        assert translation.lower() == 'apfel' or translation.lower() == 'äpfel'
    
    def test_get_translation_not_found(self, sample_dict_file):
        """测试获取不存在的词的翻译"""
        bilingual_dict = BilingualDictionary()
        bilingual_dict.load_dictionary(sample_dict_file)
        
        translation = bilingual_dict.get_translation('nonexistent')
        assert translation is None
    
    def test_create_word_pairs(self, sample_dict_file, sample_vocabularies):
        """测试创建词对"""
        en_vocab, de_vocab = sample_vocabularies
        
        bilingual_dict = BilingualDictionary()
        bilingual_dict.load_dictionary(sample_dict_file)
        
        # 从前10个最常见词创建词对
        word_pairs = bilingual_dict.create_word_pairs(
            source_words=list(en_vocab)[:10],
            target_vocab=de_vocab
        )
        
        # 验证返回类型
        assert isinstance(word_pairs, list)
        
        # 验证词对格式
        for en_word, de_word in word_pairs:
            assert isinstance(en_word, str)
            assert isinstance(de_word, str)
            assert en_word in en_vocab
            assert de_word in de_vocab
    
    def test_create_word_pairs_filters_by_target_vocab(
        self, 
        sample_dict_file, 
        sample_vocabularies
    ):
        """测试词对创建时过滤目标词汇表"""
        en_vocab, de_vocab = sample_vocabularies
        
        # 限制德语词汇表
        limited_de_vocab = {'apfel', 'auto', 'haus'}
        
        bilingual_dict = BilingualDictionary()
        bilingual_dict.load_dictionary(sample_dict_file)
        
        word_pairs = bilingual_dict.create_word_pairs(
            source_words=['apple', 'car', 'house', 'computer'],
            target_vocab=limited_de_vocab
        )
        
        # 验证所有德语词都在限制的词汇表中
        for en_word, de_word in word_pairs:
            assert de_word in limited_de_vocab
    
    def test_split_train_test(self, sample_dict_file, sample_vocabularies):
        """测试训练/测试集分割"""
        en_vocab, de_vocab = sample_vocabularies
        
        bilingual_dict = BilingualDictionary()
        bilingual_dict.load_dictionary(sample_dict_file)
        
        # 创建词对
        word_pairs = bilingual_dict.create_word_pairs(
            source_words=list(en_vocab),
            target_vocab=de_vocab
        )
        
        # 分割
        train_size = 5
        train_pairs, test_pairs = bilingual_dict.split_train_test(
            word_pairs, 
            train_size=train_size
        )
        
        # 验证分割
        assert len(train_pairs) == min(train_size, len(word_pairs))
        assert len(test_pairs) == len(word_pairs) - len(train_pairs)
        
        # 验证没有重叠
        train_en = {en for en, de in train_pairs}
        test_en = {en for en, de in test_pairs}
        assert len(train_en & test_en) == 0
    
    def test_split_train_test_order(self, sample_dict_file, sample_vocabularies):
        """测试分割保持顺序（前N个作为训练）"""
        en_vocab, de_vocab = sample_vocabularies
        
        bilingual_dict = BilingualDictionary()
        bilingual_dict.load_dictionary(sample_dict_file)
        
        word_pairs = bilingual_dict.create_word_pairs(
            source_words=list(en_vocab),
            target_vocab=de_vocab
        )
        
        train_size = 3
        train_pairs, test_pairs = bilingual_dict.split_train_test(
            word_pairs,
            train_size=train_size
        )
        
        # 训练集应该是前N个
        assert train_pairs == word_pairs[:train_size]
        assert test_pairs == word_pairs[train_size:]
    
    def test_download_dictionary(self, tmp_path):
        """测试下载词典"""
        bilingual_dict = BilingualDictionary()
        save_path = tmp_path / "downloaded_dict.txt"
        
        # 测试下载接口（不实际下载）
        # 这个方法在实际使用时会实现
        assert hasattr(bilingual_dict, 'download_dictionary')


class TestBilingualDictionaryIntegration:
    """集成测试：完整工作流"""
    
    def test_full_workflow(self, sample_dict_file, sample_vocabularies):
        """测试完整工作流程"""
        en_vocab, de_vocab = sample_vocabularies
        
        # 1. 加载词典
        bilingual_dict = BilingualDictionary()
        bilingual_dict.load_dictionary(sample_dict_file)
        
        # 2. 创建词对
        word_pairs = bilingual_dict.create_word_pairs(
            source_words=list(en_vocab),
            target_vocab=de_vocab
        )
        
        assert len(word_pairs) > 0
        
        # 3. 分割训练/测试
        train_pairs, test_pairs = bilingual_dict.split_train_test(
            word_pairs,
            train_size=5
        )
        
        assert len(train_pairs) == 5
        assert len(test_pairs) > 0
        
        # 4. 验证结果
        for en_word, de_word in train_pairs + test_pairs:
            assert en_word in en_vocab
            assert de_word in de_vocab


class TestBilingualDictionaryEdgeCases:
    """边界情况测试"""
    
    def test_empty_source_words(self, sample_dict_file, sample_vocabularies):
        """测试空的源词列表"""
        _, de_vocab = sample_vocabularies
        
        bilingual_dict = BilingualDictionary()
        bilingual_dict.load_dictionary(sample_dict_file)
        
        word_pairs = bilingual_dict.create_word_pairs(
            source_words=[],
            target_vocab=de_vocab
        )
        
        assert word_pairs == []
    
    def test_no_valid_translations(self, sample_dict_file):
        """测试没有有效翻译的情况"""
        bilingual_dict = BilingualDictionary()
        bilingual_dict.load_dictionary(sample_dict_file)
        
        # 目标词汇表中没有任何翻译
        word_pairs = bilingual_dict.create_word_pairs(
            source_words=['apple', 'car'],
            target_vocab={'xyz', 'abc'}  # 不存在的词
        )
        
        assert word_pairs == []
    
    def test_train_size_larger_than_pairs(self, sample_dict_file, sample_vocabularies):
        """测试训练集大小超过总词对数"""
        en_vocab, de_vocab = sample_vocabularies
        
        bilingual_dict = BilingualDictionary()
        bilingual_dict.load_dictionary(sample_dict_file)
        
        word_pairs = bilingual_dict.create_word_pairs(
            source_words=list(en_vocab)[:3],
            target_vocab=de_vocab
        )
        
        train_pairs, test_pairs = bilingual_dict.split_train_test(
            word_pairs,
            train_size=100  # 比总数大
        )
        
        # 所有词对应该在训练集中
        assert len(train_pairs) == len(word_pairs)
        assert len(test_pairs) == 0
