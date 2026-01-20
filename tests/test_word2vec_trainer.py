"""
Test suite for Word2VecTrainer class
Following TDD: Write tests first, then implement
Sprint 2: Task 2 - Train Word2Vec models
"""
import pytest
from pathlib import Path
from gensim.models import Word2Vec
from src.models.word2vec_trainer import Word2VecTrainer
from src.utils.config import Word2VecConfig


@pytest.fixture
def sample_training_data():
    """提供示例训练数据"""
    return [
        ['this', 'is', 'a', 'test', 'sentence'],
        ['another', 'test', 'sentence', 'here'],
        ['word2vec', 'is', 'great', 'for', 'embeddings'],
        ['machine', 'learning', 'is', 'powerful'],
        ['natural', 'language', 'processing', 'is', 'interesting']
    ]


@pytest.fixture
def trained_model(sample_training_data, tmp_path):
    """提供一个已训练的模型"""
    config = Word2VecConfig(
        vector_size=50,  # 小一点以加快测试
        window=3,
        min_count=1,
        epochs=5
    )
    trainer = Word2VecTrainer(config)
    model = trainer.train(sample_training_data)
    return trainer, model


class TestWord2VecTrainer:
    """Test suite for Word2VecTrainer class"""
    
    def test_initialization(self):
        """测试初始化"""
        config = Word2VecConfig(vector_size=100, window=5)
        trainer = Word2VecTrainer(config)
        
        assert trainer._config == config
        assert trainer._model is None
    
    def test_train_model(self, sample_training_data):
        """测试模型训练"""
        config = Word2VecConfig(
            vector_size=50,
            window=3,
            min_count=1,
            epochs=5
        )
        trainer = Word2VecTrainer(config)
        model = trainer.train(sample_training_data)
        
        # 验证模型已训练
        assert model is not None
        assert isinstance(model, Word2Vec)
        assert trainer._model is model
        
        # 验证模型参数
        assert model.vector_size == 50
        assert model.window == 3
        assert model.sg == 1  # skipgram
    
    def test_train_model_creates_vocabulary(self, sample_training_data):
        """测试训练后创建词汇表"""
        config = Word2VecConfig(vector_size=50, window=3, min_count=1)
        trainer = Word2VecTrainer(config)
        trainer.train(sample_training_data)
        
        vocab = trainer.vocabulary
        assert len(vocab) > 0
        assert 'test' in vocab
        assert 'sentence' in vocab
    
    def test_save_and_load_model(self, trained_model, tmp_path):
        """测试模型保存和加载"""
        trainer, model = trained_model
        
        # 保存模型
        save_path = tmp_path / "test_model.model"
        trainer.save_model(save_path)
        
        # 验证文件已创建
        assert save_path.exists()
        
        # 加载模型
        new_trainer = Word2VecTrainer(trainer._config)
        new_trainer.load_model(save_path)
        
        # 验证加载的模型
        assert new_trainer._model is not None
        assert new_trainer._model.vector_size == model.vector_size
        assert len(new_trainer.vocabulary) == len(trainer.vocabulary)
    
    def test_get_vector(self, trained_model):
        """测试获取词向量"""
        trainer, model = trained_model
        
        # 获取存在的词的向量
        vector = trainer.get_vector('test')
        
        assert vector is not None
        assert len(vector) == 50  # vector_size
        assert vector.shape == (50,)
    
    def test_get_vector_unknown_word(self, trained_model):
        """测试获取不存在的词的向量"""
        trainer, model = trained_model
        
        # 不存在的词应该返回 None 或抛出异常
        with pytest.raises(KeyError):
            trainer.get_vector('nonexistent_word_xyz')
    
    def test_get_most_similar(self, trained_model):
        """测试获取相似词"""
        trainer, model = trained_model
        
        similar_words = trainer.get_most_similar('test', topn=3)
        
        # 验证返回格式
        assert isinstance(similar_words, list)
        assert len(similar_words) <= 3
        
        # 验证每个元素是 (word, score) 元组
        for word, score in similar_words:
            assert isinstance(word, str)
            assert isinstance(score, float)
            assert 0 <= score <= 1
    
    def test_get_most_similar_with_nonexistent_word(self, trained_model):
        """测试查询不存在的词的相似词"""
        trainer, model = trained_model
        
        with pytest.raises(KeyError):
            trainer.get_most_similar('nonexistent_word_xyz')
    
    def test_compute_analogy(self, trained_model):
        """测试词类比计算"""
        trainer, model = trained_model
        
        # 由于训练数据小，可能无法找到好的类比
        # 但至少应该能执行不报错
        try:
            results = trainer.compute_analogy(
                positive=['sentence', 'test'],
                negative=['is'],
                topn=3
            )
            
            assert isinstance(results, list)
            assert len(results) <= 3
            
            for word, score in results:
                assert isinstance(word, str)
                assert isinstance(score, float)
        except KeyError:
            # 如果词不在词汇表中，允许抛出 KeyError
            pass
    
    def test_vocabulary_property(self, trained_model):
        """测试 vocabulary 属性"""
        trainer, model = trained_model
        
        vocab = trainer.vocabulary
        
        assert isinstance(vocab, set)
        assert len(vocab) > 0
        assert 'test' in vocab
    
    def test_model_not_trained_error(self):
        """测试未训练模型时的错误处理"""
        config = Word2VecConfig()
        trainer = Word2VecTrainer(config)
        
        # 未训练模型时，这些操作应该抛出错误
        with pytest.raises(ValueError):
            trainer.get_vector('test')
        
        with pytest.raises(ValueError):
            trainer.get_most_similar('test')
        
        with pytest.raises(ValueError):
            _ = trainer.vocabulary


class TestWord2VecTrainerIntegration:
    """集成测试：完整的训练-保存-加载流程"""
    
    def test_full_workflow(self, sample_training_data, tmp_path):
        """测试完整工作流程"""
        # 1. 训练模型
        config = Word2VecConfig(vector_size=50, window=3, min_count=1, epochs=5)
        trainer = Word2VecTrainer(config)
        model = trainer.train(sample_training_data)
        
        # 2. 查询相似词
        similar = trainer.get_most_similar('test', topn=2)
        assert len(similar) > 0
        
        # 3. 获取词向量
        vector = trainer.get_vector('test')
        assert vector.shape == (50,)
        
        # 4. 保存模型
        model_path = tmp_path / "workflow_model.model"
        trainer.save_model(model_path)
        
        # 5. 加载模型
        new_trainer = Word2VecTrainer(config)
        new_trainer.load_model(model_path)
        
        # 6. 验证加载后的功能
        new_similar = new_trainer.get_most_similar('test', topn=2)
        assert len(new_similar) > 0
        
        new_vector = new_trainer.get_vector('test')
        assert new_vector.shape == (50,)
        
        # 词向量应该相同（或非常接近）
        import numpy as np
        assert np.allclose(vector, new_vector)


class TestWord2VecTrainerConfiguration:
    """测试不同配置"""
    
    def test_different_vector_sizes(self, sample_training_data):
        """测试不同的向量维度"""
        for size in [10, 50, 100]:
            config = Word2VecConfig(vector_size=size, window=3, min_count=1, epochs=3)
            trainer = Word2VecTrainer(config)
            trainer.train(sample_training_data)
            
            vector = trainer.get_vector('test')
            assert vector.shape == (size,)
    
    def test_different_window_sizes(self, sample_training_data):
        """测试不同的窗口大小"""
        for window in [2, 5, 10]:
            config = Word2VecConfig(vector_size=50, window=window, min_count=1, epochs=3)
            trainer = Word2VecTrainer(config)
            model = trainer.train(sample_training_data)
            
            assert model.window == window
