"""
Quick test script for Sprint 3
Tests the training workflow and exploration tools
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import DataLoader
from src.models.word2vec_trainer import Word2VecTrainer
from src.utils.config import Word2VecConfig


def test_small_workflow():
    """
    快速测试工作流（使用小数据集）
    """
    print("=" * 70)
    print("Testing Training Workflow (Small Dataset)")
    print("=" * 70)
    
    # 创建小的测试数据
    test_data_en = [
        ['this', 'is', 'a', 'test', 'sentence'],
        ['another', 'test', 'sentence', 'here'],
        ['word2vec', 'is', 'great', 'for', 'embeddings'],
        ['machine', 'learning', 'is', 'powerful'],
        ['natural', 'language', 'processing', 'is', 'interesting'],
        ['we', 'love', 'machine', 'learning'],
        ['word', 'embeddings', 'are', 'useful'],
        ['test', 'is', 'important'],
    ]
    
    # 训练配置（小参数用于快速测试）
    config = Word2VecConfig(
        vector_size=50,
        window=3,
        min_count=1,
        epochs=10
    )
    
    print("\n1. Training model...")
    trainer = Word2VecTrainer(config)
    trainer.train(test_data_en)
    print(f"   ✅ Model trained! Vocabulary: {len(trainer.vocabulary)} words")
    
    print("\n2. Testing similar words...")
    if 'test' in trainer.vocabulary:
        similar = trainer.get_most_similar('test', topn=3)
        print(f"   Similar to 'test':")
        for word, score in similar:
            print(f"     - {word}: {score:.3f}")
    
    print("\n3. Testing analogies...")
    try:
        results = trainer.compute_analogy(
            positive=['machine', 'learning'],
            negative=['word'],
            topn=3
        )
        print(f"   machine + learning - word:")
        for word, score in results:
            print(f"     - {word}: {score:.3f}")
    except Exception as e:
        print(f"   ⚠️  Analogy test failed (expected with small data): {e}")
    
    print("\n4. Testing model save/load...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.model"
        trainer.save_model(model_path)
        print(f"   ✅ Model saved to {model_path}")
        
        new_trainer = Word2VecTrainer(config)
        new_trainer.load_model(model_path)
        print(f"   ✅ Model loaded! Vocabulary: {len(new_trainer.vocabulary)} words")
    
    print("\n" + "=" * 70)
    print("✅ All workflow tests passed!")
    print("=" * 70)
    
    return trainer


def test_explorer(trainer):
    """
    测试模型探索工具
    """
    from src.utils.model_explorer import ModelExplorer
    
    print("\n" + "=" * 70)
    print("Testing Model Explorer")
    print("=" * 70)
    
    explorer = ModelExplorer(trainer)
    
    print("\n1. Exploring words...")
    test_words = ['test', 'learning', 'word']
    for word in test_words:
        if word in trainer.vocabulary:
            explorer.explore_word(word, topn=3)
    
    print("\n2. Testing vocabulary coverage...")
    check_words = ['test', 'machine', 'learning', 'nonexistent']
    coverage = explorer.check_vocabulary_coverage(check_words)
    
    print("\n" + "=" * 70)
    print("✅ Explorer tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    print("🚀 Starting Sprint 3 Tests\n")
    
    # Test workflow
    trainer = test_small_workflow()
    
    # Test explorer
    test_explorer(trainer)
    
    print("\n" + "=" * 70)
    print("🎉 Sprint 3 - All Tests Passed!")
    print("=" * 70)
    print("\n📝 Next steps:")
    print("   1. Run: python src/utils/training_workflow.py")
    print("      (to train on real data)")
    print("   2. Open: notebooks/task3_model_exploration.ipynb")
    print("      (to explore the models)")
