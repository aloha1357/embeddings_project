"""
Tests for TranslationEvaluator class.

TDD approach: Write tests first (Red), then implement (Green), then refactor.
"""

import pytest
import numpy as np
from pathlib import Path

from src.evaluation.translation_evaluator import TranslationEvaluator
from src.models.word2vec_trainer import Word2VecTrainer
from src.models.embedding_projector import EmbeddingProjector
from src.utils.config import Word2VecConfig


class TestTranslationEvaluator:
    """Test cases for TranslationEvaluator initialization and basic functionality."""
    
    def test_initialization(self):
        """Test that TranslationEvaluator can be initialized."""
        evaluator = TranslationEvaluator()
        assert evaluator is not None
    
    def test_compute_precision_at_k(self):
        """Test computing P@k for a single source-target pair."""
        # Create a simple scenario with known embeddings
        np.random.seed(42)
        
        # Create target embeddings where we control similarity
        target_embeddings = np.random.randn(10, 50).astype(np.float32)
        target_words = [f"word_{i}" for i in range(10)]
        
        # Projected vector very similar to target_embeddings[3]
        projected_vector = target_embeddings[3] + np.random.randn(50).astype(np.float32) * 0.01
        correct_translation = "word_3"
        
        evaluator = TranslationEvaluator()
        precision = evaluator.compute_precision_at_k(
            projected_vector=projected_vector,
            correct_translation=correct_translation,
            target_embeddings=target_embeddings,
            target_words=target_words,
            k=5
        )
        
        # Should find correct translation in top 5
        assert precision == 1.0
    
    def test_compute_precision_at_k_not_found(self):
        """Test P@k when correct translation is not in top-k."""
        np.random.seed(42)
        
        target_embeddings = np.random.randn(10, 50).astype(np.float32)
        target_words = [f"word_{i}" for i in range(10)]
        
        # Projected vector very similar to target_embeddings[9] (last one)
        projected_vector = target_embeddings[9] + np.random.randn(50).astype(np.float32) * 0.01
        correct_translation = "word_0"  # But correct is word_0
        
        evaluator = TranslationEvaluator()
        precision = evaluator.compute_precision_at_k(
            projected_vector=projected_vector,
            correct_translation=correct_translation,
            target_embeddings=target_embeddings,
            target_words=target_words,
            k=3  # Only check top 3
        )
        
        # Should not find correct translation in top 3
        assert precision == 0.0
    
    def test_evaluate_test_pairs(self):
        """Test evaluating multiple test pairs."""
        np.random.seed(42)
        
        # Create test scenario
        n_words = 20
        target_embeddings = np.random.randn(n_words, 50).astype(np.float32)
        target_words = [f"target_{i}" for i in range(n_words)]
        
        # Create test pairs where projected vectors are close to their targets
        test_pairs = []
        for i in range(5):
            source_word = f"source_{i}"
            target_word = f"target_{i}"
            # Projected vector similar to target i
            projected = target_embeddings[i] + np.random.randn(50).astype(np.float32) * 0.1
            test_pairs.append((source_word, target_word, projected))
        
        evaluator = TranslationEvaluator()
        results = evaluator.evaluate_test_pairs(
            test_pairs=test_pairs,
            target_embeddings=target_embeddings,
            target_words=target_words,
            k_values=[1, 5, 10]
        )
        
        assert "p@1" in results
        assert "p@5" in results
        assert "p@10" in results
        assert 0.0 <= results["p@1"] <= 1.0
        assert 0.0 <= results["p@5"] <= 1.0
        assert 0.0 <= results["p@10"] <= 1.0
        # P@5 should be >= P@1 (larger k can only increase or maintain precision)
        assert results["p@5"] >= results["p@1"]
    
    def test_evaluate_with_projector(self):
        """Test evaluation using EmbeddingProjector integration."""
        # Create simple models
        source_sentences = [
            ["cat", "dog", "animal"],
            ["king", "queen", "royal"],
            ["man", "woman", "person"]
        ]
        target_sentences = [
            ["katze", "hund", "tier"],
            ["könig", "königin", "königlich"],
            ["mann", "frau", "person"]
        ]
        
        config = Word2VecConfig(vector_size=30, window=2, min_count=1, epochs=10)
        
        source_trainer = Word2VecTrainer(config)
        source_trainer.train(source_sentences)
        
        target_trainer = Word2VecTrainer(config)
        target_trainer.train(target_sentences)
        
        # Train projection
        train_pairs = [("cat", "katze"), ("king", "könig")]
        projector = EmbeddingProjector()
        X_source, Z_target = projector.extract_vectors(
            train_pairs, source_trainer.model, target_trainer.model
        )
        projector.learn_projection(X_source, Z_target)
        
        # Test pairs
        test_pairs = [("dog", "hund"), ("queen", "königin")]
        
        evaluator = TranslationEvaluator()
        results = evaluator.evaluate_with_projector(
            test_pairs=test_pairs,
            source_model=source_trainer.model,
            target_model=target_trainer.model,
            projector=projector,
            k_values=[1, 3, 5]
        )
        
        assert "p@1" in results
        assert "p@3" in results
        assert "p@5" in results
        assert "total_pairs" in results
        assert results["total_pairs"] == 2


class TestTranslationEvaluatorIntegration:
    """Integration tests for TranslationEvaluator."""
    
    def test_full_evaluation_workflow(self):
        """Test complete workflow: train -> project -> evaluate."""
        # Create synthetic data with known relationships
        np.random.seed(42)
        
        # English sentences
        en_sentences = [
            ["hello", "world", "good"],
            ["hello", "there", "nice"],
            ["world", "peace", "great"],
            ["good", "morning", "wonderful"]
        ]
        
        # German sentences (using similar contexts)
        de_sentences = [
            ["hallo", "welt", "gut"],
            ["hallo", "da", "schön"],
            ["welt", "frieden", "großartig"],
            ["gut", "morgen", "wunderbar"]
        ]
        
        config = Word2VecConfig(vector_size=40, window=2, min_count=1, epochs=20)
        
        en_trainer = Word2VecTrainer(config)
        en_trainer.train(en_sentences)
        
        de_trainer = Word2VecTrainer(config)
        de_trainer.train(de_sentences)
        
        # Training pairs
        train_pairs = [("hello", "hallo"), ("world", "welt")]
        
        projector = EmbeddingProjector()
        X, Z = projector.extract_vectors(train_pairs, en_trainer.model, de_trainer.model)
        projector.learn_projection(X, Z)
        
        # Test pairs
        test_pairs = [("good", "gut"), ("there", "da")]
        
        evaluator = TranslationEvaluator()
        results = evaluator.evaluate_with_projector(
            test_pairs=test_pairs,
            source_model=en_trainer.model,
            target_model=de_trainer.model,
            projector=projector,
            k_values=[1, 3, 5]
        )
        
        # Verify structure
        assert isinstance(results, dict)
        assert all(k in results for k in ["p@1", "p@3", "p@5", "total_pairs"])
        assert results["total_pairs"] == 2
    
    def test_evaluation_with_larger_vocabulary(self):
        """Test evaluation with larger vocabulary (more realistic scenario)."""
        np.random.seed(42)
        
        # Create larger vocabulary
        vocab_size = 50
        source_words = [f"en_word_{i}" for i in range(vocab_size)]
        target_words = [f"de_word_{i}" for i in range(vocab_size)]
        
        # Create synthetic embeddings
        source_embeddings = np.random.randn(vocab_size, 30).astype(np.float32)
        target_embeddings = np.random.randn(vocab_size, 30).astype(np.float32)
        
        # Learn a projection (even if random, for testing)
        projector = EmbeddingProjector()
        projector.learn_projection(source_embeddings[:30], target_embeddings[:30])
        
        # Create test pairs (use the last 20 as test)
        test_pairs = []
        for i in range(30, 40):
            source_vec = source_embeddings[i]
            projected_vec = projector.project(source_vec.reshape(1, -1))[0]
            test_pairs.append((source_words[i], target_words[i], projected_vec))
        
        evaluator = TranslationEvaluator()
        results = evaluator.evaluate_test_pairs(
            test_pairs=test_pairs,
            target_embeddings=target_embeddings,
            target_words=target_words,
            k_values=[1, 5, 10]
        )
        
        assert results["total_pairs"] == 10
        # With random embeddings, P@1 should be very low
        assert 0.0 <= results["p@1"] <= 0.5


class TestTranslationEvaluatorEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_test_pairs(self):
        """Test evaluation with empty test pairs."""
        target_embeddings = np.random.randn(10, 30).astype(np.float32)
        target_words = [f"word_{i}" for i in range(10)]
        
        evaluator = TranslationEvaluator()
        results = evaluator.evaluate_test_pairs(
            test_pairs=[],
            target_embeddings=target_embeddings,
            target_words=target_words,
            k_values=[1, 5]
        )
        
        assert results["p@1"] == 0.0
        assert results["p@5"] == 0.0
        assert results["total_pairs"] == 0
    
    def test_k_larger_than_vocabulary(self):
        """Test when k is larger than vocabulary size."""
        np.random.seed(42)
        
        target_embeddings = np.random.randn(5, 30).astype(np.float32)
        target_words = [f"word_{i}" for i in range(5)]
        
        # Projected vector similar to word_2
        projected = target_embeddings[2] + np.random.randn(30).astype(np.float32) * 0.01
        
        evaluator = TranslationEvaluator()
        precision = evaluator.compute_precision_at_k(
            projected_vector=projected,
            correct_translation="word_2",
            target_embeddings=target_embeddings,
            target_words=target_words,
            k=100  # Much larger than vocab size
        )
        
        # Should still find it (k is clamped to vocab size)
        assert precision == 1.0
    
    def test_correct_translation_not_in_vocabulary(self):
        """Test when correct translation is not in target vocabulary."""
        np.random.seed(42)
        
        target_embeddings = np.random.randn(10, 30).astype(np.float32)
        target_words = [f"word_{i}" for i in range(10)]
        
        projected = target_embeddings[0] + np.random.randn(30).astype(np.float32) * 0.01
        
        evaluator = TranslationEvaluator()
        precision = evaluator.compute_precision_at_k(
            projected_vector=projected,
            correct_translation="missing_word",  # Not in vocabulary
            target_embeddings=target_embeddings,
            target_words=target_words,
            k=5
        )
        
        # Cannot find missing word
        assert precision == 0.0
    
    def test_multiple_k_values(self):
        """Test evaluation with multiple k values."""
        np.random.seed(42)
        
        target_embeddings = np.random.randn(20, 30).astype(np.float32)
        target_words = [f"word_{i}" for i in range(20)]
        
        # Create test pair where target is at position 7 (when sorted by similarity)
        test_pairs = [(
            "source",
            "word_7",
            target_embeddings[7] + np.random.randn(30).astype(np.float32) * 0.1
        )]
        
        evaluator = TranslationEvaluator()
        results = evaluator.evaluate_test_pairs(
            test_pairs=test_pairs,
            target_embeddings=target_embeddings,
            target_words=target_words,
            k_values=[1, 3, 5, 10]
        )
        
        # P@10 should likely be 1.0 (target is within top 10)
        # P@1 might be 0.0 depending on noise
        assert "p@1" in results
        assert "p@3" in results
        assert "p@5" in results
        assert "p@10" in results
        assert results["total_pairs"] == 1
    
    def test_identical_embeddings(self):
        """Test with identical projected and target embedding."""
        target_embeddings = np.random.randn(10, 30).astype(np.float32)
        target_words = [f"word_{i}" for i in range(10)]
        
        # Exact match
        projected = target_embeddings[5].copy()
        
        evaluator = TranslationEvaluator()
        precision = evaluator.compute_precision_at_k(
            projected_vector=projected,
            correct_translation="word_5",
            target_embeddings=target_embeddings,
            target_words=target_words,
            k=1
        )
        
        # Should be rank 1 with perfect match
        assert precision == 1.0
