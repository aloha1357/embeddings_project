"""
Tests for EmbeddingProjector class.

TDD approach: Write tests first (Red), then implement (Green), then refactor.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from src.models.embedding_projector import EmbeddingProjector
from src.models.word2vec_trainer import Word2VecTrainer
from src.utils.config import Word2VecConfig


class TestEmbeddingProjector:
    """Test cases for EmbeddingProjector initialization and basic functionality."""
    
    def test_initialization(self):
        """Test that EmbeddingProjector can be initialized."""
        projector = EmbeddingProjector()
        assert projector is not None
        assert projector.projection_matrix is None
    
    def test_extract_vectors_from_word_pairs(self, tmp_path):
        """Test extracting embedding vectors for word pairs."""
        # Create a small Word2Vec model
        sentences = [
            ["hello", "world"],
            ["hello", "there"],
            ["world", "peace"]
        ]
        
        config = Word2VecConfig(vector_size=10, window=2, min_count=1)
        trainer = Word2VecTrainer(config)
        trainer.train(sentences)
        
        # Create word pairs - using words that exist in both "models"
        word_pairs = [("hello", "world"), ("world", "peace")]
        
        projector = EmbeddingProjector()
        source_vectors, target_vectors = projector.extract_vectors(
            word_pairs=word_pairs,
            source_model=trainer.model,
            target_model=trainer.model  # Using same model for testing
        )
        
        assert source_vectors.shape == (2, 10)
        assert target_vectors.shape == (2, 10)
        assert source_vectors.dtype == np.float32
    
    def test_extract_vectors_filters_missing_words(self, tmp_path):
        """Test that extract_vectors filters out word pairs where words are not in vocabulary."""
        sentences = [
            ["hello", "world"],
            ["hello", "there"]
        ]
        
        config = Word2VecConfig(vector_size=10, window=2, min_count=1)
        trainer = Word2VecTrainer(config)
        trainer.train(sentences)
        
        # Include some words not in vocabulary
        word_pairs = [
            ("hello", "world"),
            ("missing", "word"),  # Not in vocab
            ("there", "hello")
        ]
        
        projector = EmbeddingProjector()
        source_vectors, target_vectors = projector.extract_vectors(
            word_pairs=word_pairs,
            source_model=trainer.model,
            target_model=trainer.model
        )
        
        # Should only have 2 valid pairs (hello-world, there-hello)
        assert source_vectors.shape[0] == 2
        assert target_vectors.shape[0] == 2
    
    def test_learn_projection_matrix(self):
        """Test learning projection matrix using least squares."""
        # Create simple source and target vectors
        np.random.seed(42)
        source_vectors = np.random.randn(100, 50).astype(np.float32)
        target_vectors = np.random.randn(100, 50).astype(np.float32)
        
        projector = EmbeddingProjector()
        projector.learn_projection(source_vectors, target_vectors)
        
        assert projector.projection_matrix is not None
        assert projector.projection_matrix.shape == (50, 50)
    
    def test_project_vectors(self):
        """Test projecting vectors using learned projection matrix."""
        np.random.seed(42)
        source_vectors = np.random.randn(100, 50).astype(np.float32)
        target_vectors = np.random.randn(100, 50).astype(np.float32)
        
        projector = EmbeddingProjector()
        projector.learn_projection(source_vectors, target_vectors)
        
        # Project some new vectors
        new_vectors = np.random.randn(10, 50).astype(np.float32)
        projected = projector.project(new_vectors)
        
        assert projected.shape == (10, 50)
        assert projected.dtype == np.float32
    
    def test_project_without_learned_matrix_raises_error(self):
        """Test that projecting without learning raises an error."""
        projector = EmbeddingProjector()
        vectors = np.random.randn(10, 50).astype(np.float32)
        
        with pytest.raises(ValueError, match="Projection matrix has not been learned"):
            projector.project(vectors)
    
    def test_save_and_load_projection_matrix(self, tmp_path):
        """Test saving and loading projection matrix."""
        np.random.seed(42)
        source_vectors = np.random.randn(100, 50).astype(np.float32)
        target_vectors = np.random.randn(100, 50).astype(np.float32)
        
        projector = EmbeddingProjector()
        projector.learn_projection(source_vectors, target_vectors)
        
        # Save matrix
        save_path = tmp_path / "projection_matrix.npy"
        projector.save_projection_matrix(str(save_path))
        
        # Load into new projector
        new_projector = EmbeddingProjector()
        new_projector.load_projection_matrix(str(save_path))
        
        assert new_projector.projection_matrix is not None
        np.testing.assert_array_almost_equal(
            projector.projection_matrix,
            new_projector.projection_matrix
        )


class TestEmbeddingProjectorIntegration:
    """Integration tests for EmbeddingProjector."""
    
    def test_full_projection_workflow(self, tmp_path):
        """Test complete workflow: train models -> extract vectors -> learn projection -> project."""
        # Train English model
        en_sentences = [
            ["king", "man", "ruler"],
            ["queen", "woman", "ruler"],
            ["man", "person"],
            ["woman", "person"]
        ]
        en_config = Word2VecConfig(vector_size=20, window=2, min_count=1, epochs=10)
        en_trainer = Word2VecTrainer(en_config)
        en_trainer.train(en_sentences)
        
        # Train German model (using English words for testing)
        de_sentences = [
            ["könig", "mann", "herrscher"],
            ["königin", "frau", "herrscher"],
            ["mann", "person"],
            ["frau", "person"]
        ]
        de_config = Word2VecConfig(vector_size=20, window=2, min_count=1, epochs=10)
        de_trainer = Word2VecTrainer(de_config)
        de_trainer.train(de_sentences)
        
        # Create word pairs for training
        train_pairs = [
            ("king", "könig"),
            ("queen", "königin"),
            ("man", "mann")
        ]
        
        # Extract vectors
        projector = EmbeddingProjector()
        X_source, Z_target = projector.extract_vectors(
            word_pairs=train_pairs,
            source_model=en_trainer.model,
            target_model=de_trainer.model
        )
        
        assert X_source.shape == (3, 20)
        assert Z_target.shape == (3, 20)
        
        # Learn projection
        projector.learn_projection(X_source, Z_target)
        assert projector.projection_matrix is not None
        
        # Project a test word
        test_vector = en_trainer.model.wv["woman"].reshape(1, -1)
        projected = projector.project(test_vector)
        
        assert projected.shape == (1, 20)
    
    def test_projection_improves_translation(self, tmp_path):
        """Test that projection brings source vectors closer to target vectors."""
        # Create simple synthetic data where target is a linear transformation of source
        np.random.seed(42)
        true_W = np.random.randn(30, 30)
        source_vectors = np.random.randn(100, 30).astype(np.float32)
        target_vectors = (source_vectors @ true_W).astype(np.float32)
        
        # Add some noise
        target_vectors += np.random.randn(100, 30).astype(np.float32) * 0.1
        
        projector = EmbeddingProjector()
        projector.learn_projection(source_vectors, target_vectors)
        
        # Test projection on new data
        test_source = np.random.randn(20, 30).astype(np.float32)
        test_target = (test_source @ true_W).astype(np.float32)
        
        # Project test source
        projected = projector.project(test_source)
        
        # Distance before projection
        dist_before = np.mean(np.linalg.norm(test_source - test_target, axis=1))
        
        # Distance after projection
        dist_after = np.mean(np.linalg.norm(projected - test_target, axis=1))
        
        # Projection should reduce distance
        assert dist_after < dist_before


class TestEmbeddingProjectorEdgeCases:
    """Test edge cases and error handling."""
    
    def test_extract_vectors_with_empty_pairs(self):
        """Test extracting vectors with empty word pairs list."""
        sentences = [["hello", "world"]]
        config = Word2VecConfig(vector_size=10, window=2, min_count=1)
        trainer = Word2VecTrainer(config)
        trainer.train(sentences)
        
        projector = EmbeddingProjector()
        source_vectors, target_vectors = projector.extract_vectors(
            word_pairs=[],
            source_model=trainer.model,
            target_model=trainer.model
        )
        
        assert source_vectors.shape == (0, 10)
        assert target_vectors.shape == (0, 10)
    
    def test_learn_projection_with_mismatched_dimensions(self):
        """Test that learning with mismatched vector dimensions raises error."""
        source_vectors = np.random.randn(100, 50).astype(np.float32)
        target_vectors = np.random.randn(100, 60).astype(np.float32)  # Different dimension
        
        projector = EmbeddingProjector()
        
        with pytest.raises(ValueError, match="Source and target vectors must have the same dimension"):
            projector.learn_projection(source_vectors, target_vectors)
    
    def test_learn_projection_with_mismatched_counts(self):
        """Test that learning with different numbers of vectors raises error."""
        source_vectors = np.random.randn(100, 50).astype(np.float32)
        target_vectors = np.random.randn(90, 50).astype(np.float32)  # Different count
        
        projector = EmbeddingProjector()
        
        with pytest.raises(ValueError, match="Source and target must have the same number of vectors"):
            projector.learn_projection(source_vectors, target_vectors)
    
    def test_project_with_wrong_dimension(self):
        """Test that projecting vectors with wrong dimension raises error."""
        source_vectors = np.random.randn(100, 50).astype(np.float32)
        target_vectors = np.random.randn(100, 50).astype(np.float32)
        
        projector = EmbeddingProjector()
        projector.learn_projection(source_vectors, target_vectors)
        
        # Try to project vectors with wrong dimension
        wrong_vectors = np.random.randn(10, 60).astype(np.float32)
        
        with pytest.raises(ValueError, match="Input vectors dimension .* does not match"):
            projector.project(wrong_vectors)
