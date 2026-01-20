"""
Tests for EmbeddingVisualizer class.

TDD approach: Write tests first (Red), then implement (Green), then refactor.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from src.visualization.embedding_visualizer import EmbeddingVisualizer
from src.models.word2vec_trainer import Word2VecTrainer
from src.utils.config import Word2VecConfig


class TestEmbeddingVisualizer:
    """Test cases for EmbeddingVisualizer initialization and basic functionality."""
    
    def test_initialization(self):
        """Test that EmbeddingVisualizer can be initialized."""
        visualizer = EmbeddingVisualizer()
        assert visualizer is not None
    
    def test_reduce_dimensions_tsne(self):
        """Test t-SNE dimensionality reduction."""
        # Create high-dimensional embeddings
        embeddings = np.random.randn(50, 100).astype(np.float32)
        
        visualizer = EmbeddingVisualizer()
        reduced = visualizer.reduce_dimensions(
            embeddings=embeddings,
            method='tsne',
            n_components=2
        )
        
        assert reduced.shape == (50, 2)
        assert reduced.dtype == np.float32
    
    def test_reduce_dimensions_pca(self):
        """Test PCA dimensionality reduction."""
        embeddings = np.random.randn(50, 100).astype(np.float32)
        
        visualizer = EmbeddingVisualizer()
        reduced = visualizer.reduce_dimensions(
            embeddings=embeddings,
            method='pca',
            n_components=2
        )
        
        assert reduced.shape == (50, 2)
        assert reduced.dtype == np.float32
    
    def test_reduce_dimensions_invalid_method(self):
        """Test that invalid method raises error."""
        embeddings = np.random.randn(50, 100).astype(np.float32)
        
        visualizer = EmbeddingVisualizer()
        with pytest.raises(ValueError, match="Unsupported dimensionality reduction method"):
            visualizer.reduce_dimensions(embeddings, method='invalid')
    
    def test_plot_embeddings_basic(self, tmp_path):
        """Test basic embedding scatter plot."""
        embeddings_2d = np.random.randn(20, 2).astype(np.float32)
        words = [f"word_{i}" for i in range(20)]
        
        visualizer = EmbeddingVisualizer()
        fig = visualizer.plot_embeddings(
            embeddings_2d=embeddings_2d,
            words=words,
            title="Test Embeddings"
        )
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Save and verify
        output_path = tmp_path / "test_plot.png"
        fig.savefig(output_path)
        assert output_path.exists()
        plt.close(fig)
    
    def test_plot_embeddings_with_labels(self, tmp_path):
        """Test embedding plot with custom labels and colors."""
        embeddings_2d = np.random.randn(30, 2).astype(np.float32)
        words = [f"word_{i}" for i in range(30)]
        labels = np.array([0] * 10 + [1] * 10 + [2] * 10)
        
        visualizer = EmbeddingVisualizer()
        fig = visualizer.plot_embeddings(
            embeddings_2d=embeddings_2d,
            words=words,
            labels=labels,
            title="Labeled Embeddings"
        )
        
        assert fig is not None
        output_path = tmp_path / "labeled_plot.png"
        fig.savefig(output_path)
        assert output_path.exists()
        plt.close(fig)
    
    def test_plot_bilingual_embeddings(self, tmp_path):
        """Test plotting source and target embeddings together."""
        source_embeddings = np.random.randn(20, 2).astype(np.float32)
        target_embeddings = np.random.randn(15, 2).astype(np.float32)
        source_words = [f"en_{i}" for i in range(20)]
        target_words = [f"de_{i}" for i in range(15)]
        
        visualizer = EmbeddingVisualizer()
        fig = visualizer.plot_bilingual_embeddings(
            source_embeddings_2d=source_embeddings,
            target_embeddings_2d=target_embeddings,
            source_words=source_words,
            target_words=target_words,
            title="Bilingual Embeddings"
        )
        
        assert fig is not None
        output_path = tmp_path / "bilingual_plot.png"
        fig.savefig(output_path)
        assert output_path.exists()
        plt.close(fig)
    
    def test_plot_semantic_relationships(self, tmp_path):
        """Test plotting semantic relationships (analogies)."""
        # Create embeddings where we can define relationships
        embeddings_2d = np.array([
            [0, 0],     # king
            [1, 0],     # queen
            [0, 1],     # man
            [1, 1],     # woman
            [0.5, 0.5]  # person
        ], dtype=np.float32)
        
        words = ["king", "queen", "man", "woman", "person"]
        
        relationships = [
            ("king", "queen", "man", "woman")  # king - man + woman ≈ queen
        ]
        
        visualizer = EmbeddingVisualizer()
        fig = visualizer.plot_semantic_relationships(
            embeddings_2d=embeddings_2d,
            words=words,
            relationships=relationships,
            title="Semantic Relationships"
        )
        
        assert fig is not None
        output_path = tmp_path / "relationships_plot.png"
        fig.savefig(output_path)
        assert output_path.exists()
        plt.close(fig)
    
    def test_plot_translation_pairs(self, tmp_path):
        """Test plotting translation pairs with connecting lines."""
        source_embeddings = np.array([
            [0, 0],
            [1, 1],
            [2, 0]
        ], dtype=np.float32)
        
        target_embeddings = np.array([
            [0.1, 0.1],
            [1.1, 1.1],
            [2.1, 0.1]
        ], dtype=np.float32)
        
        pairs = [
            ("hello", "hallo"),
            ("world", "welt"),
            ("good", "gut")
        ]
        
        visualizer = EmbeddingVisualizer()
        fig = visualizer.plot_translation_pairs(
            source_embeddings_2d=source_embeddings,
            target_embeddings_2d=target_embeddings,
            pairs=pairs,
            title="Translation Pairs"
        )
        
        assert fig is not None
        output_path = tmp_path / "translation_pairs.png"
        fig.savefig(output_path)
        assert output_path.exists()
        plt.close(fig)
    
    def test_create_embedding_comparison(self, tmp_path):
        """Test creating side-by-side comparison of embeddings."""
        embeddings1 = np.random.randn(20, 2).astype(np.float32)
        embeddings2 = np.random.randn(20, 2).astype(np.float32)
        words = [f"word_{i}" for i in range(20)]
        
        visualizer = EmbeddingVisualizer()
        fig = visualizer.create_embedding_comparison(
            embeddings1_2d=embeddings1,
            embeddings2_2d=embeddings2,
            words=words,
            title1="Before Projection",
            title2="After Projection"
        )
        
        assert fig is not None
        output_path = tmp_path / "comparison.png"
        fig.savefig(output_path)
        assert output_path.exists()
        plt.close(fig)


class TestEmbeddingVisualizerIntegration:
    """Integration tests for EmbeddingVisualizer with real Word2Vec models."""
    
    def test_visualize_word2vec_model(self, tmp_path):
        """Test visualizing embeddings from trained Word2Vec model."""
        # Train a small model
        sentences = [
            ["king", "man", "ruler", "power"],
            ["queen", "woman", "ruler", "power"],
            ["man", "person", "human"],
            ["woman", "person", "human"],
            ["boy", "child", "young"],
            ["girl", "child", "young"]
        ]
        
        config = Word2VecConfig(vector_size=50, window=2, min_count=1, epochs=20)
        trainer = Word2VecTrainer(config)
        trainer.train(sentences)
        
        # Get embeddings for specific words
        words = ["king", "queen", "man", "woman", "boy", "girl"]
        embeddings = np.array([trainer.model.wv[word] for word in words], dtype=np.float32)
        
        # Reduce to 2D
        visualizer = EmbeddingVisualizer()
        embeddings_2d = visualizer.reduce_dimensions(embeddings, method='tsne')
        
        # Plot
        fig = visualizer.plot_embeddings(
            embeddings_2d=embeddings_2d,
            words=words,
            title="Word2Vec Embeddings"
        )
        
        assert fig is not None
        output_path = tmp_path / "word2vec_plot.png"
        fig.savefig(output_path)
        assert output_path.exists()
        plt.close(fig)
    
    def test_full_visualization_workflow(self, tmp_path):
        """Test complete visualization workflow with multiple plots."""
        # Create models
        en_sentences = [
            ["hello", "world", "good", "morning"],
            ["hello", "there", "nice", "day"],
            ["world", "peace", "love", "harmony"]
        ]
        
        de_sentences = [
            ["hallo", "welt", "gut", "morgen"],
            ["hallo", "da", "schön", "tag"],
            ["welt", "frieden", "liebe", "harmonie"]
        ]
        
        config = Word2VecConfig(vector_size=50, window=2, min_count=1, epochs=20)
        
        en_trainer = Word2VecTrainer(config)
        en_trainer.train(en_sentences)
        
        de_trainer = Word2VecTrainer(config)
        de_trainer.train(de_sentences)
        
        # Get embeddings
        en_words = ["hello", "world", "good"]
        de_words = ["hallo", "welt", "gut"]
        
        en_embeddings = np.array([en_trainer.model.wv[w] for w in en_words], dtype=np.float32)
        de_embeddings = np.array([de_trainer.model.wv[w] for w in de_words], dtype=np.float32)
        
        # Combine and reduce
        all_embeddings = np.vstack([en_embeddings, de_embeddings])
        
        visualizer = EmbeddingVisualizer()
        all_2d = visualizer.reduce_dimensions(all_embeddings, method='tsne')
        
        en_2d = all_2d[:3]
        de_2d = all_2d[3:]
        
        # Create bilingual plot
        fig = visualizer.plot_bilingual_embeddings(
            source_embeddings_2d=en_2d,
            target_embeddings_2d=de_2d,
            source_words=en_words,
            target_words=de_words,
            title="English-German Embeddings"
        )
        
        assert fig is not None
        output_path = tmp_path / "bilingual_workflow.png"
        fig.savefig(output_path)
        assert output_path.exists()
        plt.close(fig)


class TestEmbeddingVisualizerEdgeCases:
    """Test edge cases and error handling."""
    
    def test_reduce_dimensions_too_few_samples(self):
        """Test t-SNE with fewer samples than required."""
        # t-SNE typically needs more samples than components
        embeddings = np.random.randn(2, 100).astype(np.float32)
        
        visualizer = EmbeddingVisualizer()
        # Should handle gracefully (might fall back to PCA or adjust perplexity)
        reduced = visualizer.reduce_dimensions(embeddings, method='tsne', n_components=2)
        
        assert reduced.shape[0] == 2
        assert reduced.shape[1] == 2
    
    def test_plot_embeddings_empty(self):
        """Test plotting with empty data."""
        visualizer = EmbeddingVisualizer()
        
        with pytest.raises(ValueError, match="Cannot plot empty embeddings"):
            visualizer.plot_embeddings(
                embeddings_2d=np.array([]).reshape(0, 2),
                words=[]
            )
    
    def test_plot_embeddings_mismatched_length(self):
        """Test plotting with mismatched embeddings and words length."""
        embeddings_2d = np.random.randn(10, 2).astype(np.float32)
        words = ["word1", "word2"]  # Only 2 words for 10 embeddings
        
        visualizer = EmbeddingVisualizer()
        
        with pytest.raises(ValueError, match="Number of embeddings.*must match"):
            visualizer.plot_embeddings(embeddings_2d, words)
    
    def test_plot_semantic_relationships_missing_word(self):
        """Test plotting relationships when a word is missing."""
        embeddings_2d = np.random.randn(5, 2).astype(np.float32)
        words = ["king", "queen", "man", "woman", "person"]
        
        # Relationship with non-existent word
        relationships = [("king", "missing", "man", "woman")]
        
        visualizer = EmbeddingVisualizer()
        
        with pytest.raises(ValueError, match="Word.*not found in vocabulary"):
            visualizer.plot_semantic_relationships(
                embeddings_2d, words, relationships
            )
    
    def test_reduce_dimensions_3d(self):
        """Test reducing to 3 dimensions."""
        embeddings = np.random.randn(30, 100).astype(np.float32)
        
        visualizer = EmbeddingVisualizer()
        reduced = visualizer.reduce_dimensions(
            embeddings, method='pca', n_components=3
        )
        
        assert reduced.shape == (30, 3)
    
    def test_plot_with_highlight_words(self, tmp_path):
        """Test plotting with specific words highlighted."""
        embeddings_2d = np.random.randn(20, 2).astype(np.float32)
        words = [f"word_{i}" for i in range(20)]
        highlight_words = ["word_5", "word_10", "word_15"]
        
        visualizer = EmbeddingVisualizer()
        fig = visualizer.plot_embeddings(
            embeddings_2d=embeddings_2d,
            words=words,
            highlight_words=highlight_words,
            title="Highlighted Words"
        )
        
        assert fig is not None
        output_path = tmp_path / "highlighted.png"
        fig.savefig(output_path)
        assert output_path.exists()
        plt.close(fig)
    
    def test_save_plot_to_file(self, tmp_path):
        """Test saving plot directly to file."""
        embeddings_2d = np.random.randn(15, 2).astype(np.float32)
        words = [f"word_{i}" for i in range(15)]
        
        visualizer = EmbeddingVisualizer()
        output_path = tmp_path / "direct_save.png"
        
        visualizer.save_embedding_plot(
            embeddings_2d=embeddings_2d,
            words=words,
            output_path=str(output_path),
            title="Direct Save Test"
        )
        
        assert output_path.exists()
