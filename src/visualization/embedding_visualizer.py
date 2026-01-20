"""
Embedding Visualizer Module.

This module provides visualization utilities for word embeddings, including:
- Dimensionality reduction (t-SNE, PCA)
- 2D/3D scatter plots
- Bilingual embedding visualization
- Semantic relationship plots (analogies)
- Translation pair visualization

Key components:
- reduce_dimensions(): t-SNE/PCA for high-dim → 2D/3D
- plot_embeddings(): Basic scatter plot with labels
- plot_bilingual_embeddings(): Side-by-side source/target visualization
- plot_semantic_relationships(): Visualize word analogies (king-man+woman≈queen)
- plot_translation_pairs(): Show source→target connections
- create_embedding_comparison(): Before/after projection comparison

Mathematical background:
- t-SNE: Preserves local structure, good for clusters
- PCA: Preserves global variance, faster than t-SNE
- Cosine similarity for nearest neighbors

Usage:
    visualizer = EmbeddingVisualizer()
    
    # Reduce dimensions
    embeddings_2d = visualizer.reduce_dimensions(
        embeddings, method='tsne', n_components=2
    )
    
    # Plot
    fig = visualizer.plot_embeddings(
        embeddings_2d=embeddings_2d,
        words=words,
        title="Word Embeddings"
    )
    
    # Save
    fig.savefig("embeddings.png", dpi=300, bbox_inches='tight')
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


class EmbeddingVisualizer:
    """
    Visualize word embeddings using dimensionality reduction and plotting.
    
    This class provides comprehensive visualization tools for exploring
    word embedding spaces, including support for bilingual embeddings,
    semantic relationships, and translation quality analysis.
    
    Example:
        >>> visualizer = EmbeddingVisualizer()
        >>> embeddings_2d = visualizer.reduce_dimensions(embeddings)
        >>> fig = visualizer.plot_embeddings(embeddings_2d, words)
        >>> fig.savefig("plot.png")
    """
    
    def __init__(self):
        """Initialize EmbeddingVisualizer."""
        logger.info("EmbeddingVisualizer initialized")
    
    def reduce_dimensions(
        self,
        embeddings: np.ndarray,
        method: str = 'tsne',
        n_components: int = 2,
        **kwargs
    ) -> np.ndarray:
        """
        Reduce embedding dimensions using t-SNE or PCA.
        
        t-SNE is better for visualization (preserves local structure) but slower.
        PCA is faster but may not capture complex relationships.
        
        Args:
            embeddings: High-dimensional embeddings, shape (n_samples, n_features)
            method: 'tsne' or 'pca'
            n_components: Target dimensions (typically 2 or 3)
            **kwargs: Additional arguments for t-SNE or PCA
        
        Returns:
            Reduced embeddings, shape (n_samples, n_components)
        
        Raises:
            ValueError: If method is not supported
        
        Example:
            >>> embeddings = np.random.randn(100, 300)
            >>> embeddings_2d = visualizer.reduce_dimensions(embeddings, method='tsne')
        """
        if method.lower() == 'tsne':
            # Adjust perplexity for small datasets
            n_samples = embeddings.shape[0]
            # Perplexity must be less than n_samples and at least 1
            perplexity = min(30, max(1, (n_samples - 1) // 2))
            
            # If dataset is too small for t-SNE, fall back to PCA
            if n_samples < 4:
                logger.warning(
                    f"Dataset too small for t-SNE (n={n_samples}), falling back to PCA"
                )
                return self.reduce_dimensions(embeddings, method='pca', n_components=n_components)
            
            logger.info(f"Reducing dimensions with t-SNE (perplexity={perplexity})")
            
            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=42,
                **kwargs
            )
            reduced = tsne.fit_transform(embeddings)
            
        elif method.lower() == 'pca':
            logger.info(f"Reducing dimensions with PCA to {n_components}D")
            
            pca = PCA(n_components=n_components, random_state=42, **kwargs)
            reduced = pca.fit_transform(embeddings)
            
            # Log explained variance
            variance_ratio = pca.explained_variance_ratio_
            logger.info(f"PCA explained variance: {variance_ratio.sum():.2%}")
            
        else:
            raise ValueError(
                f"Unsupported dimensionality reduction method: '{method}'. "
                "Use 'tsne' or 'pca'."
            )
        
        return reduced.astype(np.float32)
    
    def plot_embeddings(
        self,
        embeddings_2d: np.ndarray,
        words: List[str],
        labels: Optional[np.ndarray] = None,
        highlight_words: Optional[List[str]] = None,
        title: str = "Word Embeddings",
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Create scatter plot of 2D embeddings with word labels.
        
        Args:
            embeddings_2d: 2D embeddings, shape (n_words, 2)
            words: List of word labels
            labels: Optional category labels for coloring, shape (n_words,)
            highlight_words: Optional list of words to highlight
            title: Plot title
            figsize: Figure size (width, height)
        
        Returns:
            Matplotlib Figure object
        
        Raises:
            ValueError: If embeddings and words length mismatch
        
        Example:
            >>> fig = visualizer.plot_embeddings(
            ...     embeddings_2d, words, title="My Embeddings"
            ... )
            >>> fig.savefig("output.png")
        """
        if len(embeddings_2d) == 0:
            raise ValueError("Cannot plot empty embeddings")
        
        if len(embeddings_2d) != len(words):
            raise ValueError(
                f"Number of embeddings ({len(embeddings_2d)}) must match "
                f"number of words ({len(words)})"
            )
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Scatter plot with optional labels
        if labels is not None:
            scatter = ax.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=labels,
                cmap='tab10',
                alpha=0.6,
                s=50
            )
            plt.colorbar(scatter, ax=ax, label="Category")
        else:
            ax.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                alpha=0.6,
                s=50,
                color='steelblue'
            )
        
        # Add word labels
        for i, word in enumerate(words):
            if highlight_words and word in highlight_words:
                # Highlight specific words
                ax.annotate(
                    word,
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=10,
                    fontweight='bold',
                    color='red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
                )
            else:
                ax.annotate(
                    word,
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=8,
                    alpha=0.7
                )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        logger.info(f"Created embedding plot: {title}")
        return fig
    
    def plot_bilingual_embeddings(
        self,
        source_embeddings_2d: np.ndarray,
        target_embeddings_2d: np.ndarray,
        source_words: List[str],
        target_words: List[str],
        title: str = "Bilingual Embeddings",
        figsize: Tuple[int, int] = (14, 7)
    ) -> plt.Figure:
        """
        Plot source and target language embeddings side by side.
        
        Args:
            source_embeddings_2d: Source language embeddings, shape (n_source, 2)
            target_embeddings_2d: Target language embeddings, shape (n_target, 2)
            source_words: Source language words
            target_words: Target language words
            title: Overall plot title
            figsize: Figure size
        
        Returns:
            Matplotlib Figure object
        
        Example:
            >>> fig = visualizer.plot_bilingual_embeddings(
            ...     en_embeddings, de_embeddings, en_words, de_words
            ... )
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Source language plot
        ax1.scatter(
            source_embeddings_2d[:, 0],
            source_embeddings_2d[:, 1],
            alpha=0.6,
            s=50,
            color='steelblue',
            label='Source'
        )
        for i, word in enumerate(source_words):
            ax1.annotate(
                word,
                (source_embeddings_2d[i, 0], source_embeddings_2d[i, 1]),
                fontsize=8,
                alpha=0.7
            )
        ax1.set_title("Source Language", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Dimension 1")
        ax1.set_ylabel("Dimension 2")
        ax1.grid(True, alpha=0.3)
        
        # Target language plot
        ax2.scatter(
            target_embeddings_2d[:, 0],
            target_embeddings_2d[:, 1],
            alpha=0.6,
            s=50,
            color='coral',
            label='Target'
        )
        for i, word in enumerate(target_words):
            ax2.annotate(
                word,
                (target_embeddings_2d[i, 0], target_embeddings_2d[i, 1]),
                fontsize=8,
                alpha=0.7
            )
        ax2.set_title("Target Language", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Dimension 1")
        ax2.set_ylabel("Dimension 2")
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        logger.info(f"Created bilingual embedding plot: {title}")
        return fig
    
    def plot_semantic_relationships(
        self,
        embeddings_2d: np.ndarray,
        words: List[str],
        relationships: List[Tuple[str, str, str, str]],
        title: str = "Semantic Relationships",
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Visualize semantic relationships (analogies) in embedding space.
        
        Shows relationships like: king - man + woman ≈ queen
        
        Args:
            embeddings_2d: 2D embeddings, shape (n_words, 2)
            words: List of word labels
            relationships: List of (word_a, word_b, word_c, word_d) tuples
                          representing: word_a - word_c + word_d ≈ word_b
            title: Plot title
            figsize: Figure size
        
        Returns:
            Matplotlib Figure object
        
        Raises:
            ValueError: If any word in relationships is not in vocabulary
        
        Example:
            >>> relationships = [("king", "queen", "man", "woman")]
            >>> fig = visualizer.plot_semantic_relationships(
            ...     embeddings_2d, words, relationships
            ... )
        """
        # Create word to index mapping
        word_to_idx = {word: i for i, word in enumerate(words)}
        
        # Validate all words exist
        for a, b, c, d in relationships:
            for word in [a, b, c, d]:
                if word not in word_to_idx:
                    raise ValueError(f"Word '{word}' not found in vocabulary")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot all embeddings
        ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            alpha=0.3,
            s=30,
            color='lightgray'
        )
        
        # Highlight words in relationships
        relationship_words = set()
        for a, b, c, d in relationships:
            relationship_words.update([a, b, c, d])
        
        for word in relationship_words:
            idx = word_to_idx[word]
            ax.scatter(
                embeddings_2d[idx, 0],
                embeddings_2d[idx, 1],
                s=100,
                color='red',
                zorder=5
            )
            ax.annotate(
                word,
                (embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                zorder=6
            )
        
        # Draw arrows for relationships
        colors = ['blue', 'green', 'purple', 'orange']
        for i, (a, b, c, d) in enumerate(relationships):
            idx_a = word_to_idx[a]
            idx_c = word_to_idx[c]
            idx_d = word_to_idx[d]
            idx_b = word_to_idx[b]
            
            color = colors[i % len(colors)]
            
            # Arrow from c to a (king - man)
            ax.arrow(
                embeddings_2d[idx_c, 0], embeddings_2d[idx_c, 1],
                embeddings_2d[idx_a, 0] - embeddings_2d[idx_c, 0],
                embeddings_2d[idx_a, 1] - embeddings_2d[idx_c, 1],
                head_width=0.05, head_length=0.1,
                fc=color, ec=color, alpha=0.6, linestyle='--'
            )
            
            # Arrow from d to b (woman → queen)
            ax.arrow(
                embeddings_2d[idx_d, 0], embeddings_2d[idx_d, 1],
                embeddings_2d[idx_b, 0] - embeddings_2d[idx_d, 0],
                embeddings_2d[idx_b, 1] - embeddings_2d[idx_d, 1],
                head_width=0.05, head_length=0.1,
                fc=color, ec=color, alpha=0.6, linestyle='--'
            )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        logger.info(f"Created semantic relationships plot: {title}")
        return fig
    
    def plot_translation_pairs(
        self,
        source_embeddings_2d: np.ndarray,
        target_embeddings_2d: np.ndarray,
        pairs: List[Tuple[str, str]],
        title: str = "Translation Pairs",
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot translation pairs with connecting lines.
        
        Shows how source words map to target words via projection.
        
        Args:
            source_embeddings_2d: Source embeddings, shape (n_pairs, 2)
            target_embeddings_2d: Target embeddings, shape (n_pairs, 2)
            pairs: List of (source_word, target_word) tuples
            title: Plot title
            figsize: Figure size
        
        Returns:
            Matplotlib Figure object
        
        Example:
            >>> pairs = [("hello", "hallo"), ("world", "welt")]
            >>> fig = visualizer.plot_translation_pairs(
            ...     source_2d, target_2d, pairs
            ... )
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot source words
        ax.scatter(
            source_embeddings_2d[:, 0],
            source_embeddings_2d[:, 1],
            s=100,
            color='steelblue',
            alpha=0.6,
            label='Source',
            zorder=3
        )
        
        # Plot target words
        ax.scatter(
            target_embeddings_2d[:, 0],
            target_embeddings_2d[:, 1],
            s=100,
            color='coral',
            alpha=0.6,
            label='Target',
            zorder=3
        )
        
        # Draw connecting lines
        for i, (src_word, tgt_word) in enumerate(pairs):
            ax.plot(
                [source_embeddings_2d[i, 0], target_embeddings_2d[i, 0]],
                [source_embeddings_2d[i, 1], target_embeddings_2d[i, 1]],
                'k--',
                alpha=0.3,
                linewidth=1,
                zorder=1
            )
            
            # Label source word
            ax.annotate(
                src_word,
                (source_embeddings_2d[i, 0], source_embeddings_2d[i, 1]),
                fontsize=9,
                color='darkblue',
                fontweight='bold',
                zorder=4
            )
            
            # Label target word
            ax.annotate(
                tgt_word,
                (target_embeddings_2d[i, 0], target_embeddings_2d[i, 1]),
                fontsize=9,
                color='darkred',
                fontweight='bold',
                zorder=4
            )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        logger.info(f"Created translation pairs plot: {title}")
        return fig
    
    def create_embedding_comparison(
        self,
        embeddings1_2d: np.ndarray,
        embeddings2_2d: np.ndarray,
        words: List[str],
        title1: str = "Before",
        title2: str = "After",
        figsize: Tuple[int, int] = (16, 7)
    ) -> plt.Figure:
        """
        Create side-by-side comparison of two embedding sets.
        
        Useful for visualizing before/after projection effects.
        
        Args:
            embeddings1_2d: First embedding set, shape (n_words, 2)
            embeddings2_2d: Second embedding set, shape (n_words, 2)
            words: Word labels
            title1: Title for first plot
            title2: Title for second plot
            figsize: Figure size
        
        Returns:
            Matplotlib Figure object
        
        Example:
            >>> fig = visualizer.create_embedding_comparison(
            ...     before_2d, after_2d, words,
            ...     title1="Before Projection",
            ...     title2="After Projection"
            ... )
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # First plot
        ax1.scatter(
            embeddings1_2d[:, 0],
            embeddings1_2d[:, 1],
            alpha=0.6,
            s=50,
            color='steelblue'
        )
        for i, word in enumerate(words):
            ax1.annotate(
                word,
                (embeddings1_2d[i, 0], embeddings1_2d[i, 1]),
                fontsize=8,
                alpha=0.7
            )
        ax1.set_title(title1, fontsize=12, fontweight='bold')
        ax1.set_xlabel("Dimension 1")
        ax1.set_ylabel("Dimension 2")
        ax1.grid(True, alpha=0.3)
        
        # Second plot
        ax2.scatter(
            embeddings2_2d[:, 0],
            embeddings2_2d[:, 1],
            alpha=0.6,
            s=50,
            color='coral'
        )
        for i, word in enumerate(words):
            ax2.annotate(
                word,
                (embeddings2_2d[i, 0], embeddings2_2d[i, 1]),
                fontsize=8,
                alpha=0.7
            )
        ax2.set_title(title2, fontsize=12, fontweight='bold')
        ax2.set_xlabel("Dimension 1")
        ax2.set_ylabel("Dimension 2")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        logger.info(f"Created embedding comparison: {title1} vs {title2}")
        return fig
    
    def save_embedding_plot(
        self,
        embeddings_2d: np.ndarray,
        words: List[str],
        output_path: str,
        title: str = "Word Embeddings",
        dpi: int = 300,
        **kwargs
    ) -> None:
        """
        Create and save embedding plot directly to file.
        
        Args:
            embeddings_2d: 2D embeddings, shape (n_words, 2)
            words: Word labels
            output_path: Output file path (e.g., "plot.png")
            title: Plot title
            dpi: Resolution (dots per inch)
            **kwargs: Additional arguments for plot_embeddings()
        
        Example:
            >>> visualizer.save_embedding_plot(
            ...     embeddings_2d, words, "output.png", dpi=300
            ... )
        """
        fig = self.plot_embeddings(embeddings_2d, words, title=title, **kwargs)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved embedding plot to {output_path}")
