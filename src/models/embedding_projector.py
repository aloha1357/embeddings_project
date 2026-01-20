"""
Embedding Projector Module.

This module implements cross-lingual word embedding projection using linear transformation.
Given word pairs with embeddings in source and target languages, it learns a projection matrix
that maps source embeddings to target embedding space.

Key components:
- Vector extraction from Word2Vec models
- Projection matrix learning using least squares (np.linalg.lstsq)
- Vector projection using learned matrix
- Model persistence (save/load)

Mathematical formulation:
Given source embeddings X (n x d) and target embeddings Z (n x d),
find projection matrix W (d x d) that minimizes:
    ||XW - Z||²_F
    
Solution: W = (X^T X)^-1 X^T Z (via least squares)

Usage:
    projector = EmbeddingProjector()
    
    # Extract vectors for word pairs
    X_source, Z_target = projector.extract_vectors(
        word_pairs=[("hello", "hallo"), ("world", "welt")],
        source_model=en_model,
        target_model=de_model
    )
    
    # Learn projection matrix
    projector.learn_projection(X_source, Z_target)
    
    # Project new vectors
    projected = projector.project(new_source_vectors)
    
    # Save/load
    projector.save_projection_matrix("projection.npy")
    projector.load_projection_matrix("projection.npy")
"""

from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path
import logging
from gensim.models import Word2Vec


logger = logging.getLogger(__name__)


class EmbeddingProjector:
    """
    Cross-lingual word embedding projector.
    
    This class implements functionality for learning and applying linear transformations
    to map word embeddings from source language to target language space.
    
    Attributes:
        projection_matrix: Learned projection matrix W (d x d), None if not yet learned
    
    Example:
        >>> projector = EmbeddingProjector()
        >>> X, Z = projector.extract_vectors(pairs, en_model, de_model)
        >>> projector.learn_projection(X, Z)
        >>> projected = projector.project(new_vectors)
    """
    
    def __init__(self):
        """Initialize EmbeddingProjector with no learned projection matrix."""
        self.projection_matrix: Optional[np.ndarray] = None
        logger.info("EmbeddingProjector initialized")
    
    def extract_vectors(
        self,
        word_pairs: List[Tuple[str, str]],
        source_model: Word2Vec,
        target_model: Word2Vec
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract embedding vectors for word pairs from trained models.
        
        For each word pair (source_word, target_word), retrieves the embedding vectors
        from the respective models. Filters out pairs where either word is not in
        the vocabulary.
        
        Args:
            word_pairs: List of (source_word, target_word) tuples
            source_model: Trained Word2Vec model for source language
            target_model: Trained Word2Vec model for target language
        
        Returns:
            Tuple of (source_vectors, target_vectors):
                - source_vectors: np.ndarray of shape (n_valid_pairs, vector_size)
                - target_vectors: np.ndarray of shape (n_valid_pairs, vector_size)
        
        Example:
            >>> pairs = [("hello", "hallo"), ("world", "welt")]
            >>> X, Z = projector.extract_vectors(pairs, en_model, de_model)
            >>> X.shape  # (2, 100) if vector_size=100
        """
        source_vectors_list = []
        target_vectors_list = []
        skipped_pairs = 0
        
        for source_word, target_word in word_pairs:
            # Check if both words exist in vocabularies
            if source_word in source_model.wv and target_word in target_model.wv:
                source_vectors_list.append(source_model.wv[source_word])
                target_vectors_list.append(target_model.wv[target_word])
            else:
                skipped_pairs += 1
        
        if skipped_pairs > 0:
            logger.warning(
                f"Skipped {skipped_pairs} word pairs due to missing vocabulary entries. "
                f"Valid pairs: {len(source_vectors_list)}/{len(word_pairs)}"
            )
        
        # Convert to numpy arrays
        if len(source_vectors_list) > 0:
            source_vectors = np.array(source_vectors_list, dtype=np.float32)
            target_vectors = np.array(target_vectors_list, dtype=np.float32)
        else:
            # Return empty arrays with correct shape (0, vector_size)
            vector_size = source_model.wv.vector_size
            source_vectors = np.empty((0, vector_size), dtype=np.float32)
            target_vectors = np.empty((0, vector_size), dtype=np.float32)
            logger.warning("No valid word pairs found - all words missing from vocabularies")
        
        logger.info(
            f"Extracted vectors for {len(source_vectors)} word pairs. "
            f"Shape: {source_vectors.shape}"
        )
        
        return source_vectors, target_vectors
    
    def learn_projection(
        self,
        source_vectors: np.ndarray,
        target_vectors: np.ndarray
    ) -> None:
        """
        Learn projection matrix using least squares method.
        
        Solves the linear regression problem:
            min_W ||XW - Z||²_F
        
        where X is source_vectors (n x d), Z is target_vectors (n x d),
        and W is the projection matrix (d x d).
        
        Uses np.linalg.lstsq for numerical stability.
        
        Args:
            source_vectors: Source language embeddings, shape (n, d)
            target_vectors: Target language embeddings, shape (n, d)
        
        Raises:
            ValueError: If vector dimensions or counts don't match
        
        Side effects:
            Sets self.projection_matrix to learned matrix W
        
        Example:
            >>> X = np.random.randn(1000, 100)
            >>> Z = np.random.randn(1000, 100)
            >>> projector.learn_projection(X, Z)
            >>> projector.projection_matrix.shape  # (100, 100)
        """
        # Validate inputs
        if source_vectors.shape[0] != target_vectors.shape[0]:
            raise ValueError(
                f"Source and target must have the same number of vectors. "
                f"Got source: {source_vectors.shape[0]}, target: {target_vectors.shape[0]}"
            )
        
        if source_vectors.shape[1] != target_vectors.shape[1]:
            raise ValueError(
                f"Source and target vectors must have the same dimension. "
                f"Got source: {source_vectors.shape[1]}, target: {target_vectors.shape[1]}"
            )
        
        n_samples, vector_dim = source_vectors.shape
        logger.info(f"Learning projection matrix from {n_samples} word pairs, dimension {vector_dim}")
        
        # Solve least squares: XW = Z => W = (X^T X)^-1 X^T Z
        # np.linalg.lstsq solves this efficiently and handles singular matrices
        self.projection_matrix, residuals, rank, s = np.linalg.lstsq(
            source_vectors,
            target_vectors,
            rcond=None
        )
        
        logger.info(
            f"Projection matrix learned. Shape: {self.projection_matrix.shape}, "
            f"Rank: {rank}, Residual sum: {residuals.sum() if len(residuals) > 0 else 0:.4f}"
        )
    
    def project(self, vectors: np.ndarray) -> np.ndarray:
        """
        Project vectors using learned projection matrix.
        
        Applies the transformation: projected = vectors @ W
        
        Args:
            vectors: Input vectors to project, shape (n, d)
        
        Returns:
            Projected vectors, shape (n, d)
        
        Raises:
            ValueError: If projection matrix not learned or dimension mismatch
        
        Example:
            >>> test_vectors = np.random.randn(10, 100)
            >>> projected = projector.project(test_vectors)
            >>> projected.shape  # (10, 100)
        """
        if self.projection_matrix is None:
            raise ValueError(
                "Projection matrix has not been learned yet. "
                "Call learn_projection() first."
            )
        
        expected_dim = self.projection_matrix.shape[0]
        if vectors.shape[1] != expected_dim:
            raise ValueError(
                f"Input vectors dimension ({vectors.shape[1]}) does not match "
                f"projection matrix dimension ({expected_dim})"
            )
        
        # Apply projection: XW
        projected = vectors @ self.projection_matrix
        
        logger.debug(f"Projected {len(vectors)} vectors from shape {vectors.shape} to {projected.shape}")
        
        return projected.astype(np.float32)
    
    def save_projection_matrix(self, filepath: str) -> None:
        """
        Save learned projection matrix to disk.
        
        Args:
            filepath: Path to save the projection matrix (.npy file)
        
        Raises:
            ValueError: If projection matrix not learned yet
        
        Example:
            >>> projector.save_projection_matrix("models/projection.npy")
        """
        if self.projection_matrix is None:
            raise ValueError(
                "No projection matrix to save. "
                "Call learn_projection() first."
            )
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(filepath, self.projection_matrix)
        logger.info(f"Projection matrix saved to {filepath}")
    
    def load_projection_matrix(self, filepath: str) -> None:
        """
        Load projection matrix from disk.
        
        Args:
            filepath: Path to load the projection matrix from (.npy file)
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If loaded data has invalid shape
        
        Side effects:
            Sets self.projection_matrix to loaded matrix
        
        Example:
            >>> projector = EmbeddingProjector()
            >>> projector.load_projection_matrix("models/projection.npy")
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Projection matrix file not found: {filepath}")
        
        self.projection_matrix = np.load(filepath)
        
        # Validate loaded matrix is square
        if len(self.projection_matrix.shape) != 2:
            raise ValueError(
                f"Invalid projection matrix shape: {self.projection_matrix.shape}. "
                "Expected 2D array."
            )
        
        if self.projection_matrix.shape[0] != self.projection_matrix.shape[1]:
            raise ValueError(
                f"Projection matrix must be square. "
                f"Got shape: {self.projection_matrix.shape}"
            )
        
        logger.info(f"Projection matrix loaded from {filepath}. Shape: {self.projection_matrix.shape}")
    
    @property
    def is_trained(self) -> bool:
        """
        Check if projection matrix has been learned.
        
        Returns:
            True if projection matrix is available, False otherwise
        """
        return self.projection_matrix is not None
    
    @property
    def matrix_dimension(self) -> Optional[int]:
        """
        Get dimension of projection matrix.
        
        Returns:
            Dimension (d) if matrix learned, None otherwise
        """
        if self.projection_matrix is None:
            return None
        return self.projection_matrix.shape[0]
