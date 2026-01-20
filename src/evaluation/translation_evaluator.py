"""
Translation Evaluator Module.

This module implements evaluation metrics for cross-lingual word translation tasks.
The primary metric is Precision@k (P@k), which measures whether the correct translation
appears in the top-k nearest neighbors of the projected source embedding.

Key components:
- P@k computation for single word pairs
- Batch evaluation for test sets
- Integration with EmbeddingProjector for end-to-end evaluation
- Support for multiple k values (P@1, P@5, P@10, etc.)

Evaluation workflow:
1. Project source word embedding to target space
2. Find k nearest neighbors in target embedding space
3. Check if correct translation is in top-k neighbors
4. Aggregate results across test set

Mathematical formulation:
P@k = (1/N) * Σ I(correct_translation ∈ TopK(projected_vector))

where:
- N = number of test pairs
- I = indicator function (1 if true, 0 if false)
- TopK = k nearest neighbors by cosine similarity

Usage:
    evaluator = TranslationEvaluator()
    
    # Method 1: Direct evaluation with pre-projected vectors
    results = evaluator.evaluate_test_pairs(
        test_pairs=[(src, tgt, projected_vec), ...],
        target_embeddings=de_embeddings,
        target_words=de_words,
        k_values=[1, 5, 10]
    )
    
    # Method 2: End-to-end with projector
    results = evaluator.evaluate_with_projector(
        test_pairs=[(src, tgt), ...],
        source_model=en_model,
        target_model=de_model,
        projector=projector,
        k_values=[1, 5, 10]
    )
    
    print(f"P@1: {results['p@1']:.2%}")
    print(f"P@5: {results['p@5']:.2%}")
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
from gensim.models import Word2Vec
import logging

from src.models.embedding_projector import EmbeddingProjector


logger = logging.getLogger(__name__)


class TranslationEvaluator:
    """
    Evaluates translation quality using Precision@k metric.
    
    This class measures how often the correct translation appears in the top-k
    nearest neighbors when source embeddings are projected to target space.
    
    Higher P@k values indicate better translation performance.
    
    Example:
        >>> evaluator = TranslationEvaluator()
        >>> results = evaluator.evaluate_with_projector(
        ...     test_pairs=[("hello", "hallo"), ("world", "welt")],
        ...     source_model=en_model,
        ...     target_model=de_model,
        ...     projector=projector,
        ...     k_values=[1, 5, 10]
        ... )
        >>> print(f"P@1: {results['p@1']:.1%}")
    """
    
    def __init__(self):
        """Initialize TranslationEvaluator."""
        logger.info("TranslationEvaluator initialized")
    
    def compute_precision_at_k(
        self,
        projected_vector: np.ndarray,
        correct_translation: str,
        target_embeddings: np.ndarray,
        target_words: List[str],
        k: int
    ) -> float:
        """
        Compute Precision@k for a single source-target pair.
        
        Projects the source embedding and checks if the correct translation
        appears in the top-k nearest neighbors in the target space.
        
        Args:
            projected_vector: Projected source embedding, shape (d,)
            correct_translation: The correct target word
            target_embeddings: All target word embeddings, shape (vocab_size, d)
            target_words: List of target vocabulary words
            k: Number of nearest neighbors to consider
        
        Returns:
            1.0 if correct translation is in top-k, 0.0 otherwise
        
        Example:
            >>> precision = evaluator.compute_precision_at_k(
            ...     projected_vector=projected,
            ...     correct_translation="hallo",
            ...     target_embeddings=de_embeddings,
            ...     target_words=de_words,
            ...     k=5
            ... )
        """
        # Check if correct translation exists in vocabulary
        if correct_translation not in target_words:
            logger.debug(f"Translation '{correct_translation}' not in target vocabulary")
            return 0.0
        
        # Normalize projected vector for cosine similarity
        projected_norm = projected_vector / (np.linalg.norm(projected_vector) + 1e-8)
        
        # Normalize target embeddings
        target_norms = np.linalg.norm(target_embeddings, axis=1, keepdims=True) + 1e-8
        target_embeddings_norm = target_embeddings / target_norms
        
        # Compute cosine similarities
        similarities = target_embeddings_norm @ projected_norm
        
        # Get top-k indices (highest similarities)
        k_clamped = min(k, len(target_words))
        top_k_indices = np.argsort(similarities)[-k_clamped:][::-1]
        
        # Get top-k words
        top_k_words = [target_words[idx] for idx in top_k_indices]
        
        # Check if correct translation is in top-k
        if correct_translation in top_k_words:
            rank = top_k_words.index(correct_translation) + 1
            logger.debug(f"Translation '{correct_translation}' found at rank {rank}/{k_clamped}")
            return 1.0
        else:
            logger.debug(f"Translation '{correct_translation}' not in top-{k_clamped}")
            return 0.0
    
    def evaluate_test_pairs(
        self,
        test_pairs: List[Tuple[str, str, np.ndarray]],
        target_embeddings: np.ndarray,
        target_words: List[str],
        k_values: List[int]
    ) -> Dict[str, float]:
        """
        Evaluate multiple test pairs with multiple k values.
        
        Args:
            test_pairs: List of (source_word, target_word, projected_vector) tuples
            target_embeddings: All target word embeddings, shape (vocab_size, d)
            target_words: List of target vocabulary words
            k_values: List of k values to evaluate (e.g., [1, 5, 10])
        
        Returns:
            Dictionary with keys:
                - 'p@k' for each k in k_values
                - 'total_pairs': number of test pairs evaluated
        
        Example:
            >>> results = evaluator.evaluate_test_pairs(
            ...     test_pairs=[("hello", "hallo", projected_hello), ...],
            ...     target_embeddings=de_embeddings,
            ...     target_words=de_words,
            ...     k_values=[1, 5, 10]
            ... )
            >>> print(f"P@1: {results['p@1']:.2%}, P@5: {results['p@5']:.2%}")
        """
        if len(test_pairs) == 0:
            logger.warning("No test pairs provided for evaluation")
            results = {f"p@{k}": 0.0 for k in k_values}
            results["total_pairs"] = 0
            return results
        
        logger.info(f"Evaluating {len(test_pairs)} test pairs with k values: {k_values}")
        
        # Initialize counters for each k
        precision_sums = {k: 0.0 for k in k_values}
        
        # Evaluate each test pair
        for source_word, target_word, projected_vector in test_pairs:
            for k in k_values:
                precision = self.compute_precision_at_k(
                    projected_vector=projected_vector,
                    correct_translation=target_word,
                    target_embeddings=target_embeddings,
                    target_words=target_words,
                    k=k
                )
                precision_sums[k] += precision
        
        # Compute average precision for each k
        n_pairs = len(test_pairs)
        results = {
            f"p@{k}": precision_sums[k] / n_pairs
            for k in k_values
        }
        results["total_pairs"] = n_pairs
        
        # Log results
        for k in k_values:
            logger.info(f"P@{k}: {results[f'p@{k}']:.2%} ({int(precision_sums[k])}/{n_pairs})")
        
        return results
    
    def evaluate_with_projector(
        self,
        test_pairs: List[Tuple[str, str]],
        source_model: Word2Vec,
        target_model: Word2Vec,
        projector: EmbeddingProjector,
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, float]:
        """
        End-to-end evaluation with EmbeddingProjector integration.
        
        This method handles the complete workflow:
        1. Extract source embeddings for test words
        2. Project to target space using learned projection matrix
        3. Compute P@k metrics
        
        Args:
            test_pairs: List of (source_word, target_word) tuples
            source_model: Trained Word2Vec model for source language
            target_model: Trained Word2Vec model for target language
            projector: Trained EmbeddingProjector with learned projection matrix
            k_values: List of k values to evaluate (default: [1, 5, 10])
        
        Returns:
            Dictionary with evaluation results:
                - 'p@k' for each k value
                - 'total_pairs': number of valid test pairs evaluated
                - 'skipped_pairs': number of pairs skipped due to missing words
        
        Raises:
            ValueError: If projector has not been trained
        
        Example:
            >>> results = evaluator.evaluate_with_projector(
            ...     test_pairs=[("dog", "hund"), ("cat", "katze")],
            ...     source_model=en_model,
            ...     target_model=de_model,
            ...     projector=projector,
            ...     k_values=[1, 5, 10]
            ... )
        """
        if not projector.is_trained:
            raise ValueError(
                "Projector must be trained before evaluation. "
                "Call projector.learn_projection() first."
            )
        
        logger.info(f"Starting evaluation with {len(test_pairs)} test pairs")
        
        # Prepare test pairs with projected vectors
        test_pairs_with_projections = []
        skipped = 0
        
        for source_word, target_word in test_pairs:
            # Check if both words exist in vocabularies
            if source_word not in source_model.wv:
                logger.debug(f"Source word '{source_word}' not in vocabulary, skipping")
                skipped += 1
                continue
            
            if target_word not in target_model.wv:
                logger.debug(f"Target word '{target_word}' not in vocabulary, skipping")
                skipped += 1
                continue
            
            # Get source embedding and project it
            source_vector = source_model.wv[source_word].reshape(1, -1)
            projected_vector = projector.project(source_vector)[0]
            
            test_pairs_with_projections.append((
                source_word,
                target_word,
                projected_vector
            ))
        
        if skipped > 0:
            logger.warning(f"Skipped {skipped} test pairs due to missing vocabulary")
        
        # Get all target embeddings and words
        target_words = list(target_model.wv.index_to_key)
        target_embeddings = np.array([
            target_model.wv[word] for word in target_words
        ], dtype=np.float32)
        
        # Evaluate
        results = self.evaluate_test_pairs(
            test_pairs=test_pairs_with_projections,
            target_embeddings=target_embeddings,
            target_words=target_words,
            k_values=k_values
        )
        
        results["skipped_pairs"] = skipped
        
        return results
    
    def compute_recall_at_k(
        self,
        test_pairs: List[Tuple[str, str, np.ndarray]],
        target_embeddings: np.ndarray,
        target_words: List[str],
        k: int
    ) -> float:
        """
        Compute Recall@k (alternative metric).
        
        For single-translation tasks, Recall@k equals Precision@k.
        This method is provided for completeness and potential future extensions
        to handle multiple valid translations per source word.
        
        Args:
            test_pairs: List of (source_word, target_word, projected_vector) tuples
            target_embeddings: All target word embeddings
            target_words: List of target vocabulary words
            k: Number of nearest neighbors
        
        Returns:
            Recall@k score (same as Precision@k for single translations)
        """
        results = self.evaluate_test_pairs(
            test_pairs=test_pairs,
            target_embeddings=target_embeddings,
            target_words=target_words,
            k_values=[k]
        )
        return results[f"p@{k}"]
