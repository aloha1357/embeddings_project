"""
Model exploration utilities
Tools for exploring Word2Vec models - Task 3
"""
from typing import List, Tuple, Dict, Optional
from src.models.word2vec_trainer import Word2VecTrainer


class ModelExplorer:
    """
    探索 Word2Vec 模型的工具类
    
    功能:
    - 查找相似词
    - 分析词类比
    - 比较多个词的相似度
    - 探索词关系
    """
    
    def __init__(self, trainer: Word2VecTrainer):
        """
        初始化探索器
        
        Args:
            trainer: 已训练的 Word2VecTrainer
        """
        self.trainer = trainer
    
    def explore_word(self, word: str, topn: int = 10) -> None:
        """
        探索一个词的相似词
        
        Args:
            word: 要探索的词
            topn: 显示的相似词数量
        """
        if word not in self.trainer.vocabulary:
            print(f"⚠️  Word '{word}' not in vocabulary")
            return
        
        print(f"\n{'='*60}")
        print(f"Exploring: '{word}'")
        print(f"{'='*60}")
        
        similar = self.trainer.get_most_similar(word, topn=topn)
        
        print(f"\nMost similar words (top {topn}):")
        for i, (sim_word, score) in enumerate(similar, 1):
            print(f"  {i:2d}. {sim_word:20s} (similarity: {score:.4f})")
    
    def explore_words(self, words: List[str], topn: int = 5) -> None:
        """
        探索多个词
        
        Args:
            words: 词列表
            topn: 每个词显示的相似词数量
        """
        for word in words:
            self.explore_word(word, topn=topn)
    
    def find_analogies(
        self,
        positive: List[str],
        negative: List[str],
        topn: int = 5
    ) -> Optional[List[Tuple[str, float]]]:
        """
        寻找词类比
        
        例如: woman + king - man ≈ queen
        
        Args:
            positive: 正向词（加法）
            negative: 负向词（减法）
            topn: 返回结果数量
        
        Returns:
            结果列表或 None
        """
        # 检查所有词是否在词汇表中
        missing = []
        for word in positive + negative:
            if word not in self.trainer.vocabulary:
                missing.append(word)
        
        if missing:
            print(f"⚠️  Words not in vocabulary: {', '.join(missing)}")
            return None
        
        print(f"\n{'='*60}")
        print(f"Analogy: {' + '.join(positive)} - {' - '.join(negative)}")
        print(f"{'='*60}")
        
        try:
            results = self.trainer.compute_analogy(
                positive=positive,
                negative=negative,
                topn=topn
            )
            
            print(f"\nTop {topn} results:")
            for i, (word, score) in enumerate(results, 1):
                print(f"  {i:2d}. {word:20s} (score: {score:.4f})")
            
            return results
        except Exception as e:
            print(f"❌ Error computing analogy: {e}")
            return None
    
    def explore_analogy_patterns(self) -> None:
        """
        探索常见的类比模式
        """
        print(f"\n{'='*70}")
        print("Exploring Common Analogy Patterns")
        print(f"{'='*70}")
        
        # 定义一些常见的类比模式
        patterns = [
            # 性别类比
            {
                'name': 'Gender (man/woman)',
                'examples': [
                    (['woman', 'king'], ['man']),
                    (['woman', 'actor'], ['man']),
                    (['man', 'queen'], ['woman']),
                ]
            },
            # 国家-首都
            {
                'name': 'Country-Capital',
                'examples': [
                    (['paris', 'germany'], ['france']),
                    (['berlin', 'france'], ['germany']),
                ]
            },
            # 形容词比较级
            {
                'name': 'Comparative Forms',
                'examples': [
                    (['bigger', 'small'], ['big']),
                    (['worse', 'good'], ['bad']),
                ]
            },
            # 复数形式
            {
                'name': 'Singular-Plural',
                'examples': [
                    (['countries', 'city'], ['country']),
                    (['children', 'person'], ['child']),
                ]
            }
        ]
        
        for pattern in patterns:
            print(f"\n{'-'*70}")
            print(f"Pattern: {pattern['name']}")
            print(f"{'-'*70}")
            
            for positive, negative in pattern['examples']:
                self.find_analogies(positive, negative, topn=3)
    
    def analyze_word_relationships(
        self,
        word_pairs: List[Tuple[str, str]]
    ) -> None:
        """
        分析词对之间的关系
        
        Args:
            word_pairs: 词对列表
        """
        print(f"\n{'='*60}")
        print("Word Relationship Analysis")
        print(f"{'='*60}")
        
        from scipy.spatial.distance import cosine
        
        for word1, word2 in word_pairs:
            if word1 not in self.trainer.vocabulary:
                print(f"⚠️  '{word1}' not in vocabulary")
                continue
            if word2 not in self.trainer.vocabulary:
                print(f"⚠️  '{word2}' not in vocabulary")
                continue
            
            vec1 = self.trainer.get_vector(word1)
            vec2 = self.trainer.get_vector(word2)
            
            # 计算余弦相似度
            similarity = 1 - cosine(vec1, vec2)
            
            print(f"\n{word1:15s} <-> {word2:15s}: {similarity:.4f}")
    
    def check_vocabulary_coverage(self, words: List[str]) -> Dict[str, bool]:
        """
        检查词汇表覆盖率
        
        Args:
            words: 要检查的词列表
        
        Returns:
            词典 {word: in_vocabulary}
        """
        coverage = {}
        in_vocab = 0
        
        for word in words:
            is_in = word in self.trainer.vocabulary
            coverage[word] = is_in
            if is_in:
                in_vocab += 1
        
        print(f"\nVocabulary Coverage: {in_vocab}/{len(words)} ({in_vocab/len(words)*100:.1f}%)")
        
        missing = [w for w, is_in in coverage.items() if not is_in]
        if missing:
            print(f"Missing words: {', '.join(missing)}")
        
        return coverage


def explore_task3_examples(trainer: Word2VecTrainer) -> None:
    """
    执行 Task 3 中要求的探索
    
    Args:
        trainer: 英语模型训练器
    """
    explorer = ModelExplorer(trainer)
    
    print("\n" + "="*70)
    print("TASK 3: Model Exploration")
    print("="*70)
    
    # (1) 探索指定的词
    print("\n" + "="*70)
    print("Part 1: Exploring specific words")
    print("="*70)
    
    task_words = ['small', 'expensive', 'coordinating']
    explorer.explore_words(task_words, topn=10)
    
    # (2) 尝试找类比
    print("\n\n" + "="*70)
    print("Part 2: Finding analogies")
    print("="*70)
    
    # 尝试一些可能的类比
    analogy_attempts = [
        (['woman', 'king'], ['man']),
        (['queen', 'man'], ['woman']),
        (['larger', 'small'], ['large']),
        (['expensive', 'low'], ['high']),
    ]
    
    for positive, negative in analogy_attempts:
        explorer.find_analogies(positive, negative, topn=5)


if __name__ == "__main__":
    from src.utils.training_workflow import load_trained_models
    
    # 加载模型
    en_trainer, de_trainer = load_trained_models()
    
    # 探索英语模型
    explore_task3_examples(en_trainer)
