"""
Training workflow script
Integrates DataLoader and Word2VecTrainer to prepare and train models
Sprint 3: Task 2 & 3
"""
from pathlib import Path
import gzip
from typing import Tuple, List
from src.data.data_loader import DataLoader
from src.models.word2vec_trainer import Word2VecTrainer
from src.utils.config import corpus_config, word2vec_config


class TrainingWorkflow:
    """
    完整的训练工作流程
    
    集成 DataLoader 和 Word2VecTrainer 来：
    1. 下载和解压数据
    2. 预处理数据
    3. 训练英语和德语模型
    4. 保存模型
    """
    
    def __init__(self):
        """初始化工作流"""
        self.data_loader = DataLoader()
        self.en_trainer = Word2VecTrainer(word2vec_config)
        self.de_trainer = Word2VecTrainer(word2vec_config)
    
    def download_and_extract_corpus(self) -> Path:
        """
        下载并解压语料库
        
        Returns:
            解压后的 TSV 文件路径
        """
        gz_path = corpus_config.raw_file_path
        
        # 如果已经下载，跳过
        if gz_path.exists():
            print(f"Corpus already downloaded: {gz_path}")
        else:
            print(f"Downloading corpus from {corpus_config.url}...")
            self.data_loader.download_corpus(corpus_config.url, gz_path)
            print(f"Downloaded to {gz_path}")
        
        # 解压文件
        tsv_path = gz_path.parent / gz_path.stem  # 移除 .gz 扩展名
        
        if tsv_path.exists():
            print(f"Corpus already extracted: {tsv_path}")
        else:
            print(f"Extracting {gz_path}...")
            with gzip.open(gz_path, 'rb') as f_in:
                with open(tsv_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            print(f"Extracted to {tsv_path}")
        
        return tsv_path
    
    def prepare_data(
        self, 
        corpus_path: Path,
        lowercase: bool = True
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """
        准备训练数据
        
        Args:
            corpus_path: 语料库文件路径
            lowercase: 是否转小写
        
        Returns:
            (英语数据, 德语数据) 元组
        """
        print(f"Loading parallel corpus from {corpus_path}...")
        en_data, de_data = self.data_loader.load_parallel_corpus(corpus_path)
        
        print(f"Loaded {len(en_data)} sentence pairs")
        print(f"English example: {' '.join(en_data[0][:10])}...")
        print(f"German example: {' '.join(de_data[0][:10])}...")
        
        return en_data, de_data
    
    def train_models(
        self,
        en_data: List[List[str]],
        de_data: List[List[str]]
    ) -> Tuple[Word2VecTrainer, Word2VecTrainer]:
        """
        训练英语和德语模型
        
        Args:
            en_data: 英语训练数据
            de_data: 德语训练数据
        
        Returns:
            (英语训练器, 德语训练器) 元组
        """
        print("\nTraining English model...")
        print(f"Parameters: vector_size={word2vec_config.vector_size}, "
              f"window={word2vec_config.window}, sg={word2vec_config.sg}")
        
        self.en_trainer.train(en_data)
        print(f"English model trained! Vocabulary size: {len(self.en_trainer.vocabulary)}")
        
        print("\nTraining German model...")
        self.de_trainer.train(de_data)
        print(f"German model trained! Vocabulary size: {len(self.de_trainer.vocabulary)}")
        
        return self.en_trainer, self.de_trainer
    
    def save_models(self) -> Tuple[Path, Path]:
        """
        保存训练好的模型
        
        Returns:
            (英语模型路径, 德语模型路径) 元组
        """
        en_path = word2vec_config.en_model_path
        de_path = word2vec_config.de_model_path
        
        print(f"\nSaving English model to {en_path}...")
        self.en_trainer.save_model(en_path)
        
        print(f"Saving German model to {de_path}...")
        self.de_trainer.save_model(de_path)
        
        print("Models saved successfully!")
        return en_path, de_path
    
    def run_full_workflow(self) -> Tuple[Word2VecTrainer, Word2VecTrainer]:
        """
        运行完整工作流程
        
        Returns:
            (英语训练器, 德语训练器) 元组
        """
        print("=" * 70)
        print("Starting Training Workflow")
        print("=" * 70)
        
        # 1. 下载和解压
        corpus_path = self.download_and_extract_corpus()
        
        # 2. 准备数据
        en_data, de_data = self.prepare_data(corpus_path)
        
        # 3. 训练模型
        en_trainer, de_trainer = self.train_models(en_data, de_data)
        
        # 4. 保存模型
        self.save_models()
        
        print("\n" + "=" * 70)
        print("Training Workflow Completed!")
        print("=" * 70)
        
        return en_trainer, de_trainer


def load_trained_models() -> Tuple[Word2VecTrainer, Word2VecTrainer]:
    """
    加载已训练的模型
    
    Returns:
        (英语训练器, 德语训练器) 元组
    
    Raises:
        FileNotFoundError: 如果模型文件不存在
    """
    en_trainer = Word2VecTrainer(word2vec_config)
    de_trainer = Word2VecTrainer(word2vec_config)
    
    en_path = word2vec_config.en_model_path
    de_path = word2vec_config.de_model_path
    
    if not en_path.exists():
        raise FileNotFoundError(
            f"English model not found at {en_path}. "
            f"Run TrainingWorkflow().run_full_workflow() first."
        )
    
    if not de_path.exists():
        raise FileNotFoundError(
            f"German model not found at {de_path}. "
            f"Run TrainingWorkflow().run_full_workflow() first."
        )
    
    print(f"Loading English model from {en_path}...")
    en_trainer.load_model(en_path)
    
    print(f"Loading German model from {de_path}...")
    de_trainer.load_model(de_path)
    
    print(f"Models loaded successfully!")
    print(f"English vocabulary: {len(en_trainer.vocabulary)} words")
    print(f"German vocabulary: {len(de_trainer.vocabulary)} words")
    
    return en_trainer, de_trainer


if __name__ == "__main__":
    # 运行完整工作流程
    workflow = TrainingWorkflow()
    en_trainer, de_trainer = workflow.run_full_workflow()
    
    # 简单测试
    print("\n" + "=" * 70)
    print("Quick Test")
    print("=" * 70)
    
    test_words = ['small', 'expensive', 'government']
    for word in test_words:
        if word in en_trainer.vocabulary:
            similar = en_trainer.get_most_similar(word, topn=3)
            print(f"\nMost similar to '{word}':")
            for sim_word, score in similar:
                print(f"  {sim_word}: {score:.3f}")
