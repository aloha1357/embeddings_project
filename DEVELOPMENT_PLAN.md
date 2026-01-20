# Word Embeddings Project - Development Plan (TDD Approach)

## 项目概述
使用TDD（测试驱动开发）和OOP原则开发跨语言词嵌入映射系统

## 系统架构设计 (OOD)

### 核心组件

```
embeddings_project/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py          # 数据加载和预处理
│   │   └── bilingual_dict.py       # 双语词典管理
│   ├── models/
│   │   ├── __init__.py
│   │   ├── word2vec_trainer.py     # Word2Vec训练器
│   │   └── embedding_projector.py  # 跨语言投影
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── translator_evaluator.py # 翻译评估
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── embedding_visualizer.py # 可视化
│   └── utils/
│       ├── __init__.py
│       └── config.py                # 配置管理
├── tests/
│   ├── test_data_loader.py
│   ├── test_word2vec_trainer.py
│   ├── test_bilingual_dict.py
│   ├── test_embedding_projector.py
│   ├── test_translator_evaluator.py
│   └── test_embedding_visualizer.py
├── notebooks/
│   └── embeddings_analysis.ipynb    # 最终分析笔记本
├── data/                            # 数据目录
├── models/                          # 保存的模型
└── requirements.txt
```

### 类图和职责

#### 1. DataLoader (data/data_loader.py)
**职责**: 下载、解析和预处理语料库
```python
class DataLoader:
    - download_corpus(url: str) -> Path
    - load_parallel_corpus(file_path: Path) -> Tuple[List, List]
    - tokenize(text: str) -> List[str]
    - preprocess(sentences: List[str], lowercase: bool, lemmatize: bool) -> List[List[str]]
```

#### 2. Word2VecTrainer (models/word2vec_trainer.py)
**职责**: 训练和管理Word2Vec模型
```python
class Word2VecTrainer:
    - train(data: List[List[str]], vector_size: int, window: int) -> Word2Vec
    - save_model(model: Word2Vec, path: Path) -> None
    - load_model(path: Path) -> Word2Vec
    - get_most_similar(word: str, topn: int) -> List[Tuple[str, float]]
    - compute_analogy(positive: List[str], negative: List[str]) -> List[Tuple[str, float]]
```

#### 3. BilingualDictionary (data/bilingual_dict.py)
**职责**: 管理双语词典和创建词对
```python
class BilingualDictionary:
    - load_dictionary(file_path: Path) -> Dict[str, List[str]]
    - create_word_pairs(source_vocab: List[str], target_vocab: Set[str]) -> List[Tuple[str, str]]
    - split_train_test(pairs: List[Tuple], train_size: int) -> Tuple[List, List]
```

#### 4. EmbeddingProjector (models/embedding_projector.py)
**职责**: 学习跨语言投影矩阵并进行词汇投影
```python
class EmbeddingProjector:
    - learn_projection_matrix(X: np.ndarray, Z: np.ndarray) -> np.ndarray
    - project(vector: np.ndarray) -> np.ndarray
    - translate_word(word: str, source_model: Word2Vec, target_model: Word2Vec, topn: int) -> List[str]
```

#### 5. TranslationEvaluator (evaluation/translator_evaluator.py)
**职责**: 评估翻译准确率
```python
class TranslationEvaluator:
    - evaluate_top_k(predictions: List[List[str]], ground_truth: List[str], k: int) -> float
    - analyze_errors(predictions: List, ground_truth: List) -> Dict
```

#### 6. EmbeddingVisualizer (visualization/embedding_visualizer.py)
**职责**: 可视化词嵌入
```python
class EmbeddingVisualizer:
    - reduce_dimensions(vectors: np.ndarray, n_components: int) -> np.ndarray
    - plot_words(words: List[str], vectors: np.ndarray) -> None
```

---

## Sprint 规划 (TDD)

### **Sprint 0: 项目初始化** (0.5天)
- [x] 创建项目结构
- [ ] 设置虚拟环境和依赖
- [ ] 配置pytest
- [ ] 创建配置文件

### **Sprint 1: 数据加载模块** (1天)
**Tasks**: Task 1

**TDD Steps**:
1. 写测试: `test_data_loader.py`
   - test_download_corpus()
   - test_load_parallel_corpus()
   - test_tokenize()
   - test_preprocess_lowercase()
2. 实现: `data_loader.py`
3. 重构并通过所有测试

**验收标准**:
- 能正确下载和解析.tsv.gz文件
- 正确分离DE和EN语料
- tokenize和lowercase功能正常

---

### **Sprint 2: Word2Vec训练模块** (1天)
**Tasks**: Task 2

**TDD Steps**:
1. 写测试: `test_word2vec_trainer.py`
   - test_train_model()
   - test_save_and_load_model()
   - test_model_parameters()
2. 实现: `word2vec_trainer.py`
3. 重构并通过测试

**验收标准**:
- 能训练skipgram模型
- vector_size=100, window=5
- 能保存和加载模型

---

### **Sprint 3: 模型探索功能** (1天)
**Tasks**: Task 3

**TDD Steps**:
1. 写测试: `test_word2vec_trainer.py` (扩展)
   - test_get_most_similar()
   - test_compute_analogy()
2. 实现功能并在notebook中分析
3. 验证测试通过

**验收标准**:
- most_similar功能正常
- 能计算词类比
- 分析结果记录在notebook

---

### **Sprint 4: 双语词典模块** (1天)
**Tasks**: Task 4

**TDD Steps**:
1. 写测试: `test_bilingual_dict.py`
   - test_load_dictionary()
   - test_create_word_pairs()
   - test_filter_by_vocab()
   - test_split_train_test()
2. 实现: `bilingual_dict.py`
3. 重构并验证

**验收标准**:
- 加载en-de词典
- 生成约6200个有效词对
- 正确分割训练集(5000)和测试集(~1200)

---

### **Sprint 5: 词向量提取** (0.5天)
**Tasks**: Task 5

**TDD Steps**:
1. 写测试: `test_embedding_projector.py`
   - test_extract_word_vectors()
   - test_vector_shape()
2. 实现向量提取功能
3. 验证测试

**验收标准**:
- 正确提取英语和德语词向量
- X_english 和 Z_german shape正确

---

### **Sprint 6: 投影矩阵学习** (1天)
**Tasks**: Task 6

**TDD Steps**:
1. 写测试: `test_embedding_projector.py` (扩展)
   - test_learn_projection_matrix()
   - test_projection_matrix_shape()
   - test_project_vector()
2. 实现: `embedding_projector.py`
3. 验证最小二乘解

**验收标准**:
- 使用np.linalg.lstsq计算W
- W的形状正确(100x100)
- 投影功能正常

---

### **Sprint 7: 翻译功能** (1天)
**Tasks**: Task 7

**TDD Steps**:
1. 写测试: `test_embedding_projector.py` (扩展)
   - test_translate_word()
   - test_translate_multiple_words()
2. 实现翻译功能
3. 在notebook中分析错误

**验收标准**:
- 能翻译测试词列表
- 返回top-5预测
- 错误分析完成

---

### **Sprint 8: 评估模块** (1天)
**Tasks**: Task 8

**TDD Steps**:
1. 写测试: `test_translator_evaluator.py`
   - test_evaluate_top_k()
   - test_accuracy_calculation()
   - test_edge_cases()
2. 实现: `translator_evaluator.py`
3. 评估测试集

**验收标准**:
- 正确计算top-5准确率
- 测试集评估完成

---

### **Sprint 9: 词形还原实验** (1.5天)
**Tasks**: Task 10

**TDD Steps**:
1. 扩展测试: `test_data_loader.py`
   - test_lemmatize()
2. 实现lemmatization
3. 重新训练模型
4. 比较结果

**验收标准**:
- lemmatization功能正常
- 模型重新训练完成
- 结果对比分析完成

---

### **Sprint 10: 可视化模块** (1天)
**Tasks**: Task 11

**TDD Steps**:
1. 写测试: `test_embedding_visualizer.py`
   - test_reduce_dimensions()
   - test_plot_generation()
2. 实现: `embedding_visualizer.py`
3. 生成图表

**验收标准**:
- PCA降维正常
- scatter plot正确生成
- 观察分析完成

---

### **Sprint 11: 理论分析和文档** (0.5天)
**Tasks**: Task 9

**交付物**:
- 回答Task 9的理论问题
- 完善文档
- 整理最终notebook

---

## 测试策略

### 单元测试
- 每个类的每个公共方法都要有测试
- 使用pytest fixtures管理测试数据
- Mock外部依赖（文件下载等）

### 集成测试
- 端到端翻译流程测试
- 多组件协作测试

### 测试覆盖率目标
- 最低80%代码覆盖率
- 核心逻辑100%覆盖

---

## 依赖管理

```
gensim>=4.3.0
nltk>=3.8
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
pytest>=7.4.0
pytest-cov>=4.1.0
requests>=2.31.0
```

---

## 配置管理

使用`config.py`集中管理：
- 模型超参数
- 文件路径
- API URLs
- 常量

---

## 开发原则

1. **Red-Green-Refactor**: 先写失败的测试 → 最小实现 → 重构
2. **SOLID原则**: 单一职责、开闭原则等
3. **DRY**: 不重复代码
4. **Clean Code**: 清晰命名、小函数、低耦合

---

## 下一步行动

1. 执行Sprint 0: 初始化项目结构
2. 设置测试环境
3. 开始Sprint 1: DataLoader的TDD开发
