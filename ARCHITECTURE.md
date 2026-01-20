# 系统架构设计 (OOA/OOD)

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                    Embeddings Project                        │
│                   (TDD-driven Design)                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │      Main Workflow Orchestrator      │
        │  (Facade Pattern - 简化复杂交互)      │
        └──────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐     ┌──────────────┐
│ Data Layer   │      │ Model Layer  │     │ Eval Layer   │
└──────────────┘      └──────────────┘     └──────────────┘
```

---

## 层次架构 (Layered Architecture)

### Layer 1: Data Layer (数据层)
**职责**: 数据获取、预处理、词典管理

```
┌─────────────────────────────────────────────┐
│              Data Layer                      │
├─────────────────────────────────────────────┤
│                                              │
│  ┌──────────────────────────────────────┐  │
│  │         DataLoader                   │  │
│  │  - download_corpus()                 │  │
│  │  - load_parallel_corpus()            │  │
│  │  - tokenize()                        │  │
│  │  - preprocess()                      │  │
│  └──────────────────────────────────────┘  │
│                    ▲                         │
│                    │ uses                    │
│  ┌──────────────────────────────────────┐  │
│  │    BilingualDictionary               │  │
│  │  - load_dictionary()                 │  │
│  │  - create_word_pairs()               │  │
│  │  - split_train_test()                │  │
│  └──────────────────────────────────────┘  │
│                                              │
└─────────────────────────────────────────────┘
```

**设计原则**:
- **SRP**: 每个类只负责一种数据类型
- **OCP**: 通过继承扩展新的数据源
- **DIP**: 依赖抽象接口而非具体实现

---

### Layer 2: Model Layer (模型层)
**职责**: 模型训练、词嵌入、跨语言投影

```
┌─────────────────────────────────────────────┐
│              Model Layer                     │
├─────────────────────────────────────────────┤
│                                              │
│  ┌──────────────────────────────────────┐  │
│  │       Word2VecTrainer                │  │
│  │  - train()                           │  │
│  │  - save_model()                      │  │
│  │  - load_model()                      │  │
│  │  - get_most_similar()                │  │
│  │  - compute_analogy()                 │  │
│  └──────────────────────────────────────┘  │
│                    │                         │
│                    │ collaborates            │
│                    ▼                         │
│  ┌──────────────────────────────────────┐  │
│  │     EmbeddingProjector               │  │
│  │  - learn_projection_matrix()         │  │
│  │  - project()                         │  │
│  │  - translate_word()                  │  │
│  │  - batch_translate()                 │  │
│  └──────────────────────────────────────┘  │
│                                              │
└─────────────────────────────────────────────┘
```

**设计模式**:
- **Strategy Pattern**: 不同的投影算法可替换
- **Factory Pattern**: 创建不同类型的模型

---

### Layer 3: Evaluation Layer (评估层)
**职责**: 翻译质量评估、错误分析

```
┌─────────────────────────────────────────────┐
│           Evaluation Layer                   │
├─────────────────────────────────────────────┤
│                                              │
│  ┌──────────────────────────────────────┐  │
│  │    TranslationEvaluator              │  │
│  │  - evaluate_top_k()                  │  │
│  │  - calculate_accuracy()              │  │
│  │  - analyze_errors()                  │  │
│  │  - generate_report()                 │  │
│  └──────────────────────────────────────┘  │
│                                              │
└─────────────────────────────────────────────┘
```

---

### Layer 4: Visualization Layer (可视化层)
**职责**: 词嵌入可视化

```
┌─────────────────────────────────────────────┐
│         Visualization Layer                  │
├─────────────────────────────────────────────┤
│                                              │
│  ┌──────────────────────────────────────┐  │
│  │    EmbeddingVisualizer               │  │
│  │  - reduce_dimensions()               │  │
│  │  - plot_words()                      │  │
│  │  - plot_analogies()                  │  │
│  └──────────────────────────────────────┘  │
│                                              │
└─────────────────────────────────────────────┘
```

---

## 类图详解

### DataLoader 类图

```
┌────────────────────────────────────────┐
│          DataLoader                     │
├────────────────────────────────────────┤
│ - _tokenizer: Tokenizer                │
│ - _lemmatizer: Optional[Lemmatizer]    │
├────────────────────────────────────────┤
│ + __init__(lemmatize: bool)            │
│ + download_corpus(url: str) -> Path    │
│ + load_parallel_corpus(path) -> Tuple  │
│ + tokenize(text: str) -> List[str]     │
│ + preprocess(sentences, ...) -> List   │
└────────────────────────────────────────┘
```

**关键设计决策**:
- 使用 `_tokenizer` 作为组合对象（Composition over Inheritance）
- `lemmatize` 作为可选功能，支持策略模式
- 返回类型明确，使用 Type Hints

---

### Word2VecTrainer 类图

```
┌────────────────────────────────────────┐
│        Word2VecTrainer                  │
├────────────────────────────────────────┤
│ - _model: Optional[Word2Vec]           │
│ - _config: Word2VecConfig              │
├────────────────────────────────────────┤
│ + __init__(config: Word2VecConfig)     │
│ + train(data: List[List[str]]) -> Model│
│ + save_model(path: Path) -> None       │
│ + load_model(path: Path) -> None       │
│ + get_most_similar(word, n) -> List    │
│ + compute_analogy(pos, neg) -> List    │
│ + get_vector(word: str) -> np.ndarray  │
│ + vocabulary -> Set[str]               │
└────────────────────────────────────────┘
```

**SOLID 原则应用**:
- **SRP**: 只负责 Word2Vec 相关操作
- **OCP**: 通过 config 扩展新参数
- **LSP**: 可以子类化支持其他嵌入模型
- **ISP**: 接口精简，不强迫实现不需要的方法
- **DIP**: 依赖 Word2VecConfig 抽象

---

### EmbeddingProjector 类图

```
┌────────────────────────────────────────┐
│       EmbeddingProjector                │
├────────────────────────────────────────┤
│ - _projection_matrix: Optional[ndarray]│
│ - _source_model: Word2Vec              │
│ - _target_model: Word2Vec              │
├────────────────────────────────────────┤
│ + __init__(source_model, target_model) │
│ + learn_projection_matrix(X, Z) -> None│
│ + project(vector: ndarray) -> ndarray  │
│ + translate_word(word, k) -> List[str] │
│ + batch_translate(words) -> List       │
│ + save_matrix(path: Path) -> None      │
│ + load_matrix(path: Path) -> None      │
└────────────────────────────────────────┘
```

**设计亮点**:
- 封装了投影矩阵的学习和应用
- 支持单词翻译和批量翻译
- 可以持久化投影矩阵

---

## 交互序列图 (Sequence Diagram)

### 完整翻译流程

```
User            DataLoader      Word2VecTrainer   BilingualDict    Projector      Evaluator
 │                  │                  │                │              │              │
 │ 1. load data     │                  │                │              │              │
 ├─────────────────>│                  │                │              │              │
 │                  │                  │                │              │              │
 │ 2. train models  │                  │                │              │              │
 ├──────────────────┼─────────────────>│                │              │              │
 │                  │                  │                │              │              │
 │ 3. load dict     │                  │                │              │              │
 ├──────────────────┼──────────────────┼───────────────>│              │              │
 │                  │                  │                │              │              │
 │ 4. create pairs  │                  │                │              │              │
 │<─────────────────┼──────────────────┼────────────────┤              │              │
 │                  │                  │                │              │              │
 │ 5. learn matrix  │                  │                │              │              │
 ├──────────────────┼──────────────────┼────────────────┼─────────────>│              │
 │                  │                  │                │              │              │
 │ 6. translate     │                  │                │              │              │
 ├──────────────────┼──────────────────┼────────────────┼─────────────>│              │
 │                  │                  │                │              │              │
 │ 7. evaluate      │                  │                │              │              │
 ├──────────────────┼──────────────────┼────────────────┼──────────────┼─────────────>│
 │                  │                  │                │              │              │
 │ 8. results       │                  │                │              │              │
 │<─────────────────┴──────────────────┴────────────────┴──────────────┴──────────────┘
```

---

## 依赖关系图

```
┌──────────────┐
│    Config    │  ← 所有组件都依赖配置
└──────┬───────┘
       │
       ▼
┌──────────────┐      ┌──────────────────┐
│  DataLoader  │─────>│ Word2VecTrainer  │
└──────────────┘      └────────┬─────────┘
       │                       │
       │                       │
       ▼                       ▼
┌──────────────────┐    ┌─────────────────────┐
│ BilingualDict    │───>│ EmbeddingProjector  │
└──────────────────┘    └──────────┬──────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │ TranslationEvaluator│
                         └─────────────────────┘
```

**依赖方向**: 从高层到低层，遵循依赖倒置原则

---

## 测试架构

```
┌─────────────────────────────────────────┐
│           Test Suite                     │
├─────────────────────────────────────────┤
│                                          │
│  Unit Tests (单元测试)                   │
│  ├── test_data_loader.py                │
│  ├── test_word2vec_trainer.py           │
│  ├── test_bilingual_dict.py             │
│  ├── test_embedding_projector.py        │
│  └── test_translator_evaluator.py       │
│                                          │
│  Integration Tests (集成测试)            │
│  ├── test_end_to_end_translation.py     │
│  └── test_workflow_integration.py       │
│                                          │
│  Fixtures (共享测试数据)                 │
│  └── conftest.py                         │
│      ├── sample_corpus                   │
│      ├── mock_models                     │
│      └── test_dictionary                 │
│                                          │
└─────────────────────────────────────────┘
```

---

## 配置管理架构

```python
# Centralized Configuration
Config
├── CorpusConfig (数据源配置)
├── DictionaryConfig (词典配置)
├── Word2VecConfig (模型配置)
├── ProjectionConfig (投影配置)
└── VisualizationConfig (可视化配置)
```

**优势**:
- 单一真相来源 (Single Source of Truth)
- 易于修改超参数
- 支持不同环境配置（开发、测试、生产）

---

## 扩展性设计

### 1. 支持新的嵌入模型
```python
class EmbeddingTrainer(ABC):
    @abstractmethod
    def train(self, data): pass
    
class Word2VecTrainer(EmbeddingTrainer): ...
class FastTextTrainer(EmbeddingTrainer): ...
class GloVeTrainer(EmbeddingTrainer): ...
```

### 2. 支持新的投影算法
```python
class ProjectionStrategy(ABC):
    @abstractmethod
    def learn_projection(self, X, Z): pass

class LeastSquaresProjection(ProjectionStrategy): ...
class OrthogonalProjection(ProjectionStrategy): ...
class NeuralProjection(ProjectionStrategy): ...
```

### 3. 支持新的评估指标
```python
class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, predictions, ground_truth): pass

class TopKAccuracy(Evaluator): ...
class MRR(Evaluator): ...  # Mean Reciprocal Rank
class BLEU(Evaluator): ...
```

---

## 设计模式总结

| 模式 | 应用位置 | 目的 |
|------|---------|------|
| **Strategy** | EmbeddingProjector | 不同投影算法可替换 |
| **Factory** | Trainer创建 | 统一创建不同模型 |
| **Singleton** | Config | 全局唯一配置实例 |
| **Facade** | Main Workflow | 简化复杂子系统交互 |
| **Composition** | DataLoader | 组合tokenizer而非继承 |

---

## 下一步

查看 [SPRINT_GUIDE.md](SPRINT_GUIDE.md) 开始TDD开发！
