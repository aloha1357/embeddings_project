# Sprint 执行指南

## Sprint 0: 项目初始化 ✅ 已完成

已创建：
- 完整的项目目录结构
- 所有必要的 `__init__.py` 文件
- `requirements.txt` 依赖管理
- `pytest.ini` 测试配置
- `config.py` 配置管理系统
- `DEVELOPMENT_PLAN.md` 详细开发计划
- `README.md` 项目说明

---

## 下一步：Sprint 1 - 数据加载模块 (TDD)

### TDD工作流程示例

#### Step 1: 先写测试 (Red 阶段)

创建 `tests/test_data_loader.py`:

```python
"""
测试 DataLoader 类
遵循 TDD: 先写测试，定义期望行为
"""
import pytest
from pathlib import Path
from src.data.data_loader import DataLoader


class TestDataLoader:
    """DataLoader 测试套件"""
    
    def test_tokenize_simple_sentence(self):
        """测试基本tokenization"""
        loader = DataLoader()
        text = "Hello, world! This is a test."
        tokens = loader.tokenize(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert 'hello' in tokens  # 应该lowercase
        assert '!' not in tokens  # 标点应该被移除或单独处理
    
    def test_preprocess_lowercase(self):
        """测试lowercase预处理"""
        loader = DataLoader()
        sentences = ["Hello World", "TESTING"]
        result = loader.preprocess(sentences, lowercase=True, lemmatize=False)
        
        assert all(word.islower() for sent in result for word in sent)
    
    def test_load_parallel_corpus(self, tmp_path):
        """测试加载平行语料库"""
        # 创建临时测试文件
        test_file = tmp_path / "test_corpus.tsv"
        test_file.write_text("Hello world\tHallo Welt\n")
        
        loader = DataLoader()
        en_data, de_data = loader.load_parallel_corpus(test_file)
        
        assert len(en_data) == 1
        assert len(de_data) == 1
        assert isinstance(en_data[0], list)  # tokenized
```

#### Step 2: 运行测试（应该失败）

```bash
pytest tests/test_data_loader.py
```

**预期**: 所有测试失败 ❌ (因为还没实现)

#### Step 3: 最小实现 (Green 阶段)

创建 `src/data/data_loader.py`:

```python
"""
DataLoader: 负责加载和预处理语料库数据
遵循单一职责原则 (SRP)
"""
from pathlib import Path
from typing import List, Tuple
import re
from nltk.tokenize import word_tokenize


class DataLoader:
    """加载和预处理语料库数据"""
    
    def tokenize(self, text: str) -> List[str]:
        """
        对文本进行tokenization
        
        Args:
            text: 输入文本
        
        Returns:
            token列表
        """
        # 最小实现：让测试通过
        tokens = word_tokenize(text.lower())
        # 移除标点
        tokens = [t for t in tokens if re.match(r'\w+', t)]
        return tokens
    
    def preprocess(
        self, 
        sentences: List[str], 
        lowercase: bool = True, 
        lemmatize: bool = False
    ) -> List[List[str]]:
        """
        预处理句子列表
        
        Args:
            sentences: 句子列表
            lowercase: 是否转小写
            lemmatize: 是否词形还原
        
        Returns:
            处理后的token列表的列表
        """
        result = []
        for sentence in sentences:
            tokens = self.tokenize(sentence) if lowercase else word_tokenize(sentence)
            result.append(tokens)
        return result
    
    def load_parallel_corpus(self, file_path: Path) -> Tuple[List[List[str]], List[List[str]]]:
        """
        加载平行语料库
        
        Args:
            file_path: TSV文件路径
        
        Returns:
            (英语数据, 德语数据) 元组
        """
        en_data = []
        de_data = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    en_text, de_text = parts
                    en_data.append(self.tokenize(en_text))
                    de_data.append(self.tokenize(de_text))
        
        return en_data, de_data
```

#### Step 4: 再次运行测试（应该通过）

```bash
pytest tests/test_data_loader.py -v
```

**预期**: 所有测试通过 ✅

#### Step 5: 重构 (Refactor 阶段)

- 改进代码质量
- 提取重复逻辑
- 添加文档字符串
- 优化性能

#### Step 6: 添加更多测试

继续添加边界情况测试：

```python
def test_tokenize_empty_string(self):
    """测试空字符串"""
    loader = DataLoader()
    tokens = loader.tokenize("")
    assert tokens == []

def test_tokenize_with_numbers(self):
    """测试包含数字的文本"""
    loader = DataLoader()
    tokens = loader.tokenize("There are 123 items")
    assert '123' in tokens

def test_preprocess_maintains_sentence_structure(self):
    """测试预处理保持句子结构"""
    loader = DataLoader()
    sentences = ["First sentence", "Second sentence"]
    result = loader.preprocess(sentences)
    assert len(result) == 2
```

---

## 如何开始 Sprint 1

### 1. 安装依赖

```bash
cd embeddings_project
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

### 2. 创建第一个测试文件

```bash
# 在 tests/ 目录下创建 test_data_loader.py
# 复制上面的测试代码
```

### 3. 运行测试（预期失败）

```bash
pytest tests/test_data_loader.py
```

### 4. 创建实现文件

```bash
# 在 src/data/ 目录下创建 data_loader.py
# 复制上面的实现代码
```

### 5. 再次运行测试（预期通过）

```bash
pytest tests/test_data_loader.py -v
```

### 6. 查看测试覆盖率

```bash
pytest --cov=src --cov-report=html
# 打开 htmlcov/index.html 查看覆盖率报告
```

---

## TDD 最佳实践

### 1. Red-Green-Refactor 循环

```
🔴 Red    → 写一个失败的测试
🟢 Green  → 写最小代码让测试通过
🔵 Refactor → 重构代码，保持测试通过
```

### 2. 测试命名规范

```python
def test_[function_name]_[scenario]_[expected_result]():
    # 例如：
    # test_tokenize_empty_string_returns_empty_list()
    # test_preprocess_with_lowercase_converts_all_words()
```

### 3. AAA 模式 (Arrange-Act-Assert)

```python
def test_example():
    # Arrange: 准备测试数据
    loader = DataLoader()
    text = "Test input"
    
    # Act: 执行被测试的操作
    result = loader.tokenize(text)
    
    # Assert: 验证结果
    assert len(result) > 0
```

### 4. 使用 Fixtures 共享测试数据

```python
@pytest.fixture
def sample_loader():
    """共享的 DataLoader 实例"""
    return DataLoader()

def test_with_fixture(sample_loader):
    result = sample_loader.tokenize("test")
    assert len(result) > 0
```

---

## Sprint 检查清单

每个 Sprint 完成后检查：

- [ ] 所有测试通过 (`pytest`)
- [ ] 测试覆盖率 > 80% (`pytest --cov`)
- [ ] 代码遵循 PEP 8 (`flake8` 或 `black`)
- [ ] 所有公共方法有文档字符串
- [ ] 没有重复代码 (DRY原则)
- [ ] 每个类职责单一 (SRP原则)
- [ ] 提交代码到版本控制

---

## 准备好开始了吗？

**建议顺序**:

1. ✅ Sprint 0: 项目初始化（已完成）
2. ⏭️ Sprint 1: DataLoader - 开始TDD开发！
3. Sprint 2: Word2VecTrainer
4. Sprint 3: 模型探索
5. ...

**下一步行动**：
```bash
# 告诉我你准备好了，我们就开始 Sprint 1 的 TDD 开发！
```

要开始 Sprint 1 吗？我会一步步带你完成 TDD 开发流程。
