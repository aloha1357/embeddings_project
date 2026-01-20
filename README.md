# Word Embeddings Project (TDD Approach)

基于测试驱动开发（TDD）和面向对象设计原则的跨语言词嵌入映射系统。

## 项目结构

```
embeddings_project/
├── src/                    # 源代码
│   ├── data/              # 数据处理模块
│   ├── models/            # 模型训练和投影
│   ├── evaluation/        # 评估模块
│   ├── visualization/     # 可视化模块
│   └── utils/             # 工具和配置
├── tests/                 # 测试代码
├── notebooks/             # Jupyter notebooks
├── data/                  # 数据目录
├── models/                # 保存的模型
└── DEVELOPMENT_PLAN.md    # 详细开发计划

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行测试

```bash
pytest
```

### 3. Sprint开发

按照 DEVELOPMENT_PLAN.md 中的11个sprints逐步开发。

## 开发原则

- **TDD**: 先写测试，再写实现
- **OOP**: 使用面向对象设计
- **SOLID**: 遵循SOLID原则
- **Clean Code**: 保持代码清晰可维护

## 当前进度

- [x] Sprint 0: 项目初始化
- [ ] Sprint 1: 数据加载模块
- [ ] Sprint 2: Word2Vec训练模块
- [ ] ...

详见 [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)
