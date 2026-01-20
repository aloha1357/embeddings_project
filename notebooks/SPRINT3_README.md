# Sprint 3: Model Exploration - Quick Start

## Overview

Sprint 3 implements Task 2 & 3:
- Download and prepare corpus data
- Train Word2Vec models for English and German
- Explore trained models (similar words, analogies)

## Files Created

```
src/utils/
  ├── training_workflow.py    # Complete training pipeline
  └── model_explorer.py        # Model exploration tools

notebooks/
  └── task3_model_exploration.ipynb  # Interactive exploration

scripts/
  └── test_sprint3.py          # Quick test script
```

## Quick Start

### Option 1: Quick Test (Small Data)

```bash
# Test with small dataset (fast, for verification)
python scripts/test_sprint3.py
```

### Option 2: Train on Real Data

```bash
# Download corpus and train models (takes ~5-10 minutes)
python -m src.utils.training_workflow
```

This will:
1. Download news-commentary corpus (~45MB)
2. Extract and preprocess data
3. Train English model (vector_size=100, window=5)
4. Train German model
5. Save models to `models/` directory

### Option 3: Interactive Exploration (Recommended)

```bash
# Open Jupyter notebook
jupyter notebook notebooks/task3_model_exploration.ipynb
```

The notebook provides:
- Automatic model training (if not already done)
- Guided exploration of Task 3 requirements
- Interactive cells for your own experiments

## What to Explore (Task 3)

### Part 1: Similar Words

Explore similar words to:
- `small`, `expensive`, `coordinating`

Questions:
- Are they all synonyms?
- What patterns do you notice?
- How would you explain this?

### Part 2: Word Analogies

Try to find analogies like:
- woman : queen :: man : king

Questions:
- What analogies can you find?
- What's your strategy?
- Why do some work and others don't?

## Usage Examples

### Load Trained Models

```python
from src.utils.training_workflow import load_trained_models

en_trainer, de_trainer = load_trained_models()
```

### Explore Words

```python
from src.utils.model_explorer import ModelExplorer

explorer = ModelExplorer(en_trainer)
explorer.explore_word('government', topn=10)
```

### Find Analogies

```python
# france : president :: germany : ?
explorer.find_analogies(
    positive=['france', 'president'],
    negative=['germany'],
    topn=5
)
```

### Check Similarity

```python
# Get similarity between two words
explorer.analyze_word_relationships([
    ('france', 'germany'),
    ('president', 'minister')
])
```

## Model Information

- **Corpus**: news-commentary v18 (EN-DE)
- **Vector Size**: 100 dimensions
- **Window**: 5 words
- **Algorithm**: Skip-gram (sg=1)
- **Min Count**: 5 occurrences

## Expected Vocabulary Size

- English: ~40,000-50,000 words
- German: ~60,000-70,000 words

(Exact numbers depend on corpus version)

## Troubleshooting

### Models not found?

```bash
# Train models first
python -m src.utils.training_workflow
```

### Word not in vocabulary?

```python
# Check if word exists
if 'word' in en_trainer.vocabulary:
    print("Word found!")
else:
    print("Try lowercase or different form")
```

### Import errors?

```python
# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path().absolute().parent))
```

## Next Steps

After exploring:
- Document your findings in the notebook
- Answer Task 3 questions
- Try additional word combinations
- Experiment with different topn values

Ready for **Sprint 4** - Bilingual Dictionary! 🚀
