# Cross-Lingual Word Embedding Projection

Complete implementation of cross-lingual word embedding alignment using Word2Vec and linear projection for EN-DE translation.

## Assignment Tasks (25 Points)

This project implements all 11 tasks from the NLP assignment:

**Task 1** (1P): Load and preprocess parallel corpus  
**Task 2** (1P): Train Word2Vec Skip-gram models  
**Task 3** (3P): Explore models (similarities, analogies)  
**Task 4** (2P): Create bilingual dictionary  
**Task 5** (1P): Extract word vectors  
**Task 6** (3P): Learn projection matrix  
**Task 7** (2P): Translate test words  
**Task 8** (3P): Evaluate with Precision@K  
**Task 9** (2P): Analysis and discussion  
**Task 10** (5P): Lemmatization experiment  
**Task 11** (2P): Visualization with PCA  

## How to View Task Results

### Quick Start

1. **Install dependencies:**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

2. **Open main notebook:**
```bash
jupyter notebook embeddings.ipynb
```

3. **Run all cells** to see results for Tasks 1-9 and 11

4. **View Task 10 (Lemmatization):**
```bash
jupyter notebook notebooks/task10_lemmatization_experiment.ipynb
```

### Task Verification Checklist

Open `embeddings.ipynb` and verify each task:

- ✅ **Task 1**: Cell 5 shows corpus loaded (2000 sentence pairs)
- ✅ **Task 2**: Cell 7 shows models trained (1386 EN, 1360 DE words)
- ✅ **Task 3**: Cell 9 shows word similarities and analogies
- ✅ **Task 4**: Cell 11 shows dictionary created (1268 word pairs, 80/20 split)
- ✅ **Task 5-6**: Cell 13 shows vectors extracted and projection matrix learned
- ✅ **Task 7-8**: Cell 15 shows translations and P@K metrics (P@1=0.4%, P@5=1.2%, P@10=2.0%)
- ✅ **Task 9**: Cells 18-21 contain comprehensive discussion (4 sections including parallel data analysis)
- ✅ **Task 10**: See `notebooks/task10_lemmatization_experiment.ipynb` for full experiment
- ✅ **Task 11**: Cell 17 shows t-SNE visualization with observations

### Task 10 (Lemmatization Experiment)

Open the separate notebook:
```bash
jupyter notebook notebooks/task10_lemmatization_experiment.ipynb
```

Run all cells to see:
- Baseline vs lemmatized comparison
- Vocabulary statistics
- P@K metric changes (lemmatization decreased performance by 32-39%)
- Visualization and detailed analysis

## Project Results

**Performance:**
- Training alignment: 0.941 cosine similarity
- P@1: 0.4% | P@5: 1.2% | P@10: 2.0%

**Task 10 Finding:**
- Lemmatization decreased performance on small corpus (2k sentences)
- Morphological variations contain valuable semantic information

## File Structure

```
embeddings_project/
├── embeddings.ipynb              # Main notebook (Tasks 1-9, 11)
├── notebooks/
│   └── task10_lemmatization_experiment.ipynb  # Task 10
├── src/                          # Implementation modules
├── data/                         # Corpus and alignments
├── models/                       # Trained Word2Vec models
└── requirements.txt
```

## Dependencies

- Python 3.11+
- gensim 4.4.0 (Word2Vec)
- numpy 2.3.5
- scikit-learn 1.7.2 (PCA, t-SNE)
- matplotlib 3.10.7
- nltk 3.9.2 (lemmatization)
- jupyter 1.1.1

## Author

Ming-Han  
Technical University of Munich (TUM)  
Winter 2025
