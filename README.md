# PubMed RCT - Sentence Classification in Medical Abstracts

This project classifies sentences from PubMed medical abstracts into 5 categories:
**BACKGROUND**, **OBJECTIVE**, **METHODS**, **RESULTS**, **CONCLUSIONS**.

It is based on the [PubMed 200k RCT](https://github.com/Franck-Dernoncourt/pubmed-rct) dataset.

## Project Structure

```
PubMed_RCT_Project/
├── data/                          # Dataset (PubMed RCT)
│   └── pubmed-rct/
│       ├── PubMed_20k_RCT/
│       └── PubMed_20k_RCT_numbers_replaced_with_at_sign/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Explore the dataset
│   ├── 02_baseline_model.ipynb    # TF-IDF + Naive Bayes
│   ├── 03_embeddings_model.ipynb  # GloVe embeddings model
│   ├── 04_deep_learning_model.ipynb  # Bidirectional LSTM
│   └── 05_evaluation_comparison.ipynb  # Compare all models
│
└── README.md
```

## Dataset

We use the **PubMed 20k RCT** subset (with numbers replaced by `@`), which contains:
- ~15,000 training sentences
- ~2,500 validation sentences
- ~2,500 test sentences

Each sentence is labeled with its role in the abstract (BACKGROUND, OBJECTIVE, etc.).

## Models

| # | Model | Description |
|---|-------|-------------|
| 1 | **Baseline** | TF-IDF (unigrams + bigrams) + Multinomial Naive Bayes |
| 2 | **Embeddings** | Pre-trained GloVe (100d) + GlobalAveragePooling + Dense |
| 3 | **Bi-LSTM** | Trainable embeddings + Bidirectional LSTM + Dense |

## How to Run

### Requirements

```
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

### GloVe Embeddings (for Notebook 03)

Download [GloVe 6B](https://nlp.stanford.edu/data/glove.6B.zip), extract `glove.6B.100d.txt`, and place it in the `data/` folder.

### Execution

Run the notebooks in order (01 through 05). Each notebook is self-contained and includes its own data loading code.

## References

- Dernoncourt, F., & Lee, J. Y. (2017). PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts. *IJCNLP*.
- Pennington, J., Socher, R., & Manning, C. (2014). GloVe: Global Vectors for Word Representation. *EMNLP*.
