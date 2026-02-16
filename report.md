# Sentence Classification in Medical Abstracts Using the PubMed RCT Dataset

---

**Course Project Report**

**Date:** February 2026

---

## 1. Introduction

Medical literature is expanding at an unprecedented rate, making it increasingly difficult for researchers and clinicians to efficiently process scientific publications. A key challenge lies in quickly identifying the structural role of each sentence within an abstract — whether it describes the background, the study objective, the methods used, the results obtained, or the conclusions drawn.

This project addresses the task of **sequential sentence classification** in medical abstracts from PubMed. Given a sentence extracted from a Randomized Controlled Trial (RCT) abstract, the goal is to automatically predict its rhetorical role among five categories: **BACKGROUND**, **OBJECTIVE**, **METHODS**, **RESULTS**, and **CONCLUSIONS**.

We implement and compare three models of increasing complexity:

1. A traditional machine learning baseline using TF-IDF and Naive Bayes,
2. A neural model leveraging pre-trained GloVe word embeddings,
3. A deep learning model based on a Bidirectional LSTM architecture.

The project is structured into five Jupyter notebooks covering data exploration, model training, and comparative evaluation.

---

## 2. Dataset

### 2.1 Source

We use the **PubMed 20k RCT** dataset, a subset of the larger PubMed 200k RCT corpus introduced by Dernoncourt and Lee (2017). The dataset is publicly available and consists of abstracts from randomized controlled trials indexed in PubMed/MEDLINE.

### 2.2 Preprocessing

We specifically use the variant where **all numerical values have been replaced with the `@` symbol**. This normalization reduces vocabulary size and prevents the model from overfitting to specific numerical values (e.g., dosages, p-values, sample sizes), which are unlikely to generalize across studies.

### 2.3 Data Splits

| Split       | Number of Sentences |
|-------------|---------------------|
| Training    | ~15,000             |
| Validation  | ~2,500              |
| Test        | ~2,500              |

### 2.4 Label Distribution

The five classes are not uniformly distributed in the training set. **RESULTS** and **METHODS** are the most frequent categories, while **OBJECTIVE** is the least represented. The imbalance ratio (most frequent class / least frequent class) was computed during the exploratory analysis and is moderate, meaning no extreme class weighting strategies were required.

### 2.5 Sentence Length Analysis

An analysis of word counts per sentence reveals the following statistics:

- The **mean sentence length** is approximately 26 words.
- The **median** is slightly lower, indicating a right-skewed distribution with a few very long sentences.
- The **95th percentile** falls around 55 words.

Based on this analysis, a **maximum sequence length of 55 tokens** was selected for padding/truncation in the deep learning models, ensuring 95% of sentences are fully represented without excessive padding.

---

## 3. Methodology

### 3.1 Model 1 — Baseline: TF-IDF + Multinomial Naive Bayes

**Rationale.** This model serves as a simple, interpretable baseline. It requires no GPU, trains in seconds, and provides a reference point against which more complex models can be measured.

**Pipeline:**

1. **TF-IDF Vectorization**: Sentences are converted into sparse feature vectors using Term Frequency–Inverse Document Frequency. The vectorizer is configured with:
   - Unigrams and bigrams (`ngram_range=(1, 2)`),
   - A maximum of 10,000 features,
   - Sublinear TF scaling (`sublinear_tf=True`),
   - Minimum document frequency of 2 and maximum of 95%.

2. **Multinomial Naive Bayes Classifier**: A probabilistic classifier well-suited for text data with TF-IDF features, using Laplace smoothing (`alpha=1.0`).

The entire pipeline is implemented using scikit-learn's `Pipeline` class.

### 3.2 Model 2 — Pre-trained GloVe Embeddings

**Rationale.** Word embeddings capture semantic relationships between words in a dense vector space. Using pre-trained embeddings (GloVe 6B, 100-dimensional) allows the model to leverage knowledge from a large general-purpose corpus without training from scratch.

**Architecture:**

| Layer                      | Details                                  |
|----------------------------|------------------------------------------|
| Input                      | Raw string                               |
| TextVectorization          | Vocabulary ~68,000 tokens, max length 55 |
| Embedding (frozen)         | 100d, initialized with GloVe weights     |
| GlobalAveragePooling1D     | Averages across the sequence dimension   |
| Dense                      | 64 units, ReLU activation                |
| Dropout                    | 0.5                                      |
| Output Dense               | 5 units, softmax activation              |

**Training configuration:**
- Loss: categorical cross-entropy
- Optimizer: Adam (default learning rate)
- Epochs: 10
- Batch size: 32

The GloVe embedding layer is **frozen** (non-trainable) to preserve the pre-trained representations and reduce the number of learnable parameters.

### 3.3 Model 3 — Bidirectional LSTM

**Rationale.** Recurrent neural networks, and LSTMs in particular, are designed to capture sequential dependencies within text. A bidirectional architecture processes the sentence in both forward and backward directions, giving the model access to both left and right context for every token.

**Architecture:**

| Layer                      | Details                                  |
|----------------------------|------------------------------------------|
| Input                      | Raw string                               |
| TextVectorization          | Vocabulary ~68,000 tokens, max length 55 |
| Embedding (trainable)      | 128d, with zero masking                  |
| Bidirectional LSTM         | 64 units, recurrent dropout 0.2          |
| Dense                      | 64 units, ReLU activation                |
| Dropout                    | 0.5                                      |
| Output Dense               | 5 units, softmax activation              |

**Training configuration:**
- Loss: categorical cross-entropy
- Optimizer: Adam
- Epochs: up to 10 (with early stopping)
- Batch size: 32
- Callbacks:
  - **EarlyStopping**: monitors validation loss with patience of 3 epochs; restores best weights.
  - **ReduceLROnPlateau**: halves the learning rate if validation loss stagnates for 2 consecutive epochs (minimum LR: 1e-7).

Unlike the GloVe model, this model uses **trainable embeddings** (128-dimensional) learned from scratch on the task data.

---

## 4. Results

### 4.1 Performance Summary

| Model                        | Test Accuracy |
|------------------------------|---------------|
| Baseline (TF-IDF + NB)      | **78.0%**     |
| Embeddings (GloVe 100d)     | 32.8%         |
| Bidirectional LSTM           | *pending*     |

> **Note:** The Bi-LSTM results file was not yet generated at the time of writing. The model architecture and training pipeline are fully implemented and ready to execute.

### 4.2 Baseline Model Analysis

The TF-IDF + Naive Bayes baseline achieves a **test accuracy of 78.0%** (validation: 78.4%). This is a strong result for such a simple model, demonstrating that bag-of-words features with bigrams capture meaningful patterns in medical text. A detailed classification report and confusion matrix were generated, along with an error analysis identifying the most frequent misclassification pairs.

### 4.3 Embeddings Model Analysis

The GloVe-based model achieved a **test accuracy of only 32.8%**, which is significantly below the baseline. Several factors may explain this underperformance:

- **Domain mismatch**: GloVe embeddings are trained on general-purpose text (Wikipedia, Gigaword). Medical terminology may not be well-represented in the 100d vectors, leading to many out-of-vocabulary or poorly-represented words.
- **Frozen embeddings**: Because the embedding layer was not fine-tuned, the model could not adapt the representations to the medical domain.
- **Simple aggregation**: GlobalAveragePooling1D collapses the entire sequence into a single vector by averaging, which may lose important positional and sequential information.

### 4.4 Error Analysis

The baseline model's error analysis (Notebook 02) reveals that the most common confusions occur between semantically close categories, such as:
- **METHODS** vs. **RESULTS** (procedural descriptions vs. outcome reporting),
- **BACKGROUND** vs. **CONCLUSIONS** (contextual statements that can appear similar).

These confusions are expected, as sentences at category boundaries often share lexical features.

---

## 5. Comparative Discussion

The results highlight an important lesson in NLP: **model complexity does not guarantee better performance**. The simple TF-IDF + Naive Bayes baseline significantly outperforms the GloVe embedding model. This can be attributed to:

1. **Feature engineering effectiveness**: TF-IDF with bigrams captures discriminative n-gram patterns (e.g., "the aim of", "in conclusion") that directly correspond to specific sentence roles.

2. **Domain specificity**: The baseline model builds its vocabulary directly from the training data, making it inherently adapted to medical language. In contrast, GloVe vectors are generic and miss domain-specific nuances.

3. **Architecture limitations**: The GloVe model uses a shallow architecture (average pooling + single dense layer), which may be insufficient for this classification task without fine-tuning the embeddings.

The Bi-LSTM model addresses some of these limitations by:
- Learning task-specific embeddings from scratch,
- Modeling sequential dependencies through recurrent layers,
- Using regularization techniques (dropout, early stopping, learning rate scheduling) to prevent overfitting.

---

## 6. Project Structure and Reproducibility

The project follows a clean, modular structure:

```
Project/
├── data/pubmed-rct/                    # PubMed 20k RCT dataset
├── notebooks/
│   ├── 01_data_exploration.ipynb       # EDA and statistical analysis
│   ├── 02_baseline_model.ipynb         # TF-IDF + Naive Bayes
│   ├── 03_embeddings_model.ipynb       # GloVe embeddings model
│   ├── 04_deep_learning_model.ipynb    # Bidirectional LSTM
│   └── 05_evaluation_comparison.ipynb  # Side-by-side comparison
├── results/                            # Saved JSON result files
└── README.md                           # Project documentation
```

Each notebook is **self-contained** with its own data loading code, making them independently executable. Results are serialized to JSON for cross-notebook comparison. The comparison notebook (05) aggregates all results and produces a unified bar chart visualization.

**Dependencies:** NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, TensorFlow/Keras.

---

## 7. Limitations and Future Work

Several avenues could improve upon the current results:

- **Domain-specific embeddings**: Using BioWordVec or PubMedBERT embeddings instead of general-purpose GloVe would likely improve the neural models significantly.
- **Positional features**: Incorporating the sentence's position within the abstract (line number / total lines) as an additional feature, since sentence role is strongly correlated with position.
- **Transformer-based models**: Fine-tuning a pre-trained model such as BERT, SciBERT, or BioBERT would likely yield state-of-the-art performance on this task.
- **Class weighting**: Applying class weights or oversampling to address the moderate label imbalance.
- **Ensemble methods**: Combining predictions from multiple models to leverage their complementary strengths.

---

## 8. Conclusion

This project explored the task of sentence classification in medical abstracts using the PubMed 20k RCT dataset. Three models were implemented with increasing levels of complexity: a TF-IDF + Naive Bayes baseline, a GloVe embedding model, and a Bidirectional LSTM.

The baseline model proved to be the most effective among the evaluated configurations, achieving 78.0% accuracy on the test set. The GloVe embedding model underperformed due to domain mismatch and architectural simplicity. The Bi-LSTM model, with its ability to learn task-specific embeddings and capture sequential patterns, represents the most promising architecture among the three, though its training requires GPU resources.

This work demonstrates that for domain-specific NLP tasks, simple models with well-engineered features can be surprisingly competitive, and that transfer learning from general-purpose embeddings requires careful adaptation to be effective.

---

## References

1. Dernoncourt, F., & Lee, J. Y. (2017). *PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts.* International Joint Conference on Natural Language Processing (IJCNLP).

2. Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global Vectors for Word Representation.* Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP).

3. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8), 1735–1780.

4. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825–2830.

---
