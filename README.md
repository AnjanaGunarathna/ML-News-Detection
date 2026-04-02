# ML-News-Detection

News authenticity detection project using a Naive Bayes text classification pipeline.

This repository contains a notebook-based workflow to classify news articles as:

- `0` -> Fake news
- `1` -> Real news

The model is trained with TF-IDF features and `MultinomialNB` from scikit-learn.

## Project Contents

- `Naive_Bayes.ipynb` - End-to-end notebook for data loading, preprocessing, training, and evaluation.
- `README.md` - Project documentation and setup guide.

## Workflow Summary

The notebook follows these steps:

1. Import required libraries (`pandas`, `numpy`, `scikit-learn`, plotting libraries).
2. Load datasets from `Fake.csv` and `True.csv`.
3. Add binary labels:
	- Fake -> `0`
	- Real -> `1`
4. Merge and shuffle both datasets.
5. Build a combined text field using `title + text`.
6. Clean text (lowercasing, URL/punctuation removal, regex-based cleanup).
7. Split into train/test sets (`test_size=0.2`, `random_state=42`).
8. Convert text to TF-IDF vectors (`stop_words="english"`, `max_df=0.7`).
9. Train a `MultinomialNB` model.
10. Evaluate with:
	- Accuracy score
	- Classification report
	- Confusion matrix
	- Confusion matrix visualizations (Matplotlib and Seaborn heatmap)

## Requirements

Install the following Python packages:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter

Quick install:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

## Dataset

Place the dataset files in the project root:

- `Fake.csv`
- `True.csv`

The notebook expects these exact filenames when running locally.

## How To Run

### Option 1: Run in Google Colab

1. Open `Naive_Bayes.ipynb` in Colab.
2. Run the upload cell:

```python
from google.colab import files
uploaded = files.upload()
```

3. Upload `Fake.csv` and `True.csv` when prompted.
4. Run all remaining cells in order.

### Option 2: Run locally (VS Code / Jupyter)

1. Install dependencies.
2. Ensure `Fake.csv` and `True.csv` are in the root folder.
3. Open `Naive_Bayes.ipynb`.
4. Skip or remove the Colab upload cell.
5. Run cells top to bottom.

## Output Metrics

The notebook prints:

- Overall test accuracy
- Full precision/recall/F1 classification report
- Confusion matrix values
- Training vs testing accuracy comparison

It also displays two confusion matrix plots:

- Basic Matplotlib matrix view
- Seaborn annotated heatmap

## Notes

- The split is reproducible using `random_state=42`.
- `stop_words="english"` and `max_df=0.7` help reduce noisy features.
- For production-style usage, consider saving the trained vectorizer and model with `joblib`.