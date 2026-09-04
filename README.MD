# Spam Message Detector 🚫📩

A simple machine learning project that classifies SMS text messages as **spam** or **ham** (not spam), built with `scikit-learn`. It uses TF-IDF for text vectorization and a Logistic Regression classifier, achieving **~98% accuracy** on the test set.

## 📁 Project Structure

```
spam_message_detector/
├── data/
│   └── spam.csv              # SMS Spam Collection dataset
├── notebook/
│   └── spam_detector.ipynb   # Data cleaning, training, evaluation, and a CLI demo
└── README.MD
```

## 📊 Dataset

The model is trained on the [SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), a set of 5,572 SMS messages labeled as either `ham` (legitimate) or `spam`.

| Label | Count |
|-------|-------|
| Ham   | 4,825 |
| Spam  | 747   |

## ⚙️ How It Works

1. **Load data** — reads `data/spam.csv` and keeps only the label and message columns.
2. **Clean text** — lowercases all text and strips out characters that aren't letters, numbers, spaces, or periods.
3. **Split data** — 80/20 train/test split, stratified by label so both sets keep the same spam/ham ratio.
4. **Vectorize** — converts messages into numerical features using `TfidfVectorizer`.
5. **Train** — fits a `LogisticRegression` model (with `class_weight='balanced'` to account for the class imbalance between ham and spam).
6. **Evaluate** — reports accuracy on the held-out test set.
7. **Try it out** — a small command-line prompt lets you type in your own message and see whether the model flags it as spam.

## 📈 Results

| Metric   | Score  |
|----------|--------|
| Accuracy | ~98.0% |

*(Labels are encoded internally as `ham = 1`, `spam = 0`.)*

## 🧰 Requirements

- Python 3.8+
- `pandas`
- `numpy`
- `scikit-learn`
- `jupyter` (to run the notebook)

Install them with:

```bash
pip install pandas numpy scikit-learn jupyter
```

## 🚀 Usage

1. Clone the repo:
   ```bash
   git clone https://github.com/awaisalidev13/spam_message_detector.git
   cd spam_message_detector
   ```
2. Launch the notebook:
   ```bash
   jupyter notebook notebook/spam_detector.ipynb
   ```
3. Run all cells in order. The final cell will prompt you to enter a message and tell you whether it's spam:
   ```
   ======WELCOME TO SPAM DETECTOR=====================================
   Enter the message to check : Congratulations! You've won a free prize, claim now!
   This is a spam message.
   ```

## 🔮 Possible Improvements

- Save the trained model and vectorizer (e.g. with `pickle` or `joblib`) so predictions don't require retraining every session.
- Wrap the model in a simple web app or API (Flask/FastAPI/Streamlit) instead of a CLI prompt.
- Experiment with other classifiers (Naive Bayes, SVM) or richer text preprocessing (stemming, stopword removal, n-grams).
- Add a `requirements.txt` and automated tests.

## 📄 License

No license has been specified for this repository yet.