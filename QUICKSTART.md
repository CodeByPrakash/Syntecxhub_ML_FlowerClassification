# 🚀 Quick Start Guide - Iris Classifier CLI

## Setup (1 minute)

```bash
# 1. Ensure you have pandas and scikit-learn
pip install pandas scikit-learn

# 2. Place files in same directory:
# - iris_classifier.py
# - iris_dataset.csv
```

## First Run (30 seconds)

### Train the model
```bash
python iris_classifier.py train
```

Expected output:
```
🌸 Training Iris Classifier...
   Loaded 150 samples from iris_dataset.csv
   Training Accuracy: 98.33%
   Test Accuracy: 96.67%
✅ Model saved to iris_model.pkl
✅ Encoder saved to iris_encoder.pkl
```

---

## Make Predictions (3 ways)

### Way 1: One-liner predictions
```bash
# Setosa (small flower)
python iris_classifier.py predict 5.1 3.5 1.4 0.2

# Versicolor (medium flower)
python iris_classifier.py predict 7.0 3.2 4.7 1.4

# Virginica (large flower)
python iris_classifier.py predict 6.3 3.3 6.0 2.5
```

### Way 2: Interactive mode
```bash
python iris_classifier.py predict
```

Then enter measurements when prompted:
```
Sepal Length (cm): 5.1
Sepal Width (cm):  3.5
Petal Length (cm): 1.4
Petal Width (cm):  0.2
```

Type `quit` to exit.

### Way 3: Batch predictions (from file)
Create `predictions.txt`:
```
5.1 3.5 1.4 0.2
7.0 3.2 4.7 1.4
6.3 3.3 6.0 2.5
```

Then run:
```bash
while IFS=' ' read sl sw pl pw; do
  python iris_classifier.py predict $sl $sw $pl $pw
done < predictions.txt
```

---

## Understanding the Output

```
🎯 Predicted Species: SETOSA

Confidence Scores:
  • setosa       ██████████████████████████████ 100.0%
  • versicolor   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.0%
  • virginica    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.0%
```

- **Predicted Species**: The model's best guess
- **Confidence Scores**: How sure the model is about each species
  - Longer bars = higher confidence
  - Look at the percentage values for exact confidence

---

## Typical Measurements

### Setosa (Small)
```bash
python iris_classifier.py predict 5.1 3.5 1.4 0.2
python iris_classifier.py predict 4.9 3.0 1.4 0.2
python iris_classifier.py predict 4.7 3.2 1.3 0.2
```

### Versicolor (Medium)
```bash
python iris_classifier.py predict 7.0 3.2 4.7 1.4
python iris_classifier.py predict 6.4 3.2 4.5 1.5
python iris_classifier.py predict 6.9 3.1 4.9 1.5
```

### Virginica (Large)
```bash
python iris_classifier.py predict 6.3 3.3 6.0 2.5
python iris_classifier.py predict 7.1 3.0 5.9 2.1
python iris_classifier.py predict 6.3 2.9 5.6 1.8
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Module not found: pandas` | Run `pip install pandas` |
| `Module not found: sklearn` | Run `pip install scikit-learn` |
| `Model not found` | Run `python iris_classifier.py train` first |
| `Expected 4 measurements` | Provide exactly 4 numbers |
| Numbers have wrong range? | Check the "Typical Measurements" section above |

---

## Advanced: Use in Python Code

```python
import pickle
import pandas as pd

# Load trained model
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("iris_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Make predictions
measurements = [[5.1, 3.5, 1.4, 0.2],
                [7.0, 3.2, 4.7, 1.4],
                [6.3, 3.3, 6.0, 2.5]]

X = pd.DataFrame(measurements, 
                 columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])

predictions = model.predict(X)
species = [encoder.classes_[p] for p in predictions]

for spec in species:
    print(f"Predicted: {spec}")
```

---

## That's it! 🎉

You're ready to classify iris flowers. See `README.md` for more detailed documentation.
