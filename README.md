# Iris Flower Classifier CLI

A command-line tool for predicting iris flower species based on physical measurements using a trained Decision Tree classifier.

## Features

**Three prediction modes:**
- Train a fresh model from your dataset
- Single-shot predictions from command line
- Interactive mode for multiple predictions

**Model insights:**
- Accuracy metrics on test data
- Classification report (precision, recall, F1-score)
- Confusion matrix visualization
- Probability scores for each prediction

**Robust design:**
- Input validation
- Error handling
- Model persistence (pickle format)
- Reusable trained model

---

## Installation

### Requirements
```bash
pip install pandas scikit-learn
```

Python 3.7+ recommended.

---

## Usage

### Train the Model

First, train the classifier on your iris dataset:

```bash
python iris_classifier.py train
```

This will:
- Load `iris_dataset.csv` from the current directory
- Train a Decision Tree classifier
- Display accuracy, classification report, and confusion matrix
- Save model files (`iris_model.pkl`, `iris_encoder.pkl`)

**Optional: Use a different dataset**
```bash
python iris_classifier.py train --data /path/to/custom_iris.csv
```

### Make Predictions

#### Option A: Interactive Mode
```bash
python iris_classifier.py predict
```

You'll be prompted to enter measurements one by one:
```
Enter flower measurements (or 'quit' to exit):

Sepal Length (cm): 5.1
Sepal Width (cm):  3.5
Petal Length (cm): 1.4
Petal Width (cm):  0.2
```

#### Option B: Command Line Arguments
```bash
python iris_classifier.py predict 5.1 3.5 1.4 0.2
```

#### Option C: Batch Predictions (via shell)
```bash
python iris_classifier.py predict 5.1 3.5 1.4 0.2
python iris_classifier.py predict 7.0 3.2 4.7 1.4
python iris_classifier.py predict 6.3 3.3 6.0 2.5
```

---

## Example Output

### Training Output
```
Training Iris Classifier...
   Loaded 150 samples from iris_dataset.csv

   Training Accuracy: 96.67%
   Test Accuracy: 96.67%

   Classification Report:
                precision    recall  f1-score   support
          setosa       1.00      1.00      1.00        10
      versicolor       0.95      1.00      0.97        20
       virginica       1.00      0.90      0.95        20
        accuracy                          0.97        50
       macro avg       0.98      0.97      0.97        50
    weighted avg       0.97      0.97      0.97        50

   Confusion Matrix:
              setosa   versicolor    virginica
   setosa          10            0            0
   versicolor       0           20            0
   virginica        0            2           18

Model saved to iris_model.pkl
Encoder saved to iris_encoder.pkl
```

### Prediction Output
```
============================================================
IRIS FLOWER PREDICTION
============================================================

Input Measurements:
  • Sepal Length: 5.1 cm
  • Sepal Width:  3.5 cm
  • Petal Length: 1.4 cm
  • Petal Width:  0.2 cm

Predicted Species: SETOSA

Confidence Scores:
  • setosa        ██████████████████████████████ 100.0%
  • versicolor    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.0%
  • virginica     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.0%

============================================================
```

---

## Iris Species Info

The classifier predicts three iris species:

| Species | Characteristics |
|---------|-----------------|
| **Setosa** | Smallest flowers, small petals |
| **Versicolor** | Medium-sized, longer petals than setosa |
| **Virginica** | Largest flowers, widest petals |

### Typical Measurement Ranges (cm)

| Measurement | Setosa | Versicolor | Virginica |
|------------|--------|-----------|-----------|
| Sepal Length | 4.3-5.8 | 5.7-7.0 | 6.3-7.9 |
| Sepal Width | 2.3-4.4 | 2.2-3.4 | 2.2-3.8 |
| Petal Length | 1.0-1.9 | 3.0-5.1 | 4.5-6.9 |
| Petal Width | 0.1-0.6 | 1.0-1.8 | 1.4-2.5 |

---

## How It Works

1. **Data Loading**: Reads iris measurements and species labels from CSV
2. **Encoding**: Converts species names (setosa, versicolor, virginica) to numeric labels
3. **Train-Test Split**: 80% training, 20% testing (stratified)
4. **Decision Tree Training**: Trains with max_depth=5, min_samples_split=5
5. **Model Persistence**: Saves model and label encoder as pickle files
6. **Prediction**: Takes new measurements and outputs species + confidence scores

---

## File Structure

```
iris_classifier.py          # Main CLI script
iris_dataset.csv            # Training data
iris_model.pkl              # Trained model (created after training)
iris_encoder.pkl            # Label encoder (created after training)
```

---

## Troubleshooting

###  "Model not found" error
- Run `python iris_classifier.py train` first
- Ensure `iris_dataset.csv` is in the same directory

###  "All measurements must be valid numbers"
- Ensure you enter numeric values (e.g., `5.1`, not `5.1cm`)
- Decimals are OK (e.g., `5`, `5.1`, `5.123`)

###  "Warning: Flower measurements should be positive values"
- Flower measurements must be > 0
- Check your input data

---

## Advanced Usage

### Custom Dataset Format
Your CSV must have columns:
- `SepalLengthCm`
- `SepalWidthCm`
- `PetalLengthCm`
- `PetalWidthCm`
- `Species` (values: setosa, versicolor, virginica)

### Integrating into Other Code
```python
import pickle
import pandas as pd

# Load model
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("iris_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Make prediction
X = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], 
                 columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])
prediction = model.predict(X)[0]
species = encoder.classes_[prediction]
print(f"Predicted: {species}")
```

---

## Model Details

- **Algorithm**: Decision Tree Classifier
- **Max Depth**: 5 (prevents overfitting)
- **Min Samples Split**: 5
- **Test Accuracy**: ~96-97% (depending on random seed)
- **Training Data**: 150 samples, 4 features, 3 classes

---

## License

Free to use and modify for educational purposes.
