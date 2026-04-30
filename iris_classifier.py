#!/usr/bin/env python3
"""
Iris Flower Classifier CLI
Predicts iris species from physical measurements using a trained Decision Tree model.

Usage:
    python iris_classifier.py train                    # Train and save the model
    python iris_classifier.py predict                  # Interactive prediction mode
    python iris_classifier.py predict 5.1 3.5 1.4 0.2  # Single prediction
"""

import os
import pickle
import argparse
from pathlib import Path
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


MODEL_PATH = "iris_model.pkl"
ENCODER_PATH = "iris_encoder.pkl"
FEATURE_NAMES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
SPECIES = ["setosa", "versicolor", "virginica"]


def train_model(data_path="iris_dataset.csv"):
    """Train and save the iris classifier model."""
    print("🌸 Training Iris Classifier...")
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"   Loaded {len(df)} samples from {data_path}")
    
    # Prepare features and target
    X = df[FEATURE_NAMES]
    y = df["Species"]
    
    # Encode species labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Train Decision Tree
    model = DecisionTreeClassifier(max_depth=5, random_state=42, min_samples_split=5)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    print(f"\n   Training Accuracy: {train_acc:.2%}")
    print(f"   Test Accuracy: {test_acc:.2%}")
    
    # Classification report
    y_pred = model.predict(X_test)
    print("\n   Classification Report:")
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    for line in report.split('\n'):
        print(f"   {line}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n   Confusion Matrix:")
    print(f"   {'':>10} {'setosa':>12} {'versicolor':>12} {'virginica':>12}")
    for i, species in enumerate(le.classes_):
        print(f"   {species:>10} {cm[i,0]:>12} {cm[i,1]:>12} {cm[i,2]:>12}")
    
    # Save model and encoder
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    
    print(f"\n✅ Model saved to {MODEL_PATH}")
    print(f"✅ Encoder saved to {ENCODER_PATH}")


def load_model():
    """Load the trained model and encoder."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        print("❌ Model not found. Run 'python iris_classifier.py train' first.")
        exit(1)
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    
    return model, encoder


def predict_single(sepal_length, sepal_width, petal_length, petal_width):
    """Predict species for a single sample."""
    model, encoder = load_model()
    
    # Validate inputs
    try:
        features = [float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]
    except ValueError:
        print("❌ Error: All measurements must be valid numbers.")
        return
    
    # Check reasonable ranges
    if any(x <= 0 for x in features):
        print("⚠️  Warning: Flower measurements should be positive values.")
    
    # Create input dataframe
    X = pd.DataFrame([features], columns=FEATURE_NAMES)
    
    # Predict
    pred_encoded = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0]
    pred_species = encoder.classes_[pred_encoded]
    
    # Format output
    print("\n" + "="*60)
    print("🌸 IRIS FLOWER PREDICTION")
    print("="*60)
    print(f"\nInput Measurements:")
    print(f"  • Sepal Length: {sepal_length} cm")
    print(f"  • Sepal Width:  {sepal_width} cm")
    print(f"  • Petal Length: {petal_length} cm")
    print(f"  • Petal Width:  {petal_width} cm")
    
    print(f"\n🎯 Predicted Species: {pred_species.upper()}")
    
    print(f"\nConfidence Scores:")
    for species, prob in zip(encoder.classes_, pred_proba):
        confidence_bar = "█" * int(prob * 30) + "░" * (30 - int(prob * 30))
        print(f"  • {species:12} {confidence_bar} {prob:.1%}")
    
    print("\n" + "="*60 + "\n")


def interactive_predict():
    """Launch interactive prediction mode."""
    print("\n" + "="*60)
    print("🌸 IRIS FLOWER CLASSIFIER - INTERACTIVE MODE")
    print("="*60)
    print("\nEnter flower measurements (or 'quit' to exit):\n")
    
    while True:
        try:
            sepal_length = input("Sepal Length (cm): ").strip()
            if sepal_length.lower() == 'quit':
                print("\n👋 Goodbye!\n")
                break
            
            sepal_width = input("Sepal Width (cm):  ").strip()
            petal_length = input("Petal Length (cm): ").strip()
            petal_width = input("Petal Width (cm):  ").strip()
            
            predict_single(sepal_length, sepal_width, petal_length, petal_width)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Iris Flower Species Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python iris_classifier.py train
  python iris_classifier.py predict
  python iris_classifier.py predict 5.1 3.5 1.4 0.2
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--data",
        default="iris_dataset.csv",
        help="Path to iris dataset CSV (default: iris_dataset.csv)"
    )
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument(
        "measurements",
        nargs="*",
        help="Flower measurements: sepal_length sepal_width petal_length petal_width"
    )
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_model(args.data)
    elif args.command == "predict":
        if args.measurements:
            if len(args.measurements) != 4:
                print("❌ Error: Expected 4 measurements (sepal_length sepal_width petal_length petal_width)")
                exit(1)
            predict_single(*args.measurements)
        else:
            interactive_predict()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
