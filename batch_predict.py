#!/usr/bin/env python3
"""
Batch Iris Predictor
Predicts species for multiple flowers from a CSV file and saves results.

Usage:
    python batch_predict.py input.csv output.csv
    python batch_predict.py --input data.csv --output results.csv --show
"""

import csv
import pickle
import argparse
from pathlib import Path
import pandas as pd


MODEL_PATH = "iris_model.pkl"
ENCODER_PATH = "iris_encoder.pkl"
FEATURE_NAMES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]


def load_model():
    """Load the trained model and encoder."""
    if not Path(MODEL_PATH).exists() or not Path(ENCODER_PATH).exists():
        print("❌ Model files not found. Run 'python iris_classifier.py train' first.")
        exit(1)
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    
    return model, encoder


def batch_predict(input_csv, output_csv, show=False):
    """Predict species for multiple samples from CSV."""
    print(f"📂 Loading data from {input_csv}...")
    
    # Check if input file exists
    if not Path(input_csv).exists():
        print(f"❌ File not found: {input_csv}")
        exit(1)
    
    # Load input data
    try:
        df_input = pd.read_csv(input_csv)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        exit(1)
    
    # Validate columns
    if not all(col in df_input.columns for col in FEATURE_NAMES):
        print(f"❌ CSV must contain columns: {', '.join(FEATURE_NAMES)}")
        print(f"   Found columns: {', '.join(df_input.columns)}")
        exit(1)
    
    print(f"   ✓ Loaded {len(df_input)} samples")
    
    # Load model
    print("🤖 Loading model...")
    model, encoder = load_model()
    print("   ✓ Model loaded")
    
    # Prepare features
    X = df_input[FEATURE_NAMES]
    
    # Make predictions
    print("🌸 Making predictions...")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    species_names = [encoder.classes_[p] for p in predictions]
    
    # Create output dataframe
    df_output = df_input.copy()
    df_output["Predicted_Species"] = species_names
    
    # Add confidence scores
    for i, species in enumerate(encoder.classes_):
        df_output[f"{species}_confidence"] = probabilities[:, i]
    
    # Save output
    try:
        df_output.to_csv(output_csv, index=False)
        print(f"   ✓ Predictions saved to {output_csv}")
    except Exception as e:
        print(f"❌ Error writing CSV: {e}")
        exit(1)
    
    # Display summary
    print("\n" + "="*60)
    print("📊 PREDICTION SUMMARY")
    print("="*60)
    
    species_counts = df_output["Predicted_Species"].value_counts()
    for species, count in species_counts.items():
        pct = 100 * count / len(df_output)
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"  {species:12} {bar} {count:3d} ({pct:5.1f}%)")
    
    print("\n" + "="*60)
    
    # Show sample predictions if requested
    if show:
        print("\n📋 Sample Predictions (first 10 rows):")
        print("="*60)
        
        display_cols = FEATURE_NAMES + ["Predicted_Species", "setosa_confidence", 
                                       "versicolor_confidence", "virginica_confidence"]
        
        for idx, row in df_output.head(10).iterrows():
            print(f"\nSample {idx + 1}:")
            print(f"  Sepal L: {row['SepalLengthCm']:.1f} | Sepal W: {row['SepalWidthCm']:.1f} | "
                  f"Petal L: {row['PetalLengthCm']:.1f} | Petal W: {row['PetalWidthCm']:.1f}")
            print(f"  Predicted: {row['Predicted_Species']:12} "
                  f"(Confidence: {max(row['setosa_confidence'], row['versicolor_confidence'], row['virginica_confidence']):.1%})")


def create_sample_input(filename="sample_input.csv"):
    """Create a sample input CSV for testing."""
    samples = [
        ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
        [5.1, 3.5, 1.4, 0.2],
        [7.0, 3.2, 4.7, 1.4],
        [6.3, 3.3, 6.0, 2.5],
        [5.4, 3.9, 1.7, 0.4],
        [6.4, 3.2, 4.5, 1.5],
    ]
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(samples)
    
    print(f"✅ Sample input file created: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch predict iris species from CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_predict.py input.csv output.csv
  python batch_predict.py --input data.csv --output results.csv --show
  python batch_predict.py --create-sample  # Create a sample input file
        """
    )
    
    parser.add_argument(
        "input",
        nargs="?",
        help="Input CSV file with flower measurements"
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="Output CSV file for predictions"
    )
    parser.add_argument(
        "-i", "--input",
        dest="input_file",
        help="Input CSV file (alternative to positional argument)"
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_file",
        help="Output CSV file (alternative to positional argument)"
    )
    parser.add_argument(
        "-s", "--show",
        action="store_true",
        help="Show sample predictions"
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create a sample input CSV file"
    )
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_input()
        return
    
    # Determine input/output files
    input_file = args.input_file or args.input
    output_file = args.output_file or args.output
    
    if not input_file or not output_file:
        parser.print_help()
        exit(1)
    
    batch_predict(input_file, output_file, show=args.show)


if __name__ == "__main__":
    main()
