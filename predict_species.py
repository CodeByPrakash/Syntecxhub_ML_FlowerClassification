import argparse
from pathlib import Path

import pandas as pd
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


FEATURE_COLUMNS = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
TARGET_COLUMN = "Species"


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def build_model(model_name: str):
    if model_name == "tree":
        return DecisionTreeClassifier(random_state=42)
    return SVC()


def get_input_values(args: argparse.Namespace) -> list[float]:
    if args.interactive:
        print("Enter flower measurements:")
        sl = float(input("SepalLengthCm: "))
        sw = float(input("SepalWidthCm: "))
        pl = float(input("PetalLengthCm: "))
        pw = float(input("PetalWidthCm: "))
        return [sl, sw, pl, pw]

    if None in (args.sepal_length, args.sepal_width, args.petal_length, args.petal_width):
        raise ValueError(
            "Provide all four values using --sepal-length --sepal-width --petal-length --petal-width, "
            "or use --interactive."
        )

    return [args.sepal_length, args.sepal_width, args.petal_length, args.petal_width]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Iris species from flower measurements.")
    parser.add_argument("--data", default="iris_dataset.csv", help="Path to iris CSV file.")
    parser.add_argument("--model", choices=["svm", "tree"], default="svm", help="Model type.")
    parser.add_argument("--sepal-length", type=float, dest="sepal_length", help="Sepal length in cm.")
    parser.add_argument("--sepal-width", type=float, dest="sepal_width", help="Sepal width in cm.")
    parser.add_argument("--petal-length", type=float, dest="petal_length", help="Petal length in cm.")
    parser.add_argument("--petal-width", type=float, dest="petal_width", help="Petal width in cm.")
    parser.add_argument("--interactive", action="store_true", help="Prompt for input values interactively.")
    args = parser.parse_args()

    csv_path = Path(args.data)
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find dataset: {csv_path}")

    df = load_data(csv_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    model = build_model(args.model)
    model.fit(X, y)

    values = get_input_values(args)
    sample = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    prediction = model.predict(sample)[0]

    print(f"Predicted species: {prediction}")


if __name__ == "__main__":
    main()
