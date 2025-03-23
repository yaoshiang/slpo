import json
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    # Load the results from the JSON file
    with open("sentiment_analysis_results.json", "r") as f:
        results = json.load(f)

    # Convert the results to a DataFrame
    df = pd.DataFrame(results)

    # Extract the sentiment scores and ground truth labels
    df["sentiment_score"] = df["sentiment"].apply(lambda x: x["score"])
    df["sentiment_label"] = df["sentiment"].apply(lambda x: x["label"])

    # Map ground truth labels to sentiment labels
    df["ground_truth_label"] = df["ground_truth"].apply(
        lambda x: "LABEL_1" if x > 0.5 else "LABEL_0"
    )

    # Plot calibration curve
    plt.figure(figsize=(10, 6))
    sns.histplot(
        df,
        x="sentiment_label",
        hue="ground_truth_label",
        multiple="stack",
        bins=20,
    )
    plt.title("Calibration Curve")
    plt.xlabel("Sentiment Label")
    plt.ylabel("Count")
    plt.legend(title="Ground Truth")
    plt.savefig("calibration_curve.png")

    # Statistical analysis
    accuracy = (df["sentiment_label"] == df["ground_truth_label"]).mean()
    print(f"Accuracy: {accuracy:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
