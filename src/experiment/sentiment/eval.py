"""Downloads STT, evaluates sentiment classifier.

Run as python -m experiment.sentiment.eval
"""

import json
import sys

from datasets import load_dataset
from tqdm import tqdm
from transformers import pipeline


def main():
    """Downloads STT, evaluates sentiment classifier."""

    # Initialize the sentiment-analysis pipeline once
    sentiment_analyzer = pipeline(
        "sentiment-analysis", model="textattack/roberta-base-SST-2"
    )

    def process_example(example):
        """Process a single example."""
        text = example["sentence"]
        ground_truth = example["label"]
        sentiment = sentiment_analyzer(text)[0]
        return {
            "text": text,
            "ground_truth": ground_truth,
            "sentiment": sentiment,
        }

    # Download the Stanford Sentiment Treebank (SST) dataset
    dataset = load_dataset("stanfordnlp/sst", split="train")

    # Process the dataset sequentially
    results = [process_example(example) for example in tqdm(dataset)]

    # Save results to a JSON file
    with open("sentiment_analysis_results.json", "w") as f:
        json.dump(results, f, indent=4)

    return 0


if __name__ == "__main__":
    sys.exit(main())
