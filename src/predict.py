"""Inference entry point.

The current repository does not ship serialized preprocessing/model artifacts.
Train the pipeline first and persist the complete inference bundle before using
this module for prediction.
"""

def main():
    print(
        "Prediction is not enabled yet because trained preprocessing/scaler/"
        "model artifacts are intentionally not included in the repository. "
        "Run the training pipeline and persist the artifacts before adding "
        "inference commands."
    )

if __name__ == "__main__":
    main()
