"""CLI entry point for Episode 3: First ML model."""

import argparse

from src import predict_text, train_classifier


def main():
    parser = argparse.ArgumentParser(
        prog="ai-workflow-ep3",
        description="Decode AI Using AI — Episode 3: First ML Model",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    train_parser = subparsers.add_parser("train", help="Train a text classifier")
    train_parser.add_argument("input", help="Path to the labeled CSV file")
    train_parser.add_argument("output", help="Directory to save model artifacts")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    evaluate_parser.add_argument("input", help="Path to the labeled CSV file")
    evaluate_parser.add_argument("model", help="Directory containing the saved model artifacts")

    predict_parser = subparsers.add_parser("predict", help="Run predictions with a trained model")
    predict_parser.add_argument("model", help="Directory containing the saved model artifacts")
    predict_parser.add_argument("text", help="Text input to classify")

    args = parser.parse_args()

    if args.command == "train":
        result = train_classifier(args.input, args.output)
        print(f"Training model using: {args.input}")
        print(f"Saving artifacts to:  {args.output}")
        print(f"Status:               {result['status']}")
        print(f"Train rows:           {result['train_rows']}")
        print(f"Test rows:            {result['test_rows']}")
        print(f"Train shape:          {result['train_shape']}")
        print(f"Test shape:           {result['test_shape']}")
        print(f"Logistic Regression:  {result['model_scores']['logistic_regression']['accuracy']}")
        print(f"Multinomial NB:       {result['model_scores']['multinomial_nb']['accuracy']}")
        print(f"Best model:           {result['best_model']}")
        print(f"Model file:           {result['model_file']}")
        print(f"Vectorizer file:      {result['vectorizer_file']}")
        print(f"Metrics file:         {result['metrics_file']}")
    elif args.command == "evaluate":
        print(f"Evaluating model:     {args.model}")
        print(f"Using dataset:        {args.input}")
    elif args.command == "predict":
        result = predict_text(args.model, args.text)
        print(f"Model: {args.model}")
        print(f"Text:  {args.text}")
        print(f"Prediction: {result['prediction']}")
    else:
        parser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()