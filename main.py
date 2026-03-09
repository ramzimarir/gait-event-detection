"""Main entry point for gait event detection."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from config import FS, INPUT_DATA_DIR, MODEL_IN_CHANNELS, MODEL_WEIGHTS_DIR, OUTPUT_DIR, PROJECT_ROOT
from src.data_loader import DataLoader, get_input_files, setup_logger, validate_directory_structure
from src.evaluator import GaitEventEvaluator
from src.models.cnn import CNNModel
from src.models.lstm import LSTMModel
from src.models.tcn import TCNModel
from src.utils import probs_to_events

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gait event detection pipeline")
    parser.add_argument("--model", default="cnn", choices=["cnn", "lstm", "tcn"], help="Model to run")
    parser.add_argument("--input-dir", default=str(INPUT_DATA_DIR), help="Path to input data directory")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Path to output directory")
    parser.add_argument("--fs", type=int, default=FS, help="Sampling rate (Hz)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_dir)
    output_dir = output_root / args.model

    logger = setup_logger("main", log_file=str(PROJECT_ROOT / "pipeline.log"))
    logger.info("=" * 60)
    logger.info("GAIT PIPELINE START")
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Frequency: {args.fs} Hz")
    logger.info(f"  Output: {output_dir}")

    try:
        validate_directory_structure(input_dir, output_dir)
    except FileNotFoundError as exc:
        logger.error(f"Directory structure error: {exc}")
        return

    loader = DataLoader()
    evaluator = GaitEventEvaluator(fs=args.fs)

    if args.model in {"cnn", "lstm", "tcn"}:
        model = None  # Will be loaded per-subject dynamically
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    input_files = get_input_files(input_dir)
    if not input_files:
        logger.error(f"No CSV files found in {input_dir}")
        return

    logger.info(f"\n{len(input_files)} file(s) found:")
    for input_file in input_files:
        logger.info(f"  [OK] Input: {input_file.name}")

    evaluation_results = []
    success_count = 0

    for idx, input_file in enumerate(input_files, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{idx}/{len(input_files)}] Processing: {input_file.name}")
        logger.info(f"{'='*60}")

        try:
            df_all = loader.load_data(input_file)
            df_output = df_all.copy()
            df_labels = loader.get_labels(df_all)

            # Load model dynamically based on filename
            if args.model in {"cnn", "lstm", "tcn"}:
                subject_id = input_file.stem  # e.g., "Subject_A" from "Subject_A.csv"
                model_path = MODEL_WEIGHTS_DIR / f"best_{args.model}_subject_{subject_id}.pth"
                if not model_path.exists():
                    logger.error(f"Model not found for {subject_id}: {model_path}")
                    continue
                logger.info(f"Loading {args.model.upper()} weights: {model_path.name}")
                if args.model == "cnn":
                    model = CNNModel(in_channels=MODEL_IN_CHANNELS, weights_path=model_path)
                elif args.model == "lstm":
                    model = LSTMModel(in_channels=MODEL_IN_CHANNELS, weights_path=model_path)
                else:
                    model = TCNModel(in_channels=MODEL_IN_CHANNELS, weights_path=model_path)

            for side in ["left", "right"]:
                logger.info(f"  -> Processing side: {side}")

                if args.model in {"cnn", "lstm", "tcn"}:
                    # Use all available channels for deep learning models.
                    features = loader.get_features(df_all)
                    feature_cols = [col for col in features.columns if col.endswith(f"_{side}")]
                    # Expected channel order: 3 acc + 3 gyro + 4 quat = 10.
                    all_features = features[feature_cols].to_numpy()
                    
                    results = model.detect(all_features, gyro_data=None)

                    # Convert to discrete events
                    pred_to, pred_ic = probs_to_events(results["probs"])
                    if args.model == "cnn":
                        # Save continuous probabilities
                        df_output[f"Prob_TO_{side}"] = results["probs"][:, 0]
                        df_output[f"Prob_IC_{side}"] = results["probs"][:, 1]
                        df_output[f"Pred_CNN_TO_{side}"] = pred_to
                        df_output[f"Pred_CNN_IC_{side}"] = pred_ic
                    elif args.model == "lstm":
                        df_output[f"Prob_LSTM_TO_{side}"] = results["probs"][:, 0]
                        df_output[f"Prob_LSTM_IC_{side}"] = results["probs"][:, 1]
                        df_output[f"Pred_LSTM_TO_{side}"] = pred_to
                        df_output[f"Pred_LSTM_IC_{side}"] = pred_ic
                    else:
                        df_output[f"Prob_TCN_TO_{side}"] = results["probs"][:, 0]
                        df_output[f"Prob_TCN_IC_{side}"] = results["probs"][:, 1]
                        df_output[f"Pred_TCN_TO_{side}"] = pred_to
                        df_output[f"Pred_TCN_IC_{side}"] = pred_ic

                try:
                    eval_metrics = evaluator.evaluate_file(
                        pred_to=pred_to,
                        pred_ic=pred_ic,
                        true_to=df_labels[f"TO_{side}"].values,
                        true_ic=df_labels[f"IC_{side}"].values,
                    )
                    eval_metrics["filename"] = input_file.name
                    eval_metrics["side"] = side
                    evaluation_results.append(eval_metrics)

                    logger.info(f"     MAE TO : {eval_metrics['MAE_TO']:.1f} ms")
                    logger.info(f"     MAE IC : {eval_metrics['MAE_IC']:.1f} ms")
                    logger.info(f"     Acc TO (20ms) : {eval_metrics['Acc_TO_20ms']:.1f}%")
                    logger.info(f"     Acc IC (20ms) : {eval_metrics['Acc_IC_20ms']:.1f}%")
                    logger.info(f"     Acc TO (50ms) : {eval_metrics['Acc_TO_50ms']:.1f}%")
                    logger.info(f"     Acc IC (50ms) : {eval_metrics['Acc_IC_50ms']:.1f}%")
                except Exception as exc:
                    logger.warning(f"  [!] Evaluation failed ({side}): {exc}")

            output_path = output_dir / input_file.name
            df_output.to_csv(output_path, index=False)
            logger.info(f"  -> Saved: {output_path.name}")

            success_count += 1

        except Exception as exc:
            logger.error(f"  [X] Error: {exc}", exc_info=True)
            continue

    logger.info(f"\n{'='*60}")
    logger.info("FINAL REPORT")
    logger.info(f"{'='*60}")
    logger.info(f"Files processed: {success_count}/{len(input_files)}")

    if evaluation_results:
        logger.info(f"\n{'='*60}")
        logger.info("GLOBAL STATISTICS")
        logger.info(f"{'='*60}\n")

        summary = evaluator.aggregate_results(evaluation_results)
        print(summary.round(2))

        summary_path = output_dir / "evaluation_summary.csv"
        summary.to_csv(summary_path)
        logger.info(f"\n-> Report: {summary_path}")

        details_path = output_dir / "evaluation_details.csv"
        pd.DataFrame(evaluation_results).to_csv(details_path, index=False)
        logger.info(f"-> Details: {details_path}")
    else:
        logger.warning("\n[!] No evaluation available (missing labels)")

    logger.info(f"\n{'='*60}")
    logger.info("PIPELINE COMPLETED")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
