#!/usr/bin/env python
"""
This script splits the provided dataframe into train and test sets.
"""
import argparse
import logging
import pandas as pd
import wandb
import os
import tempfile
from sklearn.model_selection import train_test_split
from wandb_utils.log_artifact import log_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    run = wandb.init(job_type="train_val_test_split")
    run.config.update(args)

    # Download input artifact
    logger.info(f"Fetching artifact {args.input}")
    artifact_local_path = run.use_artifact(args.input).file()
    logger.info(f"Artifact local path: {artifact_local_path}")

    df = pd.read_csv(artifact_local_path)
    logger.info(f"Dataframe shape: {df.shape}")

    # Split the data
    logger.info("Splitting trainval and test")
    trainval, test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=df[args.stratify_by] if args.stratify_by != 'none' else None,
    )
    logger.info(f"Trainval shape: {trainval.shape}, Test shape: {test.shape}")

    # Save the artifacts
    with tempfile.TemporaryDirectory() as tmp_dir:
        for df_split, split_name in zip([trainval, test], ['trainval', 'test']):
            logger.info(f"Uploading {split_name}_data.csv dataset")

            temp_filename = os.path.join(tmp_dir, f"{split_name}_data.csv")
            logger.info(f"Temporary filename: {temp_filename}")

            try:
                # Write dataframe to CSV
                df_split.to_csv(temp_filename, index=False)
                logger.info(f"File {temp_filename} written successfully")

                # Log the artifact
                artifact = wandb.Artifact(
                    name=f"{split_name}_data",
                    type="dataset",
                    description=f"{split_name} split of dataset",
                )
                artifact.add_file(temp_filename)

                logger.info(f"Logging artifact {split_name}_data")
                run.log_artifact(artifact)
                artifact.wait()
                logger.info(f"Artifact {split_name}_data logged successfully")
            except Exception as e:
                logger.error(f"Failed to write file {temp_filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split test and remainder")

    parser.add_argument("--input", type=str, help="Input artifact to split", required=True)
    parser.add_argument("--test_size", type=float, help="Size of the test split. Fraction of the dataset, or number of items", required=True)
    parser.add_argument("--random_seed", type=int, help="Seed for random number generator", default=42, required=False)
    parser.add_argument("--stratify_by", type=str, help="Column to use for stratification", default='none', required=False)

    args = parser.parse_args()

    go(args)
