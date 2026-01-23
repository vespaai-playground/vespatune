import argparse
import json
import logging
import os
import signal
import sys
import traceback

from vespatune.core import VespaTune
from vespatune.db import update_run_status

# Configure logging to file
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def handle_sigterm(signum, frame):
    logger.info("Received SIGTERM. Stopping training...")
    # We rely on Optuna/Core to handle gracefully if possible,
    # but strictly speaking, SIGTERM might just kill us.
    # We'll try to update status before exiting.
    # Note: DB operations in signal handler can be risky.
    sys.exit(0)


signal.signal(signal.SIGTERM, handle_sigterm)


def main():
    parser = argparse.ArgumentParser(description="VespaTune Training Worker")
    parser.add_argument("--run-id", type=int, required=True, help="Database Run ID")
    parser.add_argument("--config", type=str, required=True, help="JSON Configuration string")

    args = parser.parse_args()

    run_id = args.run_id
    try:
        config = json.loads(args.config)

        # Setup logging to run directory
        output_dir = config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        # Add file handler to logger
        fh = logging.FileHandler(os.path.join(output_dir, "run.log"))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Also redirect stdout/stderr to this log file for capture
        sys.stdout = open(os.path.join(output_dir, "stdout.log"), "w")
        sys.stderr = open(os.path.join(output_dir, "stderr.log"), "w")

        logger.info(f"Starting run {run_id} with config: {config}")

        # Initialize VespaTune
        targets = [x.strip() for x in config["target_columns"].split(";")]
        vt = VespaTune(
            train_filename=config["train_filename"],
            valid_filename=config["valid_filename"],
            output=output_dir,
            task=config["task"],
            idx=config["id_column"],
            targets=targets,
            model_type=config["model_type"],
            num_trials=config["num_trials"],
            time_limit=config["time_limit"],
        )

        # Callbacks (stubbed for now, as we need a way to communicate back to API/DB)
        # For now, we just rely on Optuna DB for results.
        # We can update DB heartbeat if needed.

        vt.train()

        logger.info("Training completed successfully.")
        update_run_status(run_id, "completed")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        update_run_status(run_id, "error", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
