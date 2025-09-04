import json
import pathlib
import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "/data/home_data/R1-Tracer/MetaGPT")
    )
)
import datasets
import asyncio
from tqdm.asyncio import tqdm
from sandbox_fusion import (
    set_sandbox_endpoint,
    set_dataset_endpoint,
    RunStatus,
)

from coding.constants import LOG_ROOT, EVAL_RESULT_ROOT
from coding.utils import code_exec_async
from termcolor import cprint

# set_sandbox_endpoint("https://faas-code-sandbox.bytedance.net/")
# set_dataset_endpoint("https://faas-code-sandbox.bytedance.net/online_judge/")
set_sandbox_endpoint("http://localhost:8080/")
set_dataset_endpoint("http://localhost:8080/online_judge/")


async def run_correctness_eval_task(
    task: dict, semaphore: asyncio.Semaphore
) -> tuple[str, bool]:
    async with semaphore:
        solution = task["solution"]
        test = task["test"]
        result = await code_exec_async(solution, test)
        return task["task_id"], result.status == RunStatus.Success


async def run_correctness_eval_tasks(
    tasks: list[dict], concurrency: int
) -> dict[str, bool]:
    """
    Input:
        tasks: list[dict]
    Output:
        dict[str, bool]: {task_id: pass/fail}
    """
    coros = []
    semaphore = asyncio.Semaphore(concurrency)
    for task in tasks:
        coros.append(run_correctness_eval_task(task, semaphore))
    results = await tqdm.gather(*coros, desc="Evaluating correctness")
    results = {task_id: result for task_id, result in results}

    # Read round and difficulty from environment variables (if provided)
    round_num = os.getenv("EVAL_ROUND", "")
    difficulty = os.getenv("EVAL_DIFFICULTY", "")
    eval_type = os.getenv("EVAL_TYPE", "")  # attack or diagnose

    # Build output filename
    if round_num and difficulty:
        if eval_type:
            filename = f"kodcode-{difficulty}-{eval_type}-round-{round_num}.json"
        else:
            filename = f"kodcode-{difficulty}-round-{round_num}.json"
    else:
        # If no environment variables, use the original naming scheme
        filename = f"{data_source}.json"

    save_path = EVAL_RESULT_ROOT / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--dataset", type=str, default="datasets/code/kodcode-light-rl-10k-hard.parquet"
    )
    argparser.add_argument("--concurrency", type=int, default=10)
    args = argparser.parse_args()
    dataset = datasets.load_dataset(
        "parquet", data_files={"train": args.dataset}, split="train"
    )
    data_source = dataset[0]["data_source"]
    id2data = {task["task_id"]: task for task in dataset}

    # Support reading LOG_ROOT from env; use default if not set
    custom_log_root = os.getenv("LOG_ROOT")
    if custom_log_root:
        log_dir = pathlib.Path(custom_log_root)
        cprint(f"Using LOG_ROOT from environment: {log_dir}", "blue")
    else:
        log_dir = LOG_ROOT / data_source
        cprint(f"Using default LOG_ROOT: {log_dir}", "blue")

    # Read round and difficulty from environment variables (if provided)
    round_num = os.getenv("EVAL_ROUND", "")
    difficulty = os.getenv("EVAL_DIFFICULTY", "")
    eval_type = os.getenv("EVAL_TYPE", "")  # attack or diagnose

    cprint(f"Loading eval tasks from {log_dir}...", "green")
    # load all json files in log_dir
    eval_tasks = []
    skipped_tasks = 0

    # If difficulty is specified, only load logs for that difficulty
    if difficulty:
        # Filter log files by difficulty
        # Get all task IDs for the specified difficulty from the dataset
        dataset_task_ids = set(dataset["task_id"])
        cprint(
            f"{difficulty.capitalize()} dataset contains {len(dataset_task_ids)} tasks",
            "blue",
        )

        # Only process tasks that exist in the specified difficulty dataset
        for log_file in log_dir.glob("*.json"):
            log_data = json.loads(log_file.read_text())
            question_id = log_data["question_ID"]

            # Only process tasks that exist in the specified difficulty dataset
            if question_id in dataset_task_ids:
                if question_id not in id2data:
                    cprint(
                        f"Warning: ID '{question_id}' from log file {log_file.name} not found in dataset, skipping...",
                        "yellow",
                    )
                    skipped_tasks += 1
                    continue
                eval_tasks.append(
                    {
                        "task_id": question_id,
                        "solution": log_data["model_prediction"],
                        "test": id2data[question_id]["test"],
                    }
                )
    else:
        # Original logic: read all files
        for log_file in log_dir.glob("*.json"):
            log_data = json.loads(log_file.read_text())
            question_id = log_data["question_ID"]
            # Skip if ID does not exist in the dataset
            if question_id not in id2data:
                cprint(
                    f"Warning: ID '{question_id}' from log file {log_file.name} not found in dataset, skipping...",
                    "yellow",
                )
                skipped_tasks += 1
                continue
            eval_tasks.append(
                {
                    "task_id": question_id,
                    "solution": log_data["model_prediction"],
                    "test": id2data[question_id]["test"],
                }
            )
    cprint(
        f"Loaded {len(eval_tasks)} eval tasks from {log_dir} (skipped {skipped_tasks} tasks due to missing IDs)",
        "green",
    )
    asyncio.run(run_correctness_eval_tasks(eval_tasks, args.concurrency))
