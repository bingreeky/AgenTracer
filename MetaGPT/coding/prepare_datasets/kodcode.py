import argparse
import pathlib

import datasets
import rich
from sandbox_fusion import RunStatus
import sys
import os

# Add the parent directory to sys.path to allow importing from coding package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import code_exec

_EMPTY_RETURN_ = {
    "data_source": "kodcode",
    "task_id": None,
    "question": None,
    "test": None,
    "entry_point": None,
    "reference_solution": None,
}

packages = [
    "beautifulsoup4",
    "fake-useragent",
    "imageio",
    "keras",
    "lxml",
    "matplotlib",
    "numpy",
    "opencv-python",
    "pillow",
    "requests",
    "rich",
    "scikit-learn",
    "sphinx-pyproject",
    "statsmodels",
    "sympy",
    "tweepy",
    "typing_extensions",
    "xgboost",
    "flask",
    "seaborn",
]
block_libs = [
    "fake-useragent",
    "keras",
    "socket",
    "torch",
    "scipy",
    "sklearn",
    "cv2",
    "scipy",
    "imageio",
    "sphinx-pyproject",
    "xgboost",
    "tweepy",
    "flask",
    "matplotlib",
    "pillow",
    "seaborn",
    "smtpd",
    "pandas",
    "bs4",
]


def process_fn(example):
    reference_solution = example["solution"]
    test_code = "from solution import *\n" + example["test"].strip()
    # skip it if reference solution requires libs from block_libs
    if any(lib in reference_solution for lib in block_libs):
        return _EMPTY_RETURN_
    if any(lib in test_code for lib in block_libs):
        return _EMPTY_RETURN_
    question = f"Please solve the programming task below in Python using a multi-agent cooperation. \n\n{example['question'].strip()}"
    test_declaration = example["test_info"][0]["function_declaration"].strip()
    if test_declaration and test_declaration.strip():
        question += f"\n\nNote that the function declaration is {test_declaration}. "

    result = code_exec(code=reference_solution, test=test_code)
    if result.status != RunStatus.Success:
        rich.print(f"[bold red]Test code failed for {example['question_id']}")
        rich.print("[bold red]Reference Solution:")
        print(reference_solution)
        rich.print("[bold red]Stdout:")
        print(result.run_result.stdout)
        rich.print("[bold red]Stderr:")
        print(result.run_result.stderr)
    return {
        "data_source": "kodcode",
        "task_id": example["question_id"],
        "question": question,
        "test": test_code,
        "entry_point": example["test_info"][0]["function_declaration"].strip(),
        "reference_solution": reference_solution,
    }


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--save_dir", default="datasets/code/")
    argparser.add_argument("--difficulty", type=list, default=["hard"])
    args = argparser.parse_args()

    data_source = "KodCode/KodCode-Light-RL-10K"
    dataset = datasets.load_dataset(data_source, split="train")
    dataset = dataset.filter(lambda x: x["gpt_difficulty"] in args.difficulty)
    dataset = dataset.map(
        function=process_fn, remove_columns=dataset.column_names, num_proc=32
    ).filter(lambda x: x != _EMPTY_RETURN_)
    difficulty_suffix = "".join(args.difficulty)
    save_path = (
        pathlib.Path(args.save_dir)
        / f"kodcode-light-rl-10k-{difficulty_suffix}.parquet"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(save_path)
