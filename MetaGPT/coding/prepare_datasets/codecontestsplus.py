import argparse
import pathlib

import datasets
import rich
from sandbox_fusion import RunStatus

from coding.utils import code_exec

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

def process_fn(example, empty_return=None):
    reference_solution = example["solution"]
    test_code = "from solution import *\n" + example["test"].strip()
    # skip it if reference solution requires libs from block_libs
    if any(lib in reference_solution for lib in block_libs):
        return empty_return
    if any(lib in test_code for lib in block_libs):
        return empty_return
    question = f"Please solve the programming task below in Python. \n\n{example['question'].strip()}"
    test_declaration = example["test_info"][0]["function_declaration"].strip()
    if test_declaration and test_declaration.strip():
        question += (
            f"\n\nNote that the function declaration is {test_declaration}. "
        )

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
        'task_id': example['question_id'],
        'question': question,
        'test': test_code,
        'entry_point': example['test_info'][0]['function_declaration'].strip(),
        'reference_solution': reference_solution,
        'difficulty': example['gpt_difficulty'],
    }


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--save_dir", default="datasets/code/")
    args = argparser.parse_args()

    data_source = "deepmind/codecontests"
    dataset = datasets.load_dataset(data_source, split="train")
    empty_return = {k: None for k in dataset.column_names}
    dataset = dataset.map(function=lambda x: process_fn(x, empty_return=empty_return), remove_columns=dataset.column_names, num_proc=64).filter(
        lambda x: x != empty_return
    )
    save_path = pathlib.Path(args.save_dir) / "kodcode-light-rl-10k.parquet"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(save_path)