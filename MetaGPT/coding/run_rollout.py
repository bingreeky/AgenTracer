import os
import sys
import json
import shutil

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "/data/home_data/R1-Tracer/MetaGPT")
    )
)
from metagpt.software_company import generate_repo
import asyncio
from tqdm.asyncio import tqdm
import importlib.util
import datasets
import pathlib
from termcolor import cprint

from coding.constants import LOG_ROOT, WORKSPACE_ROOT


def standardize_role_name(name: str) -> str:
    """Standardize role names to ensure correct role names are used instead of personal names"""
    # Role name mapping
    role_mapping = {
        # Personal name to role name mapping
        "Mike": "Team Leader",
        "Alice": "Product Manager",
        "Bob": "Architect",
        "Alex": "Engineer",
        "David": "Data Analyst",
        # Keep existing role names unchanged
        "Team Leader": "Team Leader",
        "Product Manager": "Product Manager",
        "Architect": "Architect",
        "Engineer": "Engineer",
        "Data Analyst": "Data Analyst",
        # Handle possible variants
        "TeamLeader": "Team Leader",
        "ProductManager": "Product Manager",
        "DataAnalyst": "Data Analyst",
        "Engineer2": "Engineer",
    }

    return role_mapping.get(name, name)


# Mapping of role name to prompt module and variable name
PROMPT_MODULE_MAP = {
    "Team Leader": ("metagpt.prompts.di.team_leader", "TL_INSTRUCTION"),
    "Engineer": ("metagpt.prompts.di.engineer2", "WRITE_CODE_SYSTEM_PROMPT"),
    "Architect": ("metagpt.prompts.di.architect", "ARCHITECT_INSTRUCTION"),
    "Product Manager": ("metagpt.prompts.product_manager", "EXTRA_INSTRUCTION"),
    "Data Analyst": ("metagpt.prompts.di.data_analyst", "EXTRA_INSTRUCTION"),
}


def load_prompt(role_name: str):
    module_name, var_name = PROMPT_MODULE_MAP[role_name]
    module = importlib.import_module(module_name)
    return getattr(module, var_name, f"[{var_name} not found in {module_name}]")


PROMPT_MAP = {role: load_prompt(role) for role in PROMPT_MODULE_MAP.keys()}


async def run_coding_task(
    task: dict, log: pathlib.Path, project: pathlib.Path, skip_existing: bool
):
    data_source = task["data_source"]
    task_id = task["task_id"]
    idea = (
        task["question"]
        + f"I wish you finish the task with a multi-agent cooperation"
        + f"The file name of your solution must be 'solution.py'"
    )
    if log.exists():
        if skip_existing:
            cprint(
                f"[INFO] Log for task {data_source}/{task_id} already exists, skipping...",
                "yellow",
            )
            return
        else:
            cprint(
                f"[INFO] Log for task {data_source}/{task_id} already exists, overwriting...",
                "yellow",
            )
            shutil.rmtree(project)
            log.unlink()

    cprint(f"[INFO] Running task {data_source}/{task_id}...", "green")

    history = []
    step_counter = {"step": 0}
    solution_path = project / f"solution.py"

    def log_step(content, role, name):
        # Standardize role name
        standardized_name = standardize_role_name(name)

        # Check if it's terminal output, if so merge into previous step
        if role == "Terminal" and content.startswith("Terminal output:"):
            # Merge terminal output into the previous step
            if history:
                last_step = history[-1]
                # Add terminal output into content, ensuring it has the "Terminal output:" prefix
                if not content.startswith("Terminal output:"):
                    content = f"Terminal output: {content}"
                last_step["content"] += f"\n\n{content}"
                print(
                    f"[LOG_STEP] Terminal output merged into previous step: {last_step['step']}"
                )
            else:
                # If there is no previous step, create a new step
                if not content.startswith("Terminal output:"):
                    content = f"Terminal output: {content}"
                history.append(
                    {
                        "step": step_counter["step"],
                        "content": content,
                        "role": role,
                        "name": standardized_name,
                    }
                )
                step_counter["step"] += 1
            return

        # Record normal step
        history.append(
            {
                "step": step_counter["step"],
                "content": content,
                "role": role,
                "name": standardized_name,
            }
        )
        step_counter["step"] += 1

    try:
        await generate_repo(
            idea=idea,
            n_round=8,
            code_review=True,
            run_tests=True,
            implement=True,
            project_name=project.name,
            project_path=project,
            project_dir=project,
            log_step=log_step,
            use_async=True,
        )
    except Exception as e:
        cprint(f"Error running task {data_source}/{task_id}: {e}", "red")

    cprint(f"[INFO] Finished task {data_source}/{task_id}, log saved to {log}", "green")

    # Dead-loop protection: detect if the last 3 steps include 3 assistant "end" commands
    last_cmds = []
    for h in history[-3:]:
        if h["role"] == "assistant":
            content = h["content"].strip()
            if '"command_name": "end"' in content:
                last_cmds.append(content)
    if len(last_cmds) >= 3:
        print(
            f"[FATAL] Detected >3 consecutive end/empty command, gracefully ending this task."
        )
        history.append(
            {
                "step": step_counter["step"],
                "content": "[DEADLOOP] Detected dead loop, task ended early.",
                "role": "system",
                "name": "system",
            }
        )
        step_counter["step"] += 1

    if os.path.exists(solution_path):
        model_prediction = solution_path.read_text()
    else:
        cprint(
            f"[WARN] solution.py not found for task {data_source}/{task_id}", "yellow"
        )
        model_prediction = ""

    used_roles = set(h["name"] for h in history if h.get("name") in PROMPT_MODULE_MAP)
    system_prompts = {}
    for role_name in used_roles:
        system_prompts[role_name] = PROMPT_MAP[role_name]

    with open(log, "w", encoding="utf-8") as f:
        json.dump(
            {
                "question": task["question"],
                "question_ID": task["task_id"],
                "ground_truth": task["reference_solution"],
                "model_prediction": model_prediction,
                "history": history,
                "system_prompts": system_prompts,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


async def run_tasks(
    tasks: list[dict], concurrency: int = 10, skip_existing: bool = False
):
    semaphore = asyncio.Semaphore(concurrency)
    data_source = tasks[0]["data_source"]

    async def _run(task: dict):
        async with semaphore:
            task_id = task["task_id"].replace("/", "_")
            project: pathlib.Path = WORKSPACE_ROOT / data_source / task_id
            log: pathlib.Path = LOG_ROOT / data_source / f"{task_id}.json"
            project.mkdir(parents=True, exist_ok=True)
            log.parent.mkdir(parents=True, exist_ok=True)
            await run_coding_task(task, log, project, skip_existing)

    coros = []
    for task in tasks:
        coros.append(_run(task))
    await tqdm.gather(*coros)


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--dataset", type=str, default="datasets/code/kodcode-light-rl-10k-hard.parquet"
    )
    argparser.add_argument("--concurrency", type=int, default=10)
    argparser.add_argument("--skip_existing", "-s", action="store_true")
    args = argparser.parse_args()
    dataset = datasets.load_dataset(
        "parquet", data_files={"train": args.dataset}, split="train"
    )
    data_source = dataset[0]["data_source"]
    cprint(f"[INFO] Loaded {len(dataset)} tasks from {args.dataset}", "green")
    tasks = dataset.to_list()
    for role in PROMPT_MAP.keys():
        cprint(f"System Prompt for {role}: {PROMPT_MAP[role]}", "green")
    asyncio.run(run_tasks(tasks, args.concurrency, args.skip_existing))
