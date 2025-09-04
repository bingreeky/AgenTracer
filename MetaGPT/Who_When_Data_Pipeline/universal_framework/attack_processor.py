#!/usr/bin/env python3
"""
Attack Processor - Handle attack tasks
"""

import os
import sys
import json
import asyncio
from typing import Dict, List, Any, Optional
import subprocess

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from base_processor import BaseProcessor, ProcessingStats
from config import FrameworkConfig, ProcessType

# Lazy import attack-related modules
process_success_log = None


def _import_attack_module():
    """Lazy import attack module"""
    global process_success_log
    if process_success_log is None:
        try:
            from attack_success_log_metagpt import process_success_log
        except ImportError as e:
            print(f"Warning: Unable to import attack_success_log_metagpt: {e}")
            print("Please ensure the file exists and the path is correct")


class AttackProcessor(BaseProcessor):
    """Attack processor"""

    def __init__(self, config: FrameworkConfig):
        super().__init__(config)
        self.process_type = ProcessType.ATTACK

    def load_tasks(self, round_num: int = 1) -> List[Dict[str, Any]]:
        """Load attack tasks"""
        if round_num == 1:
            # Round 1: Load from original successful task file
            task_file = self.get_task_file_path(round_num, "success")
        else:
            # Subsequent rounds: Load from previous round's attack failed task file
            task_file = self.get_attack_failed_file(round_num - 1)

        return self.load_tasks_from_file(task_file)

    async def process_single_task(self, task: Dict[str, Any]) -> bool:
        """Process single attack task"""
        if not self.validate_task(task):
            return False

        task_id = task["task_id"]
        print(f"Processing attack task: {task_id}")

        try:
            # Set environment variables
            self._set_attack_environment()

            # Check if attack analysis file already exists
            if self.check_analysis_file_exists(task_id, "attack"):
                print(f"[ATTACK_ANALYSIS] Attack analysis file already exists, loading existing file...")
                existing_analysis = self.load_existing_analysis(task_id, "attack")
                if existing_analysis:
                    print(f"[ATTACK_ANALYSIS] Successfully loaded existing attack analysis: {task_id}")
                    self.record_loaded_existing(task_id)
                    return True
                else:
                    print(
                        f"[ATTACK_ANALYSIS] Failed to load existing attack analysis, will regenerate: {task_id}"
                    )

            # Lazy import attack module (after environment variables are set)
            _import_attack_module()

            # Prepare data to pass to process_success_log
            if "log_data" in task:
                # If log_data is already in correct format, use it directly
                success_log = task["log_data"]
            elif "original_log" in task:
                # If original_log exists, use it
                success_log = task["original_log"]
            else:
                print(f"Attack task {task_id} missing correct log data format")
                return False

            # Ensure success_log contains all required fields
            if "question_ID" not in success_log:
                success_log["question_ID"] = task_id

            # Validate required fields exist
            required_fields = [
                "question",
                "question_ID",
                "ground_truth",
                "model_prediction",
            ]
            for field in required_fields:
                if field not in success_log:
                    print(f"Attack task {task_id} missing required field: {field}")
                    return False

            # Load historical attack analysis information
            current_round = getattr(self, "current_round", 1)
            previous_attack_analyses = self._load_previous_attack_analyses(
                task_id, current_round
            )

            # Validate historical data format
            if not self._validate_history_data(previous_attack_analyses, "Attack Analysis"):
                print(f"[WARNING] Historical attack analysis data format validation failed, using empty history information")
                previous_attack_analyses = []

            # Add historical information to success_log
            success_log["previous_attack_analyses"] = previous_attack_analyses

            if previous_attack_analyses:
                print(
                    f"[HISTORY] Loaded {len(previous_attack_analyses)} rounds of historical attack analyses for task {task_id}"
                )
                for i, analysis in enumerate(previous_attack_analyses):
                    print(
                        f"  Round {analysis.get('round', i+1)}: step {analysis.get('attack_step', 'Unknown')}"
                    )
            else:
                print(f"[HISTORY] Task {task_id} has no historical attack analysis information")

            # Use retry mechanism to call attack processing function
            await self.retry_with_backoff(process_success_log, success_log)

            print(f"Attack task {task_id} processed successfully")
            return True

        except Exception as e:
            print(f"Attack task {task_id} processing failed: {e}")
            return False

    def _set_attack_environment(self):
        """Set attack environment variables"""
        # Set attack script directory configuration
        current_round = getattr(self, "current_round", 1)

        attack_suggestions_dir = self.directories["attack_suggestions"].format(
            current_round
        )
        attacked_logs_dir = self.directories["attacked_logs"].format(current_round)
        attack_records_dir = self.directories["attack_records"].format(current_round)

        os.environ["ATTACK_SUGGESTION_LOG_BASE"] = attack_suggestions_dir
        os.environ["ATTACKED_LOG_BASE"] = attacked_logs_dir
        os.environ["ATTACK_RECORD_DIR"] = attack_records_dir

        print(f"Setting attack environment variables:")
        print(f"  ATTACK_SUGGESTION_LOG_BASE: {attack_suggestions_dir}")
        print(f"  ATTACKED_LOG_BASE: {attacked_logs_dir}")
        print(f"  ATTACK_RECORD_DIR: {attack_records_dir}")

    async def process_round(
        self, round_num: int, max_tasks: Optional[int] = None
    ) -> ProcessingStats:
        """Process a single round of attack tasks"""
        # Set current round
        self.current_round = round_num

        # Set environment variables at the beginning of the round
        self._set_attack_environment()

        # Call parent class method
        stats = await super().process_round(round_num, max_tasks)

        # Save statistics
        self.save_stats(round_num, stats)

        return stats

    def validate_task(self, task: Dict[str, Any]) -> bool:
        """Validate attack task format"""
        required_fields = ["task_id"]

        for field in required_fields:
            if field not in task:
                print(f"Attack task missing required field: {field}")
                return False

        # Check if log_data or original_log exists
        if "log_data" not in task and "original_log" not in task:
            print(f"Attack task {task['task_id']} missing log data")
            return False

        return True

    def get_attack_result_file(self, round_num: int) -> str:
        """Get attack result file path"""
        return self.directories["attacked_still_succeed"].format(round_num)

    def check_attack_completion(self, round_num: int) -> bool:
        """Check if attack is complete"""
        result_file = self.get_attack_result_file(round_num)
        return os.path.exists(result_file)

    def get_attack_stats(self, round_num: int) -> Dict[str, Any]:
        """Get attack statistics"""
        stats_file = os.path.join(
            self.config.output_base_dir, "stats", f"round_{round_num}_stats.json"
        )

        if os.path.exists(stats_file):
            with open(stats_file, "r", encoding="utf-8") as f:
                return json.load(f)

        return {}

    def run_evalplus(self, round_num: int):
        """Run evalplus evaluation"""
        # Use parent class's evaluation method
        super().run_evalplus(round_num, "attack")

    def _run_kodcode_evalplus(self, round_num: int, logs_dir: str):
        """Run KodCode evalplus evaluation"""
        print(f"Running KodCode {round_num} round attack result evaluation")

        # 1. Get original eval result file to determine task difficulty
        original_eval_file = self.config.get_original_eval_result_file()
        if not os.path.exists(original_eval_file):
            print(f"Warning: Original eval result file does not exist: {original_eval_file}")
            return

        # 2. Read original eval results to determine task difficulty
        with open(original_eval_file, "r", encoding="utf-8") as f:
            original_results = json.load(f)

        # 3. Group tasks by difficulty
        difficulty_groups = self._group_tasks_by_difficulty(original_results, logs_dir)

        # 4. Run evalplus for each difficulty level
        for difficulty, task_ids in difficulty_groups.items():
            if not task_ids:
                print(f"No tasks to evaluate for difficulty {difficulty}")
                continue

            print(f"Evaluating tasks for difficulty {difficulty}: {len(task_ids)} tasks")
            self._run_kodcode_evalplus_for_difficulty(
                round_num, logs_dir, difficulty, task_ids
            )

    def _group_tasks_by_difficulty(self, original_results: dict, logs_dir: str) -> dict:
        """Group tasks by difficulty based on original eval results and log files"""
        difficulty_groups = {"hard": [], "medium": [], "easy": []}

        # Check files in logs directory
        if not os.path.exists(logs_dir):
            print(f"Warning: Logs directory does not exist: {logs_dir}")
            return difficulty_groups

        # Iterate through files in logs directory
        for log_file in os.listdir(logs_dir):
            if not log_file.endswith(".json"):
                continue

            task_id = log_file.replace(".json", "")

            # Check if task is in original results
            if task_id in original_results:
                # Determine difficulty based on original result file
                # Difficulty needs to be determined based on the actual eval result file structure
                # Assuming original result file is separated by difficulty
                difficulty = self._determine_task_difficulty(task_id, original_results)
                if difficulty in difficulty_groups:
                    difficulty_groups[difficulty].append(task_id)

        return difficulty_groups

    def _determine_task_difficulty(self, task_id: str, original_results: dict) -> str:
        """Determine task difficulty level"""
        # Check which difficulty level the task is in
        eval_results_dir = os.path.join(
            self.config.workspace_dir, "coding", "eval_results"
        )

        # Check different difficulty result files in order of priority
        difficulty_files = [
            ("hard", "kodcode-hard.json"),
            ("medium", "kodcode-medium.json"),
            ("easy", "kodcode-easy.json"),
        ]

        for difficulty, filename in difficulty_files:
            result_path = os.path.join(eval_results_dir, filename)
            if os.path.exists(result_path):
                try:
                    with open(result_path, "r", encoding="utf-8") as f:
                        results = json.load(f)

                    # Check if task is in this difficulty level
                    if task_id in results:
                        print(f"Task {task_id} belongs to {difficulty} difficulty level")
                        return difficulty

                except Exception as e:
                    print(f"Warning: Failed to read eval result file {filename}: {e}")
                    continue

        # If difficulty cannot be determined, use default value
        print(f"Warning: Difficulty for task {task_id} cannot be determined, using default 'hard'")
        return "hard"

    def _run_kodcode_evalplus_for_difficulty(
        self, round_num: int, logs_dir: str, difficulty: str, task_ids: list
    ):
        """Run evalplus evaluation for specific difficulty"""
        # Build evalplus command
        dataset_file = f"datasets/code/kodcode-light-rl-10k-{difficulty}.parquet"
        dataset_path = os.path.join(self.config.workspace_dir, "coding", dataset_file)

        # Build result save path (avoid overwriting previous results)
        eval_results_dir = os.path.join(
            self.config.workspace_dir, "coding", "eval_results"
        )
        result_file = f"kodcode-{difficulty}-attack-round-{round_num}.json"
        result_path = os.path.join(eval_results_dir, result_file)

        # Ensure result directory exists
        os.makedirs(eval_results_dir, exist_ok=True)

        # Build evalplus command
        evalplus_cmd = [
            sys.executable,  # Use current Python interpreter
            os.path.join(self.config.workspace_dir, "coding", "run_eval.py"),
            "--dataset",
            dataset_path,
            "--concurrency",
            "10",
        ]

        print(f"Running KodCode {difficulty} difficulty evaluation:")
        print(f"  Dataset: {dataset_path}")
        print(f"  Log directory: {logs_dir}")
        print(f"  Result file: {result_path}")
        print(f"  Command: {' '.join(evalplus_cmd)}")

        try:
            # Set environment variables, so run_eval.py knows where to read logs from
            env = os.environ.copy()
            env["LOG_ROOT"] = logs_dir
            env["EVAL_ROUND"] = str(round_num)
            env["EVAL_DIFFICULTY"] = difficulty
            env["EVAL_TYPE"] = "attack"

            # Execute command
            result = subprocess.run(
                evalplus_cmd,
                capture_output=True,
                text=True,
                cwd=os.path.join(self.config.workspace_dir, "coding"),
                env=env,
            )

            if result.returncode == 0:
                print(f"KodCode {difficulty} difficulty evaluation executed successfully")
                if result.stdout:
                    print(f"Output: {result.stdout}")

                # Check if result file was generated
                if os.path.exists(result_path):
                    print(f"Evaluation results saved to: {result_path}")
                else:
                    print(f"Warning: Evaluation result file not generated: {result_path}")
            else:
                print(f"KodCode {difficulty} difficulty evaluation failed")
                print(f"Error: {result.stderr}")

        except Exception as e:
            print(f"Exception occurred during KodCode {difficulty} difficulty evaluation: {e}")

    def _run_generic_evalplus(self, round_num: int, logs_dir: str):
        """Run generic evalplus evaluation (non-KodCode dataset)"""
        # Build evalplus command
        evalplus_cmd = f"""
        cd {self.config.workspace_dir}/coding
        python run_eval.py --dataset {self.config.dataset_type.value} --result_dir {logs_dir}
        """

        print(f"Running generic evalplus command: {evalplus_cmd}")

        # Here you can actually execute the command
        # import subprocess
        # result = subprocess.run(evalplus_cmd, shell=True, capture_output=True, text=True)
        # print(f"Evalplus execution result: {result.stdout}")

        print("Generic evalplus evaluation completed")

    def process_attack_results(self, round_num: int):
        """Process attack results"""
        # Use parent class's evaluation result processing method
        return super().process_eval_results(round_num, "attack")

    def _get_task_difficulty(self, task_id: str) -> str:
        """Get task difficulty level"""
        # Check which difficulty level the task is in
        eval_results_dir = os.path.join(
            self.config.workspace_dir, "coding", "eval_results"
        )

        # Check different difficulty result files in order of priority
        difficulty_files = [
            ("hard", "kodcode-hard.json"),
            ("medium", "kodcode-medium.json"),
            ("easy", "kodcode-easy.json"),
        ]

        for difficulty, filename in difficulty_files:
            result_path = os.path.join(eval_results_dir, filename)
            if os.path.exists(result_path):
                try:
                    with open(result_path, "r", encoding="utf-8") as f:
                        results = json.load(f)

                    # Check if task is in this difficulty level
                    if task_id in results:
                        return difficulty

                except Exception as e:
                    print(f"Warning: Failed to read eval result file {filename}: {e}")
                    continue

        # If difficulty cannot be determined, use default value
        return "hard"
