#!/usr/bin/env python3
"""
Diagnosis Processor - Handle diagnosis tasks
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

# Lazy import diagnosis-related modules
process_failed_log = None


def _import_diagnose_module():
    """Lazy import diagnosis module"""
    global process_failed_log
    if process_failed_log is None:
        try:
            from diagnose_fail_log_metagpt import process_failed_log
        except ImportError as e:
            print(f"Warning: Unable to import diagnose_fail_log_metagpt: {e}")
            print("Please ensure the file exists and the path is correct")


class DiagnoseProcessor(BaseProcessor):
    """Diagnosis processor"""

    def __init__(self, config: FrameworkConfig):
        super().__init__(config)
        self.process_type = ProcessType.DIAGNOSE

    def load_tasks(self, round_num: int = 1) -> List[Dict[str, Any]]:
        """Load diagnosis tasks"""
        if round_num == 1:
            # Round 1: Load from original failed task file
            task_file = self.get_task_file_path(round_num, "failed")
        else:
            # Subsequent rounds: Load from previous round's diagnosis failed task file
            task_file = self.get_diagnose_failed_file(round_num - 1)

        return self.load_tasks_from_file(task_file)

    async def process_single_task(self, task: Dict[str, Any]) -> bool:
        """Process single diagnosis task"""
        if not self.validate_task(task):
            return False

        task_id = task["task_id"]
        print(f"Processing diagnosis task: {task_id}")

        try:
            # Set environment variables
            self._set_diagnose_environment()

            # Check if diagnosis analysis file already exists
            if self.check_analysis_file_exists(task_id, "diagnose"):
                print(f"[DIAGNOSE_ANALYSIS] Diagnosis analysis file already exists, loading existing file...")
                existing_analysis = self.load_existing_analysis(task_id, "diagnose")
                if existing_analysis:
                    print(f"[DIAGNOSE_ANALYSIS] Successfully loaded existing diagnosis analysis: {task_id}")
                    self.record_loaded_existing(task_id)
                    return True
                else:
                    print(
                        f"[DIAGNOSE_ANALYSIS] Failed to load existing diagnosis analysis, will regenerate: {task_id}"
                    )

            # Lazy import diagnosis module (after environment variables are set)
            _import_diagnose_module()

            # Prepare data to pass to process_failed_log
            if "log_data" in task:
                # If log_data is already in correct format, use it directly
                failed_log = task["log_data"]
            elif "original_log" in task:
                # If original_log exists, use it
                failed_log = task["original_log"]
            else:
                print(f"Diagnosis task {task_id} missing correct log data format")
                return False

            # Ensure failed_log contains task_id
            if "task_id" not in failed_log:
                failed_log["task_id"] = task_id

            # Validate required fields exist
            required_fields = [
                "task_id",
                "question",
                "ground_truth",
                "model_prediction",
            ]
            for field in required_fields:
                if field not in failed_log:
                    print(f"Diagnosis task {task_id} missing required field: {field}")
                    return False

            # Load historical diagnosis analysis information
            current_round = getattr(self, "current_round", 1)
            previous_diagnosis_analyses = self._load_previous_diagnosis_analyses(
                task_id, current_round
            )

            # Validate historical data format
            if not self._validate_history_data(previous_diagnosis_analyses, "diagnosis analysis"):
                print(f"[WARNING] Historical diagnosis analysis data format validation failed, will use empty history")
                previous_diagnosis_analyses = []

            # Add historical information to failed_log
            failed_log["previous_diagnosis_analyses"] = previous_diagnosis_analyses

            if previous_diagnosis_analyses:
                print(
                    f"[HISTORY] Loaded {len(previous_diagnosis_analyses)} rounds of historical diagnosis analysis for task {task_id}"
                )
                for i, analysis in enumerate(previous_diagnosis_analyses):
                    print(
                        f"  Round {analysis.get('round', i+1)}: step {analysis.get('mistake_step', 'Unknown')}"
                    )
            else:
                print(f"[HISTORY] Task {task_id} has no historical diagnosis analysis information")

            # Use retry mechanism to call diagnosis processing function
            await self.retry_with_backoff(process_failed_log, failed_log)

            print(f"Diagnosis task {task_id} processed successfully")
            return True

        except Exception as e:
            print(f"Diagnosis task {task_id} processing failed: {e}")
            return False

    def _set_diagnose_environment(self):
        """Set diagnosis environment variables"""
        # Set diagnosis script directory configuration
        current_round = getattr(self, "current_round", 1)

        diagnosed_logs_dir = self.directories["diagnosed_logs"].format(current_round)
        improved_logs_dir = self.directories["improved_logs"].format(current_round)
        diagnosis_records_dir = self.directories["diagnosis_records"].format(
            current_round
        )

        os.environ["DIAGNOSED_LOG_BASE"] = diagnosed_logs_dir
        os.environ["IMPROVED_LOG_BASE"] = improved_logs_dir
        os.environ["DIAGNOSIS_RECORD_DIR"] = diagnosis_records_dir

        print(f"Setting diagnosis environment variables:")
        print(f"  DIAGNOSED_LOG_BASE: {diagnosed_logs_dir}")
        print(f"  IMPROVED_LOG_BASE: {improved_logs_dir}")
        print(f"  DIAGNOSIS_RECORD_DIR: {diagnosis_records_dir}")

    async def process_round(
        self, round_num: int, max_tasks: Optional[int] = None
    ) -> ProcessingStats:
        """Process a round of diagnosis tasks"""
        # Set current round
        self.current_round = round_num

        # Set environment variables at the beginning of the round
        self._set_diagnose_environment()

        # Call parent class method
        stats = await super().process_round(round_num, max_tasks)

        # Save statistics
        self.save_stats(round_num, stats)

        return stats

    def validate_task(self, task: Dict[str, Any]) -> bool:
        """Validate diagnosis task format"""
        required_fields = ["task_id"]

        for field in required_fields:
            if field not in task:
                print(f"Diagnosis task missing required field: {field}")
                return False

        # Check if log_data or original_log exists
        if "log_data" not in task and "original_log" not in task:
            print(f"Diagnosis task {task['task_id']} missing log data")
            return False

        return True

    def get_diagnose_result_file(self, round_num: int) -> str:
        """Get diagnosis result file path"""
        return self.directories["replayed_still_failed"].format(round_num)

    def check_diagnose_completion(self, round_num: int) -> bool:
        """Check if diagnosis is complete"""
        result_file = self.get_diagnose_result_file(round_num)
        return os.path.exists(result_file)

    def get_diagnose_stats(self, round_num: int) -> Dict[str, Any]:
        """Get diagnosis statistics"""
        stats_file = os.path.join(
            self.config.output_base_dir, "stats", f"round_{round_num}_stats.json"
        )

        if os.path.exists(stats_file):
            with open(stats_file, "r", encoding="utf-8") as f:
                return json.load(f)

        return {}

    def run_evalplus(self, round_num: int):
        """Run evalplus evaluation"""
        # Use base class evaluation method
        super().run_evalplus(round_num, "diagnose")

    def _run_kodcode_evalplus(self, round_num: int, logs_dir: str):
        """Run KodCode evalplus evaluation"""
        print(f"Running KodCode round {round_num} diagnosis result evaluation")

        # 1. Get original eval result file to determine task difficulty
        original_eval_file = self.config.get_original_eval_result_file()
        if not os.path.exists(original_eval_file):
            print(f"Warning: Original eval result file does not exist: {original_eval_file}")
            return

        # 2. Read original eval results to determine difficulty for each task
        with open(original_eval_file, "r", encoding="utf-8") as f:
            original_results = json.load(f)

        # 3. Group tasks by difficulty
        difficulty_groups = self._group_tasks_by_difficulty(original_results, logs_dir)

        # 4. Run evalplus for each difficulty level
        for difficulty, task_ids in difficulty_groups.items():
            if not task_ids:
                print(f"Difficulty {difficulty} has no tasks to evaluate")
                continue

            print(f"Evaluating difficulty {difficulty} tasks: {len(task_ids)} tasks")
            self._run_kodcode_evalplus_for_difficulty(
                round_num, logs_dir, difficulty, task_ids
            )

    def _group_tasks_by_difficulty(self, original_results: dict, logs_dir: str) -> dict:
        """Group tasks by difficulty based on original eval results and log files"""
        difficulty_groups = {"hard": [], "medium": [], "easy": []}

        # Check files in logs directory
        if not os.path.exists(logs_dir):
            print(f"Warning: logs directory does not exist: {logs_dir}")
            return difficulty_groups

        # Iterate through files in logs directory
        for log_file in os.listdir(logs_dir):
            if not log_file.endswith(".json"):
                continue

            task_id = log_file.replace(".json", "")

            # Check if task exists in original results
            if task_id in original_results:
                # Determine difficulty based on original result file
                # This needs to be adjusted based on actual eval result file structure
                # Assuming original result files are separated by difficulty
                difficulty = self._determine_task_difficulty(task_id, original_results)
                if difficulty in difficulty_groups:
                    difficulty_groups[difficulty].append(task_id)

        return difficulty_groups

    def _determine_task_difficulty(self, task_id: str, original_results: dict) -> str:
        """Determine task difficulty level"""
        # Check which difficulty level result file the task belongs to
        eval_results_dir = os.path.join(
            self.config.workspace_dir, "coding", "eval_results"
        )

        # Check result files for different difficulties by priority
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

                    # Check if task exists in this difficulty level results
                    if task_id in results:
                        print(f"Task {task_id} belongs to {difficulty} difficulty level")
                        return difficulty

                except Exception as e:
                    print(f"Warning: Failed to read eval result file {filename}: {e}")
                    continue

        # If unable to determine difficulty, use default value
        print(f"Warning: Unable to determine difficulty level for task {task_id}, using default value 'hard'")
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
        result_file = f"kodcode-{difficulty}-diagnose-round-{round_num}.json"
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

        print(f"Running KodCode {difficulty} difficulty diagnosis evaluation:")
        print(f"  Dataset: {dataset_path}")
        print(f"  Log directory: {logs_dir}")
        print(f"  Result file: {result_path}")
        print(f"  Command: {' '.join(evalplus_cmd)}")

        try:
            # Set environment variables to let run_eval.py know where to read logs from
            env = os.environ.copy()
            env["LOG_ROOT"] = logs_dir
            env["EVAL_ROUND"] = str(round_num)
            env["EVAL_DIFFICULTY"] = difficulty
            env["EVAL_TYPE"] = "diagnose"

            # Execute command
            result = subprocess.run(
                evalplus_cmd,
                capture_output=True,
                text=True,
                cwd=os.path.join(self.config.workspace_dir, "coding"),
                env=env,
            )

            if result.returncode == 0:
                print(f"KodCode {difficulty} difficulty diagnosis evaluation executed successfully")
                if result.stdout:
                    print(f"Output: {result.stdout}")

                # Check if result file was generated
                if os.path.exists(result_path):
                    print(f"Diagnosis evaluation results saved to: {result_path}")
                else:
                    print(f"Warning: Diagnosis evaluation result file not generated: {result_path}")
            else:
                print(f"KodCode {difficulty} difficulty diagnosis evaluation execution failed")
                print(f"Error: {result.stderr}")

        except Exception as e:
            print(f"Exception occurred while executing KodCode {difficulty} difficulty diagnosis evaluation: {e}")

    def _run_generic_evalplus(self, round_num: int, logs_dir: str):
        """Run generic evalplus evaluation (non-KodCode datasets)"""
        # Build evalplus command
        evalplus_cmd = f"""
        cd {self.config.workspace_dir}/coding
        python run_eval.py --dataset {self.config.dataset_type.value} --result_dir {logs_dir}
        """

        print(f"Running generic evalplus command: {evalplus_cmd}")

        # This can actually execute the command
        # import subprocess
        # result = subprocess.run(evalplus_cmd, shell=True, capture_output=True, text=True)
        # print(f"Evalplus execution result: {result.stdout}")

        print("Generic evalplus evaluation completed")

    def process_diagnose_results(self, round_num: int):
        """Process diagnosis results"""
        # Use base class evaluation result processing method
        return super().process_eval_results(round_num, "diagnose")

    def _get_task_difficulty(self, task_id: str) -> str:
        """Get task difficulty level"""
        # Check which difficulty level result file the task belongs to
        eval_results_dir = os.path.join(
            self.config.workspace_dir, "coding", "eval_results"
        )

        # Check result files for different difficulties by priority
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

                    # Check if task exists in this difficulty level results
                    if task_id in results:
                        return difficulty

                except Exception as e:
                    print(f"Warning: Failed to read eval result file {filename}: {e}")
                    continue

        # If unable to determine difficulty, use default value
        return "hard"
