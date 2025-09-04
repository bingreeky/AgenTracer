#!/usr/bin/env python3
"""
Base Processor - Provides common processing logic
"""

import os
import json
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add relative imports
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import FrameworkConfig
from config import DatasetType


@dataclass
class ProcessingStats:
    """Processing statistics"""

    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    processing_time: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # New retry statistics
    retry_stats: Dict[str, Any] = None

    def __post_init__(self):
        if self.retry_stats is None:
            self.retry_stats = {
                "total_retries": 0,
                "successful_retries": 0,
                "failed_retries": 0,
                "retry_counts": {},  # Record retry count for each task
                "loaded_existing": 0,  # Number loaded from files
            }

    def start(self):
        """Start timing"""
        self.start_time = datetime.now()

    def end(self):
        """End timing"""
        self.end_time = datetime.now()
        if self.start_time:
            self.processing_time = (self.end_time - self.start_time).total_seconds()

    def to_dict(self):
        """Convert to dictionary"""
        return {
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "processing_time": self.processing_time,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "retry_stats": self.retry_stats,
        }


class BaseProcessor(ABC):
    """Base processor abstract class"""

    def __init__(self, config: FrameworkConfig):
        self.config = config
        self.stats = ProcessingStats()
        self.setup_directories()

    def setup_directories(self):
        """Setup directory structure"""
        self.directories = self.config.get_directory_structure()

        # Create base directories
        for dir_type, dir_path in self.directories.items():
            if "round_{}" in dir_path:
                # Create directories for all rounds
                for round_num in range(1, self.config.max_rounds + 1):
                    actual_path = dir_path.format(round_num)
                    # Check if it's a file path (ends with .json)
                    if actual_path.endswith(".json"):
                        # This is a file path, only create parent directory
                        parent_dir = os.path.dirname(actual_path)
                        if parent_dir:
                            os.makedirs(parent_dir, exist_ok=True)
                    else:
                        # This is a directory path, create directory
                        os.makedirs(actual_path, exist_ok=True)
            else:
                # Check if it's a file path (ends with .json)
                if dir_path.endswith(".json"):
                    # This is a file path, only create parent directory
                    parent_dir = os.path.dirname(dir_path)
                    if parent_dir:
                        os.makedirs(parent_dir, exist_ok=True)
                else:
                    # This is a directory path, create directory
                    os.makedirs(dir_path, exist_ok=True)

        # Ensure final_results directory exists
        final_results_dir = self.directories.get("final_results")
        if final_results_dir:
            os.makedirs(final_results_dir, exist_ok=True)
            print(f"Ensuring final_results directory exists: {final_results_dir}")

    @abstractmethod
    def load_tasks(self, round_num: int = 1) -> List[Dict[str, Any]]:
        """Load tasks"""
        pass

    @abstractmethod
    async def process_single_task(self, task: Dict[str, Any]) -> bool:
        """Process single task"""
        pass

    async def process_tasks_concurrently(
        self, tasks: List[Dict[str, Any]]
    ) -> List[bool]:
        """Process tasks concurrently"""
        if not self.config.enable_concurrent:
            # Serial processing
            results = []
            for task in tasks:
                result = await self.process_single_task(task)
                results.append(result)
            return results

        # Concurrent processing
        semaphore = asyncio.Semaphore(self.config.concurrent_limit)

        async def process_with_semaphore(task):
            async with semaphore:
                return await self.process_single_task(task)

        return await asyncio.gather(*[process_with_semaphore(task) for task in tasks])

    async def process_round(
        self, round_num: int, max_tasks: Optional[int] = None
    ) -> ProcessingStats:
        """Process a round of tasks"""
        print(f"\n{'='*60}")
        print(f"Starting round {round_num} processing")
        print(f"{'='*60}")

        # Reset statistics
        self.stats = ProcessingStats()
        self.stats.start()

        # Load tasks
        tasks = self.load_tasks(round_num)
        if not tasks:
            print("No tasks to process")
            return self.stats

        if max_tasks:
            tasks = tasks[:max_tasks]

        # Check checkpoint resume: filter out already processed tasks
        filtered_tasks = []
        skipped_tasks = 0

        for task in tasks:
            if self.is_task_completed(round_num, task):
                print(f"Task {task['task_id']} already processed, skipping")
                skipped_tasks += 1
            else:
                filtered_tasks.append(task)

        if skipped_tasks > 0:
            print(f"Skipped {skipped_tasks} already processed tasks")

        if not filtered_tasks:
            print("All tasks already processed")
            self.stats.total_tasks = len(tasks)
            self.stats.successful_tasks = len(tasks) - skipped_tasks
            self.stats.failed_tasks = 0
            self.stats.end()
            return self.stats

        self.stats.total_tasks = len(filtered_tasks)
        print(f"This round needs to process {len(filtered_tasks)} tasks")

        # Process tasks
        results = await self.process_tasks_concurrently(filtered_tasks)

        # Count results
        self.stats.successful_tasks = sum(results)
        self.stats.failed_tasks = len(results) - self.stats.successful_tasks
        self.stats.end()

        print(f"Round {round_num} processing completed")
        print(f"Success: {self.stats.successful_tasks}, Failed: {self.stats.failed_tasks}")
        print(f"Processing time: {self.stats.processing_time:.2f} seconds")

        # Print retry statistics
        self._print_retry_stats()

        return self.stats

    def is_task_completed(self, round_num: int, task: Dict[str, Any]) -> bool:
        """Check if task is already completed"""
        task_id = task["task_id"]

        # Check different output files based on processor type
        if _check_processor_type(self, "AttackProcessor"):
            return self._is_attack_task_completed(round_num, task_id)
        elif _check_processor_type(self, "DiagnoseProcessor"):
            return self._is_diagnose_task_completed(round_num, task_id)
        else:
            # Default check: if task file exists, consider it completed
            return False

    def _is_attack_task_completed(self, round_num: int, task_id: str) -> bool:
        """Check if attack task is completed"""
        # Check attack-related output files
        attack_suggestions_dir = self.directories["attack_suggestions"].format(
            round_num
        )
        attacked_logs_dir = self.directories["attacked_logs"].format(round_num)
        attack_records_dir = self.directories["attack_records"].format(round_num)

        # Build filenames
        safe_task_id = task_id.replace("/", "_")
        attack_suggestion_file = os.path.join(
            attack_suggestions_dir, f"{safe_task_id}.json"
        )
        attacked_log_file = os.path.join(attacked_logs_dir, f"{safe_task_id}.json")
        attack_record_file = os.path.join(attack_records_dir, f"{safe_task_id}.json")

        # If all required files exist, consider task completed
        return (
            os.path.exists(attack_suggestion_file)
            and os.path.exists(attacked_log_file)
            and os.path.exists(attack_record_file)
        )

    def _is_diagnose_task_completed(self, round_num: int, task_id: str) -> bool:
        """Check if diagnosis task is completed"""
        # Check diagnosis-related output files
        diagnosed_logs_dir = self.directories["diagnosed_logs"].format(round_num)
        improved_logs_dir = self.directories["improved_logs"].format(round_num)
        diagnosis_records_dir = self.directories["diagnosis_records"].format(round_num)

        # Build filenames
        safe_task_id = task_id.replace("/", "_")
        diagnosed_log_file = os.path.join(diagnosed_logs_dir, f"{safe_task_id}.json")
        improved_log_file = os.path.join(improved_logs_dir, f"{safe_task_id}.json")
        diagnosis_record_file = os.path.join(
            diagnosis_records_dir, f"{safe_task_id}.json"
        )

        # If all required files exist, consider task completed
        return (
            os.path.exists(diagnosed_log_file)
            and os.path.exists(improved_log_file)
            and os.path.exists(diagnosis_record_file)
        )

    def save_stats(self, round_num: int, stats: ProcessingStats):
        """Save statistics"""
        stats_dir = os.path.join(self.config.output_base_dir, "stats")
        os.makedirs(stats_dir, exist_ok=True)

        stats_file = os.path.join(stats_dir, f"round_{round_num}_stats.json")
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats.to_dict(), f, ensure_ascii=False, indent=2)

        print(f"Statistics saved to: {stats_file}")

    def get_task_file_path(self, round_num: int, task_type: str) -> str:
        """Get task file path"""
        if task_type == "success":
            if round_num == 1:
                return os.path.join(
                    self.config.output_base_dir, "success_tasks_with_logs.json"
                )
            else:
                # Subsequent rounds: load from previous round's attack failed task file
                return self.directories["attacked_still_succeed"].format(round_num - 1)
        elif task_type == "failed":
            if round_num == 1:
                return os.path.join(
                    self.config.output_base_dir, "failed_tasks_with_logs.json"
                )
            else:
                # Subsequent rounds: load from previous round's diagnosis failed task file
                return self.directories["replayed_still_failed"].format(round_num - 1)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    def load_tasks_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load tasks from file"""
        if not os.path.exists(file_path):
            print(f"Task file does not exist: {file_path}")
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tasks = json.load(f)
            print(f"Loaded {len(tasks)} tasks from {file_path}")
            return tasks
        except Exception as e:
            print(f"Failed to load task file {file_path}: {e}")
            return []

    def save_tasks_to_file(self, tasks: List[Dict[str, Any]], file_path: str):
        """Save tasks to file"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)

        print(f"Tasks saved to: {file_path}")

    def validate_task(self, task: Dict[str, Any]) -> bool:
        """Validate task format"""
        required_fields = ["task_id"]

        for field in required_fields:
            if field not in task:
                print(f"Task missing required field: {field}")
                return False

        return True

    async def retry_with_backoff(self, func, *args, **kwargs):
        """Retry mechanism with backoff"""
        for attempt in range(self.config.retry_config.max_retries):
            try:
                result = await func(*args, **kwargs)
                # Record successful retry
                if attempt > 0:
                    self.stats.retry_stats["successful_retries"] += 1
                return result
            except Exception as e:
                # Record retry count
                self.stats.retry_stats["total_retries"] += 1

                if attempt == self.config.retry_config.max_retries - 1:
                    # Record failed retry
                    self.stats.retry_stats["failed_retries"] += 1
                    raise e

                delay = self.config.retry_config.retry_delay * (
                    self.config.retry_config.backoff_factor**attempt
                )
                print(
                    f"Retry {attempt + 1}/{self.config.retry_config.max_retries}, waiting {delay} seconds..."
                )
                await asyncio.sleep(delay)

    def check_analysis_file_exists(self, task_id: str, analysis_type: str) -> bool:
        """Check if analysis file already exists"""
        safe_task_id = task_id.replace("/", "_")

        # Build possible paths based on analysis type
        if analysis_type == "attack":
            possible_paths = [
                os.path.join(
                    self.config.workspace_dir, f"{safe_task_id}_attack_analysis.json"
                ),
                os.path.join(
                    self.config.workspace_dir,
                    f"attack_{safe_task_id}",
                    f"{safe_task_id}_attack_analysis.json",
                ),
                os.path.join(
                    self.config.workspace_dir,
                    f"attack_{safe_task_id}",
                    "workspace",
                    f"{safe_task_id}_attack_analysis.json",
                ),
            ]
        elif analysis_type == "diagnose":
            possible_paths = [
                os.path.join(
                    self.config.workspace_dir, f"{safe_task_id}_diagnosis.json"
                ),
                os.path.join(
                    self.config.workspace_dir,
                    f"diagnose_{safe_task_id}",
                    f"{safe_task_id}_diagnosis.json",
                ),
                os.path.join(
                    self.config.workspace_dir,
                    f"diagnose_{safe_task_id}",
                    "workspace",
                    f"{safe_task_id}_diagnosis.json",
                ),
            ]
        else:
            return False

        for path in possible_paths:
            if os.path.exists(path):
                return True
        return False

    def load_existing_analysis(
        self, task_id: str, analysis_type: str
    ) -> Optional[Dict[str, Any]]:
        """Load existing analysis file"""
        safe_task_id = task_id.replace("/", "_")

        # Build possible paths based on analysis type
        if analysis_type == "attack":
            possible_paths = [
                os.path.join(
                    self.config.workspace_dir, f"{safe_task_id}_attack_analysis.json"
                ),
                os.path.join(
                    self.config.workspace_dir,
                    f"attack_{safe_task_id}",
                    f"{safe_task_id}_attack_analysis.json",
                ),
                os.path.join(
                    self.config.workspace_dir,
                    f"attack_{safe_task_id}",
                    "workspace",
                    f"{safe_task_id}_attack_analysis.json",
                ),
            ]
        elif analysis_type == "diagnose":
            possible_paths = [
                os.path.join(
                    self.config.workspace_dir, f"{safe_task_id}_diagnosis.json"
                ),
                os.path.join(
                    self.config.workspace_dir,
                    f"diagnose_{safe_task_id}",
                    f"{safe_task_id}_diagnosis.json",
                ),
                os.path.join(
                    self.config.workspace_dir,
                    f"diagnose_{safe_task_id}",
                    "workspace",
                    f"{safe_task_id}_diagnosis.json",
                ),
            ]
        else:
            return None

        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    print(f"Error loading existing analysis from {path}: {e}")
                    continue

        return None

    def record_retry_count(self, task_id: str, retry_count: int):
        """Record task retry count"""
        self.stats.retry_stats["retry_counts"][task_id] = retry_count

    def record_loaded_existing(self, task_id: str):
        """Record tasks loaded from files"""
        self.stats.retry_stats["loaded_existing"] += 1
        self.stats.retry_stats["retry_counts"][task_id] = 0

    def _print_retry_stats(self):
        """Print retry statistics"""
        if (
            self.stats.retry_stats["total_retries"] > 0
            or self.stats.retry_stats["loaded_existing"] > 0
        ):
            print(f"\n{'='*60}")
            print("📊 Retry Statistics")
            print(f"{'='*60}")
            print(f"Total retries: {self.stats.retry_stats['total_retries']}")
            print(f"Successful retries: {self.stats.retry_stats['successful_retries']}")
            print(f"Failed retries: {self.stats.retry_stats['failed_retries']}")
            print(f"Loaded from files: {self.stats.retry_stats['loaded_existing']}")

            if self.stats.retry_stats["retry_counts"]:
                retry_distribution = {}
                for task_id, count in self.stats.retry_stats["retry_counts"].items():
                    retry_distribution[count] = retry_distribution.get(count, 0) + 1

                print(f"\nRetry count distribution:")
                for retry_count, task_count in sorted(retry_distribution.items()):
                    if retry_count == 0:
                        print(f"  0 retries (loaded from files): {task_count} tasks")
                    else:
                        print(f"  {retry_count} retries: {task_count} tasks")

            success_rate = (
                (
                    self.stats.successful_tasks
                    + self.stats.retry_stats["loaded_existing"]
                )
                / self.stats.total_tasks
                * 100
                if self.stats.total_tasks > 0
                else 0
            )
            print(f"\nOverall success rate: {success_rate:.1f}%")

    def run_evalplus(self, round_num: int, eval_type: str = None):
        """Run evalplus evaluation"""
        print(f"\n{'='*60}")
        print(f"Running round {round_num} {eval_type or 'general'} evaluation")
        print(f"{'='*60}")

        if self.config.dataset_type == DatasetType.KODCODE:
            self._run_kodcode_eval_by_difficulty(round_num, eval_type)
        else:
            self._run_mbpp_plus_eval(round_num, eval_type)

    def _run_kodcode_eval_by_difficulty(self, round_num: int, eval_type: str = None):
        """Run KodCode evaluation by difficulty"""
        print("Running KodCode evaluation by difficulty...")

        # Get all difficulty levels' original evalplus result files
        original_files = self.config.get_all_original_eval_result_files()

        if not original_files:
            print("No evalplus result files found for any difficulty level")
            return

        # Run evaluation for each difficulty
        for difficulty in original_files.keys():
            print(f"\nProcessing difficulty: {difficulty}")

            # Get evaluation command for this difficulty
            eval_command = self.config.get_eval_command(
                round_num, eval_type, difficulty
            )
            print(f"Executing command: {eval_command}")

            try:
                import subprocess
                import sys

                # Execute evaluation command
                result = subprocess.run(
                    eval_command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.config.eval_config.timeout,
                )

                if result.returncode == 0:
                    print(f"✅ KodCode {difficulty} difficulty evaluation executed successfully")
                    if result.stdout:
                        print(f"Output: {result.stdout}")
                else:
                    print(f"❌ KodCode {difficulty} difficulty evaluation failed")
                    print(f"Error: {result.stderr}")
                    print(f"Return code: {result.returncode}")

            except subprocess.TimeoutExpired:
                print(
                    f"❌ KodCode {difficulty} difficulty evaluation timed out (>{self.config.eval_config.timeout} seconds)"
                )
            except Exception as e:
                print(f"❌ KodCode {difficulty} difficulty evaluation exception: {e}")

    def _run_kodcode_eval(self, round_num: int, eval_type: str = None):
        """Run KodCode evaluation (compatibility method)"""
        self._run_kodcode_eval_by_difficulty(round_num, eval_type)

    def _run_mbpp_plus_eval(self, round_num: int, eval_type: str = None):
        """Run MBPP+ evaluation - to be implemented"""
        print("⚠️  MBPP+ evaluation functionality not implemented yet")
        print("MBPP+ evaluation requires:")
        print("1. Process data according to prepare_for_evalplus logic")
        print("2. Activate evalplus conda environment")
        print("3. Execute evalplus.evaluate --dataset mbpp --samples path/to/***.jsonl")
        print("This functionality will be implemented in future versions")

    def process_eval_results(self, round_num: int, eval_type: str = None):
        """Process evaluation results"""
        print(f"\n{'='*60}")
        print(f"Processing round {round_num} {eval_type or 'general'} evaluation results")
        print(f"{'='*60}")

        print(f"🔍 Starting to call _merge_eval_results_by_difficulty method...")
        # Merge evaluation results from all difficulty levels
        eval_results = self._merge_eval_results_by_difficulty(round_num, eval_type)
        print(
            f"🔍 _merge_eval_results_by_difficulty method returned: {len(eval_results)} tasks"
        )

        if not eval_results:
            print(f"❌ No evaluation result files found")
            return False

        print(f"✅ Successfully merged evaluation results")
        print(f"Evaluation results contain {len(eval_results)} tasks")

        # Count successful and failed tasks
        success_count = sum(1 for result in eval_results.values() if result)
        fail_count = len(eval_results) - success_count

        print(f"Evaluation statistics:")
        print(f"  Success: {success_count} tasks")
        print(f"  Failed: {fail_count} tasks")

        # Avoid division by zero
        if len(eval_results) > 0:
            success_rate = success_count / len(eval_results) * 100
            print(f"  Success rate: {success_rate:.1f}%")
        else:
            print(f"  Success rate: 0.0% (no tasks)")

        # Generate task files needed for next round
        if eval_type == "attack":
            self._generate_attack_next_round_tasks(round_num, eval_results)
        elif eval_type == "diagnose":
            self._generate_diagnose_next_round_tasks(round_num, eval_results)

        return True

    def _merge_eval_results_by_difficulty(
        self, round_num: int, eval_type: str = None
    ) -> dict:
        """Merge evaluation results from all difficulty levels"""
        merged_results = {}
        task_id_conflicts = []  # Record task ID conflicts

        if self.config.dataset_type.value == "kodcode":
            # KodCode dataset: merge results from all difficulty levels
            difficulties = ["easy", "medium", "hard"]

            print(f"Starting to merge {eval_type} evaluation results, round {round_num}...")

            for difficulty in difficulties:
                eval_result_file = self.config.get_eval_result_file(
                    round_num, eval_type, difficulty
                )
                print(f"Checking file: {eval_result_file}")

                if os.path.exists(eval_result_file):
                    try:
                        with open(eval_result_file, "r", encoding="utf-8") as f:
                            difficulty_results = json.load(f)

                        print(
                            f"✅ Loaded {difficulty} difficulty results: {len(difficulty_results)} tasks"
                        )
                        if difficulty_results:
                            print(
                                f"  {difficulty} difficulty task IDs: {list(difficulty_results.keys())}"
                            )

                        # Check for task ID conflicts
                        for task_id in difficulty_results.keys():
                            if task_id in merged_results:
                                conflict_info = {
                                    "task_id": task_id,
                                    "existing_difficulty": merged_results.get(
                                        f"{task_id}_difficulty", "unknown"
                                    ),
                                    "new_difficulty": difficulty,
                                    "existing_result": merged_results[task_id],
                                    "new_result": difficulty_results[task_id],
                                }
                                task_id_conflicts.append(conflict_info)
                                print(
                                    f"⚠️  Found task ID conflict: {task_id} exists in both {conflict_info['existing_difficulty']} and {difficulty} difficulties"
                                )

                        # Merge results, add difficulty information for each task ID
                        for task_id, result in difficulty_results.items():
                            merged_results[task_id] = result
                            merged_results[f"{task_id}_difficulty"] = difficulty

                    except Exception as e:
                        print(f"❌ Failed to load {difficulty} difficulty results: {e}")
                else:
                    print(f"⚠️  {difficulty} difficulty result file does not exist: {eval_result_file}")

            # Report task ID conflicts
            if task_id_conflicts:
                print(f"\n⚠️  Found {len(task_id_conflicts)} task ID conflicts:")
                for conflict in task_id_conflicts:
                    print(
                        f"  Task {conflict['task_id']}: {conflict['existing_difficulty']} -> {conflict['new_difficulty']}"
                    )
                    print(
                        f"    Result change: {conflict['existing_result']} -> {conflict['new_result']}"
                    )
                print("  Note: Later difficulty results will override earlier difficulty results")

            print(
                f"Merge completed, total {len([k for k in merged_results.keys() if not k.endswith('_difficulty')])} tasks"
            )
            task_ids = [
                k for k in merged_results.keys() if not k.endswith("_difficulty")
            ]
            if task_ids:
                print(f"Merged task IDs: {task_ids}")
        else:
            # Other datasets: directly load single result file
            eval_result_file = self.config.get_eval_result_file(round_num, eval_type)

            if os.path.exists(eval_result_file):
                try:
                    with open(eval_result_file, "r", encoding="utf-8") as f:
                        merged_results = json.load(f)
                    print(f"✅ Loaded result file: {eval_result_file}")
                except Exception as e:
                    print(f"❌ Failed to load result file: {e}")
            else:
                print(f"❌ Result file does not exist: {eval_result_file}")

        return merged_results

    def _generate_attack_next_round_tasks(self, round_num: int, eval_results: dict):
        """Generate next round task files after attack"""
        print(f"\nGenerating next round tasks after round {round_num} attack...")

        # Get original task data
        original_tasks_file = os.path.join(
            self.config.output_base_dir, "success_tasks_with_logs.json"
        )
        if not os.path.exists(original_tasks_file):
            print(f"❌ Original task file does not exist: {original_tasks_file}")
            return

        try:
            with open(original_tasks_file, "r", encoding="utf-8") as f:
                original_tasks = json.load(f)

            # Categorize tasks: attack successful (eval=false) and attack failed (eval=true)
            attack_successful_tasks = []  # Attack successful, task failed, needs diagnosis
            attack_failed_tasks = []  # Attack failed, task still successful, needs continued attack

            for task in original_tasks:
                task_id = task["task_id"]
                if task_id in eval_results:
                    if eval_results[task_id]:  # eval=true, attack failed
                        attack_failed_tasks.append(task)
                    else:  # eval=false, attack successful
                        attack_successful_tasks.append(task)

            # Save attack failed tasks (need continued attack)
            if attack_failed_tasks:
                attack_failed_file = self.get_attack_failed_file(round_num)
                with open(attack_failed_file, "w", encoding="utf-8") as f:
                    json.dump(attack_failed_tasks, f, ensure_ascii=False, indent=2)
                print(
                    f"✅ Saved attack failed tasks: {len(attack_failed_tasks)} -> {attack_failed_file}"
                )

            # Save attack successful tasks (need diagnosis)
            if attack_successful_tasks:
                attack_successful_file = self.get_attack_successful_file(round_num)
                with open(attack_successful_file, "w", encoding="utf-8") as f:
                    json.dump(attack_successful_tasks, f, ensure_ascii=False, indent=2)
                print(
                    f"✅ Saved attack successful tasks: {len(attack_successful_tasks)} -> {attack_successful_file}"
                )

        except Exception as e:
            print(f"❌ Failed to generate attack next round tasks: {e}")

    def _generate_diagnose_next_round_tasks(self, round_num: int, eval_results: dict):
        """Generate next round task files after diagnosis"""
        print(f"\nGenerating next round tasks after round {round_num} diagnosis...")

        # Get original task data
        original_tasks_file = os.path.join(
            self.config.output_base_dir, "failed_tasks_with_logs.json"
        )
        if not os.path.exists(original_tasks_file):
            print(f"❌ Original task file does not exist: {original_tasks_file}")
            return

        try:
            with open(original_tasks_file, "r", encoding="utf-8") as f:
                original_tasks = json.load(f)

            # Categorize tasks: diagnosis successful (eval=true) and diagnosis failed (eval=false)
            diagnose_successful_tasks = []  # Diagnosis successful, task successful, no need to continue processing
            diagnose_failed_tasks = []  # Diagnosis failed, task still failed, needs continued diagnosis

            for task in original_tasks:
                task_id = task["task_id"]
                if task_id in eval_results:
                    if eval_results[task_id]:  # eval=true, diagnosis successful
                        diagnose_successful_tasks.append(task)
                    else:  # eval=false, diagnosis failed
                        diagnose_failed_tasks.append(task)

            # Save diagnosis failed tasks (need continued diagnosis)
            if diagnose_failed_tasks:
                diagnose_failed_file = self.get_diagnose_failed_file(round_num)
                with open(diagnose_failed_file, "w", encoding="utf-8") as f:
                    json.dump(diagnose_failed_tasks, f, ensure_ascii=False, indent=2)
                print(
                    f"✅ Saved diagnosis failed tasks: {len(diagnose_failed_tasks)} -> {diagnose_failed_file}"
                )

            # Save diagnosis successful tasks (completed)
            if diagnose_successful_tasks:
                diagnose_successful_file = self.get_diagnose_successful_file(round_num)
                with open(diagnose_successful_file, "w", encoding="utf-8") as f:
                    json.dump(
                        diagnose_successful_tasks, f, ensure_ascii=False, indent=2
                    )
                print(
                    f"✅ Saved diagnosis successful tasks: {len(diagnose_successful_tasks)} -> {diagnose_successful_file}"
                )

        except Exception as e:
            print(f"❌ Failed to generate diagnosis next round tasks: {e}")

    def get_attack_failed_file(self, round_num: int) -> str:
        """Get attack failed task file path"""
        return self.directories["attacked_still_succeed"].format(round_num)

    def get_attack_successful_file(self, round_num: int) -> str:
        """Get attack successful task file path (needs diagnosis)"""
        return os.path.join(
            self.config.output_base_dir,
            f"attack_successful_tasks_round_{round_num}.json",
        )

    def get_diagnose_failed_file(self, round_num: int) -> str:
        """Get diagnosis failed task file path"""
        return self.directories["replayed_still_failed"].format(round_num)

    def get_diagnose_successful_file(self, round_num: int) -> str:
        """Get diagnosis successful task file path"""
        return os.path.join(
            self.config.output_base_dir,
            f"diagnose_successful_tasks_round_{round_num}.json",
        )

    def get_task_difficulty(self, task_id: str) -> str:
        """Get task difficulty"""
        # Default return hard, can be extended as needed
        return "hard"

    def _load_previous_attack_analyses(
        self, task_id: str, current_round: int
    ) -> List[Dict[str, Any]]:
        """Load historical attack analysis information"""
        analyses = []

        # Iterate through all previous rounds
        for round_num in range(1, current_round):
            # Build attack suggestion file path for this round
            attack_suggestions_dir = self.directories["attack_suggestions"].format(
                round_num
            )
            previous_path = os.path.join(
                attack_suggestions_dir, f"{task_id.replace('/', '_')}.json"
            )

            print(f"[HISTORY] Attempting to load round {round_num} attack analysis: {previous_path}")

            if os.path.exists(previous_path):
                try:
                    with open(previous_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        attack_analysis = data.get("attack_analysis")
                        if attack_analysis:
                            # Add round information
                            attack_analysis["round"] = round_num
                            analyses.append(attack_analysis)
                            print(
                                f"[HISTORY] Successfully loaded round {round_num} attack analysis: step {attack_analysis.get('attack_step', 'Unknown')}"
                            )
                        else:
                            print(
                                f"[HISTORY] Round {round_num} file exists but missing attack_analysis field"
                            )
                except Exception as e:
                    print(
                        f"[HISTORY] Error loading round {round_num} attack analysis from {previous_path}: {e}"
                    )
            else:
                print(f"[HISTORY] Round {round_num} attack analysis file does not exist: {previous_path}")

        print(f"[HISTORY] Total loaded {len(analyses)} rounds of historical attack analysis")
        return analyses

    def _load_previous_diagnosis_analyses(
        self, task_id: str, current_round: int
    ) -> List[Dict[str, Any]]:
        """Load historical diagnosis analysis information"""
        analyses = []

        # Iterate through all previous rounds
        for round_num in range(1, current_round):
            # Build diagnosis log file path for this round
            diagnosed_logs_dir = self.directories["diagnosed_logs"].format(round_num)
            previous_path = os.path.join(
                diagnosed_logs_dir, f"{task_id.replace('/', '_')}.json"
            )

            print(f"[HISTORY] Attempting to load round {round_num} diagnosis analysis: {previous_path}")

            if os.path.exists(previous_path):
                try:
                    with open(previous_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        diagnosis = data.get("diagnosis")
                        if diagnosis:
                            # Add round information
                            diagnosis["round"] = round_num
                            analyses.append(diagnosis)
                            print(
                                f"[HISTORY] Successfully loaded round {round_num} diagnosis analysis: step {diagnosis.get('mistake_step', 'Unknown')}"
                            )
                        else:
                            print(
                                f"[HISTORY] Round {round_num} file exists but missing diagnosis field"
                            )
                except Exception as e:
                    print(
                        f"[HISTORY] Error loading round {round_num} diagnosis from {previous_path}: {e}"
                    )
            else:
                print(f"[HISTORY] Round {round_num} diagnosis analysis file does not exist: {previous_path}")

        print(f"[HISTORY] Total loaded {len(analyses)} rounds of historical diagnosis analysis")
        return analyses

    def _validate_history_data(
        self, history_data: List[Dict[str, Any]], data_type: str
    ) -> bool:
        """Validate historical data format"""
        if not isinstance(history_data, list):
            print(f"[HISTORY] Historical {data_type} data format error: not a list")
            return False

        for i, item in enumerate(history_data):
            if not isinstance(item, dict):
                print(f"[HISTORY] Historical {data_type} data item {i} format error: not a dictionary")
                return False

            if "round" not in item:
                print(f"[HISTORY] Historical {data_type} data item {i} missing round field")
                return False

        return True


# Add processor type imports at the end of the file to avoid circular imports
def _get_processor_types():
    """Get processor types, avoid circular imports"""
    try:
        from attack_processor import AttackProcessor
        from diagnose_processor import DiagnoseProcessor

        return AttackProcessor, DiagnoseProcessor
    except ImportError:
        # If import fails, return None
        return None, None


# Use lazy import in is_task_completed method
def _check_processor_type(processor, processor_type_name):
    """Check processor type"""
    try:
        AttackProcessor, DiagnoseProcessor = _get_processor_types()
        if processor_type_name == "AttackProcessor":
            return isinstance(processor, AttackProcessor) if AttackProcessor else False
        elif processor_type_name == "DiagnoseProcessor":
            return (
                isinstance(processor, DiagnoseProcessor) if DiagnoseProcessor else False
            )
    except:
        pass
    return False
