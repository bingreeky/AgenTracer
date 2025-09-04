#!/usr/bin/env python3
"""
Optimized Universal Attack and Diagnosis Framework - Supporting Multi-round Attacks and Diagnosis
"""

import os
import json
import asyncio
import argparse
from typing import Dict, Any, Optional, List
from datetime import datetime

from config import (
    FrameworkConfig,
    DatasetType,
    create_kodcode_config,
    create_mbpp_plus_config,
)
from base_processor import ProcessingStats
from attack_processor import AttackProcessor
from diagnose_processor import DiagnoseProcessor
from retry_manager import get_retry_manager, reset_retry_manager


def standardize_role_name(name: str) -> str:
    """Standardize role names, ensure using correct role names instead of personal names"""
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


class UniversalFramework:
    """Optimized Universal Attack and Diagnosis Framework"""

    def __init__(self, config: FrameworkConfig):
        self.config = config
        self.config.validate()

        # Initialize retry manager
        self.retry_manager = get_retry_manager(
            max_retries=config.retry_config.max_retries,
            retry_delay=config.retry_config.retry_delay,
            backoff_factor=config.retry_config.backoff_factor,
        )

        # Initialize processors
        self.attack_processor = AttackProcessor(config)
        self.diagnose_processor = DiagnoseProcessor(config)

        # Statistics
        self.round_stats = {}

        # Task completion status tracking
        self.completed_tasks = set()  # Store completed task IDs
        self.completed_tasks_file = os.path.join(
            self.config.output_base_dir, "completed_tasks.json"
        )
        self._load_completed_tasks()

    def extract_tasks_from_eval_results(self, round_num: int = 1):
        """Automatically extract and categorize tasks from evalplus results"""
        print(f"\n{'='*60}")
        print(f"Extracting round {round_num} tasks from evalplus results")
        print(f"{'='*60}")

        # Process based on dataset type
        if self.config.dataset_type == DatasetType.KODCODE:
            return self._extract_kodcode_tasks_by_difficulty(round_num)
        else:
            return self._extract_generic_tasks(round_num)

    def _extract_kodcode_tasks_by_difficulty(self, round_num: int):
        """Extract KodCode tasks by difficulty"""
        print("Extracting KodCode tasks by difficulty...")

        # Get all difficulty levels' original evalplus result files
        original_files = self.config.get_all_original_eval_result_files()

        if not original_files:
            print("No evalplus result files found for any difficulty level")
            return False

        all_success_tasks = []
        all_failed_tasks = []

        # Process tasks for each difficulty
        for difficulty, eval_result_file in original_files.items():
            print(f"\nProcessing difficulty: {difficulty}")
            print(f"Result file: {eval_result_file}")

            # Read evalplus results for this difficulty
            with open(eval_result_file, "r", encoding="utf-8") as f:
                eval_results = json.load(f)

            # Classify tasks for this difficulty
            success_tasks, failed_tasks = self._classify_kodcode_tasks_by_difficulty(
                eval_results, difficulty, round_num
            )

            all_success_tasks.extend(success_tasks)
            all_failed_tasks.extend(failed_tasks)

            print(
                f"difficulty {difficulty}: SuccessTask {len(success_tasks)} items, failedTask {len(failed_tasks)} items"
            )

        # Save all tasks
        self._save_extracted_tasks(all_success_tasks, all_failed_tasks)

        return True

    def _extract_generic_tasks(self, round_num: int):
        """Extract generic dataset tasks (non-KodCode)"""
        # Get evalplus result file path
        if round_num == 1:
            # Round one uses original evalplus results
            eval_result_file = self.config.get_original_eval_result_file()
        else:
            # Subsequent rounds use results from previous round
            # Try to read attack and diagnose result files
            attack_result_file = self.config.get_eval_result_file(
                round_num - 1, "attack"
            )
            diagnose_result_file = self.config.get_eval_result_file(
                round_num - 1, "diagnose"
            )

            # Check which file exists
            if os.path.exists(attack_result_file):
                eval_result_file = attack_result_file
                print(f"Using attack result file: {eval_result_file}")
            elif os.path.exists(diagnose_result_file):
                eval_result_file = diagnose_result_file
                print(f"Using diagnose result file: {eval_result_file}")
            else:
                # Try to use default filename
                eval_result_file = self.config.get_eval_result_file(round_num - 1)
                print(f"Using default result file: {eval_result_file}")

        if not os.path.exists(eval_result_file):
            print(f"Evalplus result file does not exist: {eval_result_file}")
            return False

        # Read evalplus results
        with open(eval_result_file, "r", encoding="utf-8") as f:
            eval_results = json.load(f)

        # Classify tasks
        success_tasks = []
        failed_tasks = []

        # Process based on dataset type different result formats
        if self.config.dataset_type == DatasetType.KODCODE:
            # KodCode format: {"task_id": true/false, ...}
            for task_id, is_success in eval_results.items():
                # Get corresponding log data
                log_data = self._get_log_data(task_id)

                # For attack tasks: If attack succeeds (eval result is false), no need to continue processing
                # For diagnose tasks: If diagnose succeeds (eval result is true), no need to continue processing
                # Here we need to judge based on current round and task type

                # Check if it's an attack result file
                is_attack_result = "attack" in eval_result_file
                is_diagnose_result = "diagnose" in eval_result_file

                if is_attack_result:
                    # Attack result: If attack succeeds (eval result is false), task failed, needs diagnosis
                    if not is_success:  # Attack succeeded, task failed
                        # Check if task is completed
                        if self.is_task_completed(task_id):
                            print(f"Task {task_id} completed, skipped")
                            continue

                        # Ensure log_data has task_id field
                        if "task_id" not in log_data:
                            log_data["task_id"] = task_id

                        failed_tasks.append(
                            {
                                "task_id": task_id,
                                "failure_info": {
                                    "status": "fail",
                                    "attack_successful": True,
                                },
                                "log_data": log_data,
                            }
                        )
                    else:  # Attack failed, task still succeeds, needs to continue attack
                        # Check if task is completed
                        if self.is_task_completed(task_id):
                            print(f"Task {task_id} completed, skipped")
                            continue

                        success_tasks.append(
                            {
                                "task_id": task_id,
                                "success_info": {
                                    "status": "pass",
                                    "attack_failed": True,
                                },
                                "log_data": log_data,
                            }
                        )

                elif is_diagnose_result:
                    # Diagnose result: If diagnose succeeds (eval result is true), task succeeds, no need to continue processing
                    # If diagnose fails (eval result is false), task still fails, needs to continue diagnosis
                    if is_success:  # Diagnose succeeded, task succeeds
                        # Check if task is completed
                        if self.is_task_completed(task_id):
                            print(f"Task {task_id} completed, skipped")
                            continue

                        # Diagnose succeeded, no need to continue processing
                        pass
                    else:  # Diagnose failed, task still fails, needs to continue diagnosis
                        # Check if task is completed
                        if self.is_task_completed(task_id):
                            print(f"Task {task_id} completed, skipped")
                            continue

                        # Ensure log_data has task_id field
                        if "task_id" not in log_data:
                            log_data["task_id"] = task_id

                        failed_tasks.append(
                            {
                                "task_id": task_id,
                                "failure_info": {
                                    "status": "fail",
                                    "diagnose_failed": True,
                                },
                                "log_data": log_data,
                            }
                        )

                else:
                    # Original result: Classify based on success/fail status
                    if is_success:
                        # Check if task is completed
                        if self.is_task_completed(task_id):
                            print(f"Task {task_id} completed, skipped")
                            continue

                        success_tasks.append(
                            {
                                "task_id": task_id,
                                "success_info": {
                                    "status": "pass",
                                },
                                "log_data": log_data,
                            }
                        )
                    else:
                        # Check if task is completed
                        if self.is_task_completed(task_id):
                            print(f"Task {task_id} completed, skipped")
                            continue

                        # Ensure log_data has task_id field
                        if "task_id" not in log_data:
                            log_data["task_id"] = task_id

                        failed_tasks.append(
                            {
                                "task_id": task_id,
                                "failure_info": {
                                    "status": "fail",
                                },
                                "log_data": log_data,
                            }
                        )
        else:
            # MBPP+ format: {"eval": {"task_id": [{"base_status": "pass/fail", "plus_status": "pass/fail"}]}}
            eval_data = eval_results.get("eval", {})
            for task_key, task_results in eval_data.items():
                if not task_results:
                    continue

                task_result = task_results[0]  # Take first result
                base_status = task_result.get("base_status", "fail")
                plus_status = task_result.get("plus_status", "fail")

                # Extract TaskID
                task_id = task_key.replace("/", "_")

                # MBPP+ success judgment logic
                is_success = base_status == "pass" and plus_status == "pass"

                # Get corresponding log data
                log_data = self._get_log_data(task_id)

                if is_success:
                    # Check if task is completed
                    if self.is_task_completed(task_id):
                        print(f"Task {task_id} completed, skipped")
                        continue

                    success_tasks.append(
                        {
                            "task_id": task_id,
                            "success_info": {
                                "base_status": base_status,
                                "plus_status": plus_status,
                            },
                            "log_data": log_data,
                        }
                    )
                else:
                    # Check if task is completed
                    if self.is_task_completed(task_id):
                        print(f"Task {task_id} completed, skipped")
                        continue

                    # Ensure log_data has task_id field
                    if "task_id" not in log_data:
                        log_data["task_id"] = task_id

                    failed_tasks.append(
                        {
                            "task_id": task_id,
                            "failure_info": {
                                "base_status": base_status,
                                "plus_status": plus_status,
                            },
                            "log_data": log_data,
                        }
                    )

        # Save classification results
        success_file = os.path.join(
            self.config.output_base_dir, "success_tasks_with_logs.json"
        )
        failed_file = os.path.join(
            self.config.output_base_dir, "failed_tasks_with_logs.json"
        )

        with open(success_file, "w", encoding="utf-8") as f:
            json.dump(success_tasks, f, ensure_ascii=False, indent=2)

        with open(failed_file, "w", encoding="utf-8") as f:
            json.dump(failed_tasks, f, ensure_ascii=False, indent=2)

        print(f"Successfully extracted tasks:")
        print(f"  Success tasks: {len(success_tasks)} items")
        print(f"  Failed tasks: {len(failed_tasks)} items")
        print(f"  Success tasks file: {success_file}")
        print(f"  Failed tasks file: {failed_file}")

        return True

    def _classify_kodcode_tasks_by_difficulty(
        self, eval_results: dict, difficulty: str, round_num: int
    ):
        """Classify KodCode tasks by difficulty"""
        success_tasks = []
        failed_tasks = []

        for task_id, is_success in eval_results.items():
            # Get corresponding log data
            log_data = self._get_log_data(task_id)

            # Add difficulty information to log_data
            log_data["difficulty"] = difficulty

            if is_success:
                # Check if task is completed
                if self.is_task_completed(task_id):
                    print(f"Task {task_id} completed, skipped")
                    continue

                success_tasks.append(
                    {
                        "task_id": task_id,
                        "difficulty": difficulty,
                        "success_info": {
                            "status": "pass",
                        },
                        "log_data": log_data,
                    }
                )
            else:
                # Check if task is completed
                if self.is_task_completed(task_id):
                    print(f"Task {task_id} completed, skipped")
                    continue

                # Ensure log_data has task_id field
                if "task_id" not in log_data:
                    log_data["task_id"] = task_id

                failed_tasks.append(
                    {
                        "task_id": task_id,
                        "difficulty": difficulty,
                        "failure_info": {
                            "status": "fail",
                        },
                        "log_data": log_data,
                    }
                )

        return success_tasks, failed_tasks

    def _save_extracted_tasks(self, success_tasks: list, failed_tasks: list):
        """Save extracted tasks"""
        # Save success tasks
        success_file = os.path.join(
            self.config.output_base_dir, "success_tasks_with_logs.json"
        )
        with open(success_file, "w", encoding="utf-8") as f:
            json.dump(success_tasks, f, ensure_ascii=False, indent=2)

        # Save failed tasks
        failed_file = os.path.join(
            self.config.output_base_dir, "failed_tasks_with_logs.json"
        )
        with open(failed_file, "w", encoding="utf-8") as f:
            json.dump(failed_tasks, f, ensure_ascii=False, indent=2)

        print(f"Successfully extracted tasks:")
        print(f"  Success tasks: {len(success_tasks)} items")
        print(f"  Failed tasks: {len(failed_tasks)} items")
        print(f"  Success tasks file: {success_file}")
        print(f"  Failed tasks file: {failed_file}")

    def _get_log_data(self, task_id: str) -> Dict[str, Any]:
        """Get task log data"""
        # Build log file path based on dataset type
        logs_dir = self.config.get_logs_dir()
        if self.config.dataset_type == DatasetType.KODCODE:
            # KodCode log file path
            log_file = os.path.join(logs_dir, f"{task_id}.json")
        else:
            # MBPP+ log file format
            task_num = task_id.replace("Mbpp_", "")
            log_file = os.path.join(logs_dir, f"mbpp_Mbpp_{task_num}.json")

        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    log_data = json.load(f)

                    # Ensure KodCode log data contains all necessary fields
                    if self.config.dataset_type == DatasetType.KODCODE:
                        # Ensure required fields exist
                        required_fields = [
                            "question",
                            "question_ID",
                            "ground_truth",
                            "model_prediction",
                        ]
                        for field in required_fields:
                            if field not in log_data:
                                print(
                                    f"Warning: KodCode log file {log_file} missing field: {field}"
                                )
                                # Provide default values for missing fields
                                if field == "question":
                                    log_data[field] = "No question provided"
                                elif field == "ground_truth":
                                    log_data[field] = "No ground truth provided"
                                elif field == "model_prediction":
                                    log_data[field] = "No model prediction provided"

                        # Ensure history field exists
                        if "history" not in log_data:
                            log_data["history"] = []

                        # Ensure system_prompts field exists
                        if "system_prompts" not in log_data:
                            log_data["system_prompts"] = {}

                        # Standardize role names in history
                        for step in log_data["history"]:
                            if "name" in step:
                                step["name"] = standardize_role_name(step["name"])

                        # Extract all agents that appear in history (after standardization), ensure system_prompts contains all agents
                        agents_in_history = set()
                        for step in log_data["history"]:
                            if "name" in step:
                                agents_in_history.add(step["name"])

                        # Add default system_prompt for missing agents
                        for agent in agents_in_history:
                            if agent not in log_data["system_prompts"]:
                                # Provide more appropriate default system prompt based on agent type
                                if agent == "Product Manager":
                                    default_prompt = "You are a product manager responsible for creating product requirement documents and defining product features."
                                elif agent == "Architect":
                                    default_prompt = "You are a software architect responsible for designing system architecture and technical specifications."
                                elif agent == "Engineer":
                                    default_prompt = "You are a world-class engineer responsible for implementing code and technical solutions."
                                elif agent == "Team Leader":
                                    default_prompt = "You are a team leader responsible for coordinating team members and managing project tasks."
                                elif agent == "Data Analyst":
                                    default_prompt = "You are a data analyst responsible for data analysis and insights generation."
                                else:
                                    default_prompt = (
                                        f"Default system prompt for {agent}"
                                    )

                                log_data["system_prompts"][agent] = default_prompt
                                print(
                                    f"  ⚠️  Added default system prompt for missing agent '{agent}'"
                                )

                    return log_data
            except Exception as e:
                print(f"Failed to read log file {log_file}: {e}")

        return {}

    def _get_task_status_for_round(self, round_num: int):
        """Get task status based on round number"""
        if round_num == 1:
            # Round 1: Load original classification results for checking
            success_tasks_file = os.path.join(
                self.config.output_base_dir, "success_tasks_with_logs.json"
            )
            failed_tasks_file = os.path.join(
                self.config.output_base_dir, "failed_tasks_with_logs.json"
            )

            has_success_tasks = (
                os.path.exists(success_tasks_file)
                and os.path.getsize(success_tasks_file) > 2
            )
            has_failed_tasks = (
                os.path.exists(failed_tasks_file)
                and os.path.getsize(failed_tasks_file) > 2
            )

            return has_success_tasks, has_failed_tasks
        else:
            # Round 2 and later: Load eval results from previous round for checking
            attack_failed_file = self.attack_processor.get_attack_failed_file(
                round_num - 1
            )
            diagnose_failed_file = self.diagnose_processor.get_diagnose_failed_file(
                round_num - 1
            )

            has_attack_failed_tasks = (
                os.path.exists(attack_failed_file)
                and os.path.getsize(attack_failed_file) > 2
            )
            has_diagnose_failed_tasks = (
                os.path.exists(diagnose_failed_file)
                and os.path.getsize(diagnose_failed_file) > 2
            )

            return has_attack_failed_tasks, has_diagnose_failed_tasks

    async def run_full_pipeline(
        self,
        max_rounds: Optional[int] = None,
        max_tasks_per_round: Optional[int] = None,
    ):
        """Run complete attack and diagnosis pipeline"""
        print(f"\n{'='*80}")
        print(f"Starting universal attack and diagnosis framework")
        print(f"Dataset: {self.config.dataset_type.value}")
        print(f"Working directory: {self.config.workspace_dir}")
        print(f"Output directory: {self.config.output_base_dir}")
        print(f"{'='*80}")

        # Determine maximum number of rounds
        if max_rounds is None:
            max_rounds = self.config.max_rounds

        # Determine maximum number of tasks per round
        if max_tasks_per_round is None:
            max_tasks_per_round = self.config.max_tasks_per_round

        print(f"Configuration information:")
        print(f"  Maximum rounds: {max_rounds}")
        print(f"  Maximum tasks per round: {max_tasks_per_round}")
        print(f"  Enable concurrency: {self.config.enable_concurrent}")
        print(f"  Concurrency limit: {self.config.concurrent_limit}")

        start_time = datetime.now()

        for round_num in range(1, max_rounds + 1):
            print(f"\n{'='*80}")
            print(f"Starting {round_num} round processing")
            print(f"{'='*80}")

            round_start_time = datetime.now()

            # 0. Round 1: Load evalplus results to extract tasks
            if round_num == 1:
                if not self.extract_tasks_from_eval_results(round_num):
                    print("Unable to load evalplus results to extract tasks, skipping this round")
                    continue

            # Check task status and decide processing strategy
            has_success_tasks, has_failed_tasks = self._get_task_status_for_round(
                round_num
            )

            if round_num == 1:
                print(f"Task status check (Round 1):")
                print(f"  Success tasks file exists: {has_success_tasks}")
                print(f"  Failed tasks file exists: {has_failed_tasks}")
            else:
                print(f"Task status check (Round {round_num}):")
                print(f"  Attack failed tasks file exists: {has_success_tasks}")
                print(f"  Diagnose failed tasks file exists: {has_failed_tasks}")

            # Calculate task allocation for this round
            remaining_tasks = (
                max_tasks_per_round if max_tasks_per_round else float("inf")
            )
            attack_tasks_limit = None
            diagnose_tasks_limit = None

            if max_tasks_per_round:
                # If there are task limits, allocate proportionally
                if has_success_tasks and has_failed_tasks:
                    # Both have tasks, allocate evenly
                    attack_tasks_limit = max_tasks_per_round // 2
                    diagnose_tasks_limit = max_tasks_per_round - attack_tasks_limit
                elif has_success_tasks:
                    # Only attack failed tasks
                    attack_tasks_limit = max_tasks_per_round
                elif has_failed_tasks:
                    # Only diagnose failed tasks
                    diagnose_tasks_limit = max_tasks_per_round

                print(f"Task allocation:")
                print(f"  Attack task limit: {attack_tasks_limit}")
                print(f"  Diagnose task limit: {diagnose_tasks_limit}")

            # 1. Attack round
            if has_success_tasks:
                if round_num == 1:
                    print(f"\n--- Round {round_num} attack (processing original success tasks) ---")
                else:
                    print(f"\n--- Round {round_num} attack (processing previous round attack failed tasks) ---")

                attack_stats = await self.attack_processor.process_round(
                    round_num, attack_tasks_limit
                )

                # 2. Run evalplus evaluation on attack results
                if attack_stats.successful_tasks > 0:
                    self.attack_processor.run_evalplus(round_num)

                    # 3. Process attack results and mark successful tasks as completed
                    self.attack_processor.process_attack_results(round_num)

                    # 4. Check attack results, mark successful tasks as completed
                    self._process_attack_completion(round_num)
            else:
                print(f"\n--- Round {round_num} attack ---")
                print("No tasks need attack, skipping attack round")
                attack_stats = ProcessingStats()

            # 4. Diagnose round
            if has_failed_tasks:
                if round_num == 1:
                    print(f"\n--- Round {round_num} diagnosis (processing original failed tasks) ---")
                else:
                    print(f"\n--- Round {round_num} diagnosis (processing previous round diagnosis failed tasks) ---")

                diagnose_stats = await self.diagnose_processor.process_round(
                    round_num, diagnose_tasks_limit
                )

                # 5. Run evalplus evaluation on diagnosis results
                if diagnose_stats.successful_tasks > 0:
                    self.diagnose_processor.run_evalplus(round_num)

                    # 6. Process diagnosis results and mark successful tasks as completed
                    self.diagnose_processor.process_diagnose_results(round_num)

                    # 7. Check diagnosis results, mark successful tasks as completed
                    self._process_diagnose_completion(round_num)
            else:
                print(f"\n--- Round {round_num} diagnosis ---")
                print("No tasks need diagnosis, skipping diagnosis round")
                diagnose_stats = ProcessingStats()

            # 7. Prepare tasks for next round (if not the last round)
            if round_num < max_rounds:
                self._prepare_next_round_tasks(round_num)

            # Record round statistics
            round_end_time = datetime.now()
            round_duration = (round_end_time - round_start_time).total_seconds()

            self.round_stats[round_num] = {
                "attack": attack_stats.to_dict(),
                "diagnose": diagnose_stats.to_dict(),
                "duration": round_duration,
                "start_time": round_start_time.isoformat(),
                "end_time": round_end_time.isoformat(),
            }

            print(f"\nRound {round_num} processing completed")
            total_tasks_processed = (
                attack_stats.total_tasks + diagnose_stats.total_tasks
            )
            total_successful = (
                attack_stats.successful_tasks + diagnose_stats.successful_tasks
            )
            print(
                f"Attack: Success {attack_stats.successful_tasks}/{attack_stats.total_tasks}"
            )
            print(
                f"Diagnosis: Success {diagnose_stats.successful_tasks}/{diagnose_stats.total_tasks}"
            )
            print(f"Total: Success {total_successful}/{total_tasks_processed}")
            if max_tasks_per_round:
                print(
                    f"Task limit: {max_tasks_per_round} (attack: {attack_tasks_limit}, diagnosis: {diagnose_tasks_limit})"
                )
            print(f"Round duration: {round_duration:.2f} seconds")

            # Check if there are still tasks to process
            if attack_stats.total_tasks == 0 and diagnose_stats.total_tasks == 0:
                print(f"Round {round_num} no tasks to process, ending early")
                break

        # Save overall statistics
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        self.save_overall_stats(start_time, end_time, total_duration)

        print(f"\n{'='*80}")
        print(f"Framework execution completed")
        print(f"Total duration: {total_duration:.2f} seconds")
        print(f"Processed rounds: {len(self.round_stats)}")
        print(f"{'='*80}")

    def _prepare_next_round_tasks(self, current_round: int):
        """Prepare tasks for next round"""
        print(f"\n{'='*60}")
        print(f"Preparing tasks for Round {current_round + 1}")
        print(f"{'='*60}")

        # Check attack and diagnosis result files from previous round
        attack_failed_file = self.attack_processor.get_attack_failed_file(current_round)
        diagnose_failed_file = self.diagnose_processor.get_diagnose_failed_file(
            current_round
        )

        print(f"Check attack failed tasks file: {attack_failed_file}")
        print(f"Check diagnosis failed tasks file: {diagnose_failed_file}")

        # Check if files exist
        attack_tasks_exist = (
            os.path.exists(attack_failed_file)
            and os.path.getsize(attack_failed_file) > 2
        )
        diagnose_tasks_exist = (
            os.path.exists(diagnose_failed_file)
            and os.path.getsize(diagnose_failed_file) > 2
        )

        print(f"Attack failed tasks file exists: {attack_tasks_exist}")
        print(f"Diagnosis failed tasks file exists: {diagnose_tasks_exist}")

        if not attack_tasks_exist and not diagnose_tasks_exist:
            print("⚠️  No tasks need to enter next round processing")
        else:
            print("✅  Next round tasks preparation completed")

    async def run_attack_only(
        self, round_num: int = 1, max_tasks: Optional[int] = None
    ):
        """Run attack process only"""
        print(f"\n{'='*60}")
        print(f"Running Round {round_num} attack")
        print(f"{'='*60}")

        stats = await self.attack_processor.process_round(round_num, max_tasks)

        if stats.successful_tasks > 0:
            self.attack_processor.run_evalplus(round_num)
            self.attack_processor.process_attack_results(round_num)

        return stats

    async def run_diagnose_only(
        self, round_num: int = 1, max_tasks: Optional[int] = None
    ):
        """Run diagnosis process only"""
        print(f"\n{'='*60}")
        print(f"Running Round {round_num} diagnosis")
        print(f"{'='*60}")

        stats = await self.diagnose_processor.process_round(round_num, max_tasks)

        if stats.successful_tasks > 0:
            self.diagnose_processor.run_evalplus(round_num)
            self.diagnose_processor.process_diagnose_results(round_num)

        return stats

    def save_overall_stats(
        self, start_time: datetime, end_time: datetime, total_duration: float
    ):
        """Save overall statistics"""
        stats_dir = os.path.join(self.config.output_base_dir, "stats")
        os.makedirs(stats_dir, exist_ok=True)

        overall_stats = {
            "framework_info": {
                "dataset_type": self.config.dataset_type.value,
                "workspace_dir": self.config.workspace_dir,
                "output_base_dir": self.config.output_base_dir,
                "max_rounds": self.config.max_rounds,
                "enable_concurrent": self.config.enable_concurrent,
            },
            "execution_info": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_duration": total_duration,
                "total_rounds": len(self.round_stats),
            },
            "round_stats": self.round_stats,
        }

        stats_file = os.path.join(stats_dir, "overall_stats.json")
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(overall_stats, f, ensure_ascii=False, indent=2)

        print(f"Overall statistics saved to: {stats_file}")

    def get_status(self) -> Dict[str, Any]:
        """Get framework status"""
        return {
            "config": self.config.to_dict(),
            "round_stats": self.round_stats,
            "attack_processor_stats": {
                "total_rounds": len(
                    [
                        k
                        for k in self.round_stats.keys()
                        if "attack" in self.round_stats[k]
                    ]
                ),
            },
            "diagnose_processor_stats": {
                "total_rounds": len(
                    [
                        k
                        for k in self.round_stats.keys()
                        if "diagnose" in self.round_stats[k]
                    ]
                ),
            },
        }

    def print_status(self):
        """Print framework status"""
        status = self.get_status()

        print(f"\n{'='*60}")
        print(f"Framework status")
        print(f"{'='*60}")
        print(f"Dataset: {status['config']['dataset_type']}")
        print(f"Working directory: {status['config']['workspace_dir']}")
        print(f"Output directory: {status['config']['output_base_dir']}")
        print(f"Maximum rounds: {status['config']['max_rounds']}")
        print(f"Enable concurrency: {status['config']['enable_concurrent']}")
        print(f"Processed rounds: {len(status['round_stats'])}")

        if status["round_stats"]:
            print(f"\nRound details:")
            for round_num, stats in status["round_stats"].items():
                attack_stats = stats["attack"]
                diagnose_stats = stats["diagnose"]
                print(f"  Round {round_num}:")
                print(
                    f"    Attack: {attack_stats['successful_tasks']}/{attack_stats['total_tasks']} Success"
                )
                print(
                    f"    Diagnosis: {diagnose_stats['successful_tasks']}/{diagnose_stats['total_tasks']} Success"
                )
                print(f"    Duration: {stats['duration']:.2f} seconds")

    def _load_completed_tasks(self):
        """Load completed task IDs from file"""
        if os.path.exists(self.completed_tasks_file):
            try:
                with open(self.completed_tasks_file, "r", encoding="utf-8") as f:
                    self.completed_tasks = set(json.load(f))
                print(f"Loaded {len(self.completed_tasks)} completed tasks")
            except json.JSONDecodeError:
                print(
                    f"Failed to load completed tasks file: {self.completed_tasks_file} is not a valid JSON file"
                )
            except Exception as e:
                print(f"Failed to load completed tasks file: {e}")
        else:
            print(f"Completed tasks file does not exist: {self.completed_tasks_file}")

    def _save_completed_tasks(self):
        """Save completed task IDs to file"""
        try:
            with open(self.completed_tasks_file, "w", encoding="utf-8") as f:
                json.dump(list(self.completed_tasks), f, ensure_ascii=False, indent=2)
            print(f"Completed tasks saved to: {self.completed_tasks_file}")
        except Exception as e:
            print(f"Failed to save completed tasks: {e}")

    def mark_task_completed(self, task_id: str):
        """Mark task as completed"""
        if task_id not in self.completed_tasks:
            self.completed_tasks.add(task_id)
            self._save_completed_tasks()
            print(f"Task {task_id} marked as completed")
        else:
            print(f"Task {task_id} already marked as completed")

    def is_task_completed(self, task_id: str) -> bool:
        """Check if task is completed"""
        return task_id in self.completed_tasks

    def _get_agent_for_step(self, history: List[Dict[str, Any]], step: Any) -> str:
        """Find agent corresponding to specified step in trajectory"""
        if step == "Unknown" or step == -1:
            return "Unknown"

        # Try exact step matching
        for step_data in history:
            if step_data.get("step") == step:
                agent_name = step_data.get("name", "Unknown")
                # Standardize role name
                return standardize_role_name(agent_name)

        # If exact matching fails, try string matching
        step_str = str(step)
        for step_data in history:
            if str(step_data.get("step", "")) == step_str:
                agent_name = step_data.get("name", "Unknown")
                # Standardize role name
                return standardize_role_name(agent_name)

        # If still not found, return Unknown
        print(f"  ⚠️  Cannot find agent corresponding to step {step} in history")
        return "Unknown"

    def save_attack_result(self, task_id: str, round_num: int):
        """Save attack success result (in reference_data format)"""
        # Build attack-related file paths
        attack_suggestions_dir = self.config.get_directory_structure()[
            "attack_suggestions"
        ].format(round_num)
        attacked_logs_dir = self.config.get_directory_structure()[
            "attacked_logs"
        ].format(round_num)
        attack_records_dir = self.config.get_directory_structure()[
            "attack_records"
        ].format(round_num)

        safe_task_id = task_id.replace("/", "_")

        # Read attack analysis
        attack_suggestion_file = os.path.join(
            attack_suggestions_dir, f"{safe_task_id}.json"
        )
        attacked_log_file = os.path.join(attacked_logs_dir, f"{safe_task_id}.json")
        attack_record_file = os.path.join(attack_records_dir, f"{safe_task_id}.json")

        if not all(
            os.path.exists(f)
            for f in [attack_suggestion_file, attacked_log_file, attack_record_file]
        ):
            print(f"Warning: Task {task_id} attack files are incomplete, cannot save result")
            return False

        try:
            # Read attack analysis
            with open(attack_suggestion_file, "r", encoding="utf-8") as f:
                attack_data = json.load(f)

            # Read attacked log
            with open(attacked_log_file, "r", encoding="utf-8") as f:
                attacked_log = json.load(f)

            # Read attack record
            with open(attack_record_file, "r", encoding="utf-8") as f:
                attack_record = json.load(f)

            # Verify if attack is really successful (has actual injection)
            attack_successful = attack_record.get("attack_successful", False)
            injection_history = attack_record.get("injection_history", [])
            actual_attack_step = attack_record.get("actual_attack_step", "Unknown")

            # Check if there is actual injection
            has_real_injection = (
                actual_attack_step != "Unknown"
                and actual_attack_step != -1
                and len(injection_history) > 0
            )

            print(f"Task {task_id} attack verification:")
            print(f"  Attack successful: {attack_successful}")
            print(f"  Has real injection: {has_real_injection}")
            print(f"  Injection count: {len(injection_history)}")
            print(f"  Actual attack step: {actual_attack_step}")

            # Check if it's a false positive case (attack_successful=True but no actual injection)
            is_false_positive = attack_successful and (
                actual_attack_step == -1 or len(injection_history) == 0
            )

            if is_false_positive:
                print(f"  🚫 False positive case: attack_successful=True but no actual injection, skipping save")
                return False

            # Get original log data to get system_prompts
            original_log_data = self._get_log_data(task_id)
            system_prompt = {}
            if original_log_data and "system_prompts" in original_log_data:
                system_prompt = original_log_data["system_prompts"]

            # Find corresponding agent in trajectory
            actual_agent = self._get_agent_for_step(
                attacked_log.get("history", []), actual_attack_step
            )

            # If actual_attack_step doesn't exist or is invalid, use attack suggestion information as fallback
            if actual_attack_step == "Unknown" or actual_attack_step == -1:
                actual_attack_step = attack_data.get("attack_analysis", {}).get(
                    "attack_step", "Unknown"
                )
                actual_agent = attack_data.get("mistake_agent", "Unknown")
                print(
                    f"  ⚠️  Using attack suggestion information as fallback: step={actual_attack_step}, agent={actual_agent}"
                )

            # Build result in reference_data format
            result_data = {
                "is_correct": False,  # Attack successful, result is incorrect
                "question": attacked_log.get("question", ""),
                "question_ID": task_id,
                "level": "2",  # Can be adjusted as needed
                "ground_truth": attacked_log.get("ground_truth", ""),
                "history": attacked_log.get("history", []),
                # Prioritize using actual_attack_step from attack record as decisive step
                "mistake_step": str(actual_attack_step),
                # Use actual agent found in trajectory
                "mistake_agent": standardize_role_name(actual_agent),
                "mistake_reason": attack_data.get("attack_analysis", {}).get(
                    "reason", "No reason provided"
                ),
                "system_prompt": system_prompt,  # Get system_prompts from original log
            }

            # Save result
            results_dir = os.path.join(self.config.output_base_dir, "final_results")
            os.makedirs(results_dir, exist_ok=True)

            result_file = os.path.join(
                results_dir, f"{safe_task_id}_attack_result.json"
            )
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)

            print(f"  ✅ Attack result saved: {result_file}")
            print(f"  Final mistake_step: {actual_attack_step}")
            print(f"  Final mistake_agent: {actual_agent}")
            return True

        except Exception as e:
            print(f"Failed to save attack result {task_id}: {e}")
            return False

    def save_diagnose_result(self, task_id: str, round_num: int):
        """Save diagnosis success result (in reference_data format)"""
        # Build diagnosis-related file paths
        diagnosed_logs_dir = self.config.get_directory_structure()[
            "diagnosed_logs"
        ].format(round_num)
        improved_logs_dir = self.config.get_directory_structure()[
            "improved_logs"
        ].format(round_num)
        diagnosis_records_dir = self.config.get_directory_structure()[
            "diagnosis_records"
        ].format(round_num)

        safe_task_id = task_id.replace("/", "_")

        # Read diagnosis analysis
        diagnosed_log_file = os.path.join(diagnosed_logs_dir, f"{safe_task_id}.json")
        improved_log_file = os.path.join(improved_logs_dir, f"{safe_task_id}.json")
        diagnosis_record_file = os.path.join(
            diagnosis_records_dir, f"{safe_task_id}.json"
        )

        if not all(
            os.path.exists(f)
            for f in [diagnosed_log_file, improved_log_file, diagnosis_record_file]
        ):
            print(f"Warning: Task {task_id} diagnosis files are incomplete, cannot save result")
            return False

        try:
            # Read diagnosis analysis
            with open(diagnosed_log_file, "r", encoding="utf-8") as f:
                diagnosis_data = json.load(f)

            # Read improved log (to get improved solution)
            with open(improved_log_file, "r", encoding="utf-8") as f:
                improved_log = json.load(f)

            # Read diagnosis record
            with open(diagnosis_record_file, "r", encoding="utf-8") as f:
                diagnosis_record = json.load(f)

            # Get original log data (from MetaGPT/coding/logs/kodcode/)
            original_log_data = self._get_log_data(task_id)
            if not original_log_data:
                print(f"Warning: Unable to get original log data for Task {task_id}")
                return False

            # Build result in reference_data format
            # Final result after diagnosis success should use original failed log as log source
            result_data = {
                "is_correct": False,  # Diagnosis successful, result is correct
                "question": original_log_data.get(
                    "question", ""
                ),  # Use original log's question
                "question_ID": task_id,
                "level": "2",  # Can be adjusted as needed
                "ground_truth": original_log_data.get(
                    "ground_truth", ""
                ),  # Use original log's ground_truth
                "history": original_log_data.get(
                    "history", []
                ),  # Use original log's history
                "model_prediction": improved_log.get(
                    "model_prediction", ""
                ),  # Use improved solution
                "mistake_agent": standardize_role_name(
                    diagnosis_data.get("diagnosis", {}).get("mistake_agent", "Unknown")
                ),
                "mistake_step": str(
                    diagnosis_data.get("diagnosis", {}).get("mistake_step", "Unknown")
                ),
                "mistake_reason": diagnosis_data.get("diagnosis", {}).get(
                    "reason", "No reason provided"
                ),
                "system_prompt": original_log_data.get(
                    "system_prompts", {}
                ),  # Get system_prompts from original log
            }

            # Save result
            results_dir = os.path.join(self.config.output_base_dir, "final_results")
            os.makedirs(results_dir, exist_ok=True)

            result_file = os.path.join(
                results_dir, f"{safe_task_id}_diagnose_result.json"
            )
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)

            print(f"Diagnosis result saved: {result_file}")
            return True

        except Exception as e:
            print(f"Failed to save diagnosis result {task_id}: {e}")
            return False

    def _process_attack_completion(self, round_num: int):
        """Process attack completion, check which tasks attack successfully and mark as completed"""
        print(f"\n{'='*60}")
        print(f"Processing Round {round_num} attack completion status")
        print(f"{'='*60}")

        # Get attack result files for all difficulties
        difficulties = ["easy", "medium", "hard"]
        all_attack_results = {}

        for difficulty in difficulties:
            attack_result_file = self.config.get_eval_result_file(
                round_num, "attack", difficulty
            )

            if os.path.exists(attack_result_file):
                try:
                    with open(attack_result_file, "r", encoding="utf-8") as f:
                        difficulty_results = json.load(f)
                    all_attack_results.update(difficulty_results)
                    print(
                        f"✅ Loaded {difficulty} difficulty results: {len(difficulty_results)} tasks"
                    )
                except Exception as e:
                    print(f"⚠️  Failed to read {difficulty} difficulty results file: {e}")
            else:
                print(f"⚠️  {difficulty} difficulty results file does not exist: {attack_result_file}")

        if not all_attack_results:
            print(f"Attack result files do not exist or are empty")
            return

        try:
            completed_count = 0
            skipped_count = 0
            for task_id, is_success in all_attack_results.items():
                # If attack successful (eval result is false), mark task as completed
                if not is_success:  # Attack successful
                    if not self.is_task_completed(task_id):
                        # Save attack result (includes validation logic)
                        if self.save_attack_result(task_id, round_num):
                            self.mark_task_completed(task_id)
                            completed_count += 1
                            print(f"Task {task_id} attack successful, result saved and marked as completed")
                        else:
                            # save_attack_result returns False possibly due to false positive case
                            print(
                                f"Task {task_id} attack successful, but saving result failed (possibly false positive case)"
                            )
                            skipped_count += 1
                    else:
                        print(f"Task {task_id} already completed, skipped")

            print(
                f"Round {round_num} attack completion processing: {completed_count} tasks marked as completed, {skipped_count} tasks skipped"
            )

        except Exception as e:
            print(f"Failed to process attack completion status: {e}")

    def _process_diagnose_completion(self, round_num: int):
        """Process diagnosis completion, check which tasks diagnosis successfully and mark as completed"""
        print(f"\n{'='*60}")
        print(f"Processing Round {round_num} diagnosis completion status")
        print(f"{'='*60}")

        # Get diagnosis result files for all difficulties
        difficulties = ["easy", "medium", "hard"]
        all_diagnose_results = {}

        for difficulty in difficulties:
            diagnose_result_file = self.config.get_eval_result_file(
                round_num, "diagnose", difficulty
            )

            if os.path.exists(diagnose_result_file):
                try:
                    with open(diagnose_result_file, "r", encoding="utf-8") as f:
                        difficulty_results = json.load(f)
                    all_diagnose_results.update(difficulty_results)
                    print(
                        f"✅ Loaded {difficulty} difficulty results: {len(difficulty_results)} tasks"
                    )
                except Exception as e:
                    print(f"⚠️  Failed to read {difficulty} difficulty results file: {e}")
            else:
                print(f"⚠️  {difficulty} difficulty results file does not exist: {diagnose_result_file}")

        if not all_diagnose_results:
            print(f"Diagnosis result files do not exist or are empty")
            return

        try:
            completed_count = 0
            for task_id, is_success in all_diagnose_results.items():
                # If diagnosis successful (eval result is true), mark task as completed
                if is_success:  # Diagnosis successful
                    if not self.is_task_completed(task_id):
                        self.mark_task_completed(task_id)
                        # Save diagnosis result
                        if self.save_diagnose_result(task_id, round_num):
                            completed_count += 1
                            print(f"Task {task_id} diagnosis successful, result saved and marked as completed")
                        else:
                            print(f"Task {task_id} diagnosis successful, but saving result failed")
                    else:
                        print(f"Task {task_id} already completed, skipped")

            print(
                f"Round {round_num} diagnosis completion processing: {completed_count} tasks marked as completed"
            )

        except Exception as e:
            print(f"Failed to process diagnosis completion status: {e}")


def create_framework(
    dataset_type: str, workspace_dir: str, output_dir: str, **kwargs
) -> UniversalFramework:
    """Create framework instance"""
    if dataset_type == "kodcode":
        config = create_kodcode_config(workspace_dir, output_dir)
    elif dataset_type == "mbpp_plus":
        config = create_mbpp_plus_config(workspace_dir, output_dir)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    # Update configuration
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return UniversalFramework(config)


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Universal attack and diagnosis framework")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["kodcode", "mbpp_plus"],
        help="Dataset type",
    )
    parser.add_argument("--work_dir", type=str, required=True, help="Working directory path")
    parser.add_argument("--output", type=str, required=True, help="Output directory path")
    parser.add_argument("--max_rounds", type=int, default=3, help="Maximum number of rounds")
    parser.add_argument("--max_tasks", type=int, default=None, help="Maximum number of tasks per round")
    parser.add_argument("--concurrent", action="store_true", help="Enable concurrent processing")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "attack", "diagnose"],
        default="full",
        help="Run mode",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=1,
        help="Specify round number (only used in attack or diagnose mode)",
    )
    parser.add_argument("--config_file", type=str, help="Configuration file path")

    # Retry related parameters
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum retry count (default: 3)",
    )
    parser.add_argument(
        "--retry_delay",
        type=int,
        default=2,
        help="Retry interval in seconds (default: 2)",
    )
    parser.add_argument(
        "--backoff_factor",
        type=float,
        default=1.5,
        help="Backoff factor (default: 1.5)",
    )

    args = parser.parse_args()

    # Load configuration from file
    if args.config_file and os.path.exists(args.config_file):
        config = FrameworkConfig.load_from_file(args.config_file)
        # Update retry configuration
        config.retry_config.max_retries = args.max_retries
        config.retry_config.retry_delay = args.retry_delay
        config.retry_config.backoff_factor = args.backoff_factor
        framework = UniversalFramework(config)
    else:
        # Create configuration from command line arguments
        framework = create_framework(
            dataset_type=args.dataset,
            workspace_dir=args.work_dir,
            output_dir=args.output,
            max_rounds=args.max_rounds,
            max_tasks_per_round=args.max_tasks,
            enable_concurrent=args.concurrent,
        )
        # Update retry configuration
        framework.config.retry_config.max_retries = args.max_retries
        framework.config.retry_config.retry_delay = args.retry_delay
        framework.config.retry_config.backoff_factor = args.backoff_factor
        # Reinitialize retry manager
        framework.retry_manager = get_retry_manager(
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            backoff_factor=args.backoff_factor,
        )

    # Print retry configuration
    print(f"Retry configuration:")
    print(f"  Maximum retry count: {framework.config.retry_config.max_retries}")
    print(f"  Retry interval: {framework.config.retry_config.retry_delay} seconds")
    print(f"  Backoff factor: {framework.config.retry_config.backoff_factor}")

    try:
        if args.mode == "full":
            await framework.run_full_pipeline(args.max_rounds, args.max_tasks)
        elif args.mode == "attack":
            await framework.run_attack_only(args.round, args.max_tasks)
        elif args.mode == "diagnose":
            await framework.run_diagnose_only(args.round, args.max_tasks)

        # Print retry statistics
        framework.retry_manager.print_stats()

        # Save retry statistics
        stats_dir = os.path.join(framework.config.output_base_dir, "stats")
        retry_stats_file = os.path.join(stats_dir, "retry_stats.json")
        framework.retry_manager.save_stats(retry_stats_file)

        framework.print_status()

    except KeyboardInterrupt:
        print("\nUser interrupted execution")
        # Save current retry statistics
        if hasattr(framework, "retry_manager"):
            stats_dir = os.path.join(framework.config.output_base_dir, "stats")
            retry_stats_file = os.path.join(stats_dir, "retry_stats_interrupted.json")
            framework.retry_manager.save_stats(retry_stats_file)
    except Exception as e:
        print(f"Execution failed: {e}")
        # Save current retry statistics
        if hasattr(framework, "retry_manager"):
            stats_dir = os.path.join(framework.config.output_base_dir, "stats")
            retry_stats_file = os.path.join(stats_dir, "retry_stats_error.json")
            framework.retry_manager.save_stats(retry_stats_file)
        import traceback

        traceback.print_exc()
    finally:
        # Reset global retry manager
        reset_retry_manager()


if __name__ == "__main__":
    asyncio.run(main())
