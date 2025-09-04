#!/usr/bin/env python3
"""
Module for diagnosing failed logs - uses the MetaGPT framework for diagnostic analysis
"""

import os
import sys
import json
import asyncio
import argparse
from typing import List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from diagnose_monitor import (
    DiagnosisAttackMonitor,
    DiagnosisInjectionInterceptor,
)


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


# Use environment variables or relative paths
metagpt_path = os.getenv(
    "METAGPT_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
sys.path.append(metagpt_path)
from metagpt.software_company import generate_repo

# Import diagnosis monitor
from diagnose_monitor import (
    DiagnosisAttackMonitor,
    DiagnosisInjectionInterceptor,
)

# Directory configuration - use environment variables or default paths
base_dir = os.path.dirname(os.path.abspath(__file__))

# Detect dataset type
dataset_type = "mbpp+"  # Default value
if "kodcode" in base_dir:
    dataset_type = "kodcode"
elif "mbpp" in base_dir:
    dataset_type = "mbpp+"

# Set default paths based on dataset type
if dataset_type == "kodcode":
    default_diagnosed_log_base = os.path.join(
        base_dir, "kodcode/diagnosed_logs_round_3"
    )
    default_improved_log_base = os.path.join(base_dir, "kodcode/improved_logs_round_3")
    default_diagnosis_record_dir = os.path.join(base_dir, "kodcode/diagnosis_records")
    default_failed_log_dir = os.path.join(
        base_dir, "kodcode/replayed_still_failed_tasks_round_2.json"
    )
else:
    default_diagnosed_log_base = os.path.join(base_dir, "mbpp+/diagnosed_logs_round_3")
    default_improved_log_base = os.path.join(base_dir, "mbpp+/improved_logs_round_3")
    default_diagnosis_record_dir = os.path.join(base_dir, "mbpp+/diagnosis_records")
    default_failed_log_dir = os.path.join(
        base_dir, "mbpp+/replayed_still_failed_tasks_round_2.json"
    )

# Load directory configuration from environment variables, or use default values
DIAGNOSED_LOG_BASE = os.getenv("DIAGNOSED_LOG_BASE", default_diagnosed_log_base)
IMPROVED_LOG_BASE = os.getenv("IMPROVED_LOG_BASE", default_improved_log_base)
DIAGNOSIS_RECORD_DIR = os.getenv("DIAGNOSIS_RECORD_DIR", default_diagnosis_record_dir)
WORKSPACE_DIR = os.getenv(
    "WORKSPACE_DIR", os.path.join(os.path.dirname(base_dir), "workspace")
)

FAILED_LOG_DIR = os.getenv("FAILED_LOG_DIR", default_failed_log_dir)

# Create necessary directories
os.makedirs(DIAGNOSED_LOG_BASE, exist_ok=True)
os.makedirs(IMPROVED_LOG_BASE, exist_ok=True)
os.makedirs(WORKSPACE_DIR, exist_ok=True)
os.makedirs(DIAGNOSIS_RECORD_DIR, exist_ok=True)


def get_diagnosed_log_path(log_name):
    """Dynamically get the diagnosis log path"""
    diagnosed_log_base = os.getenv("DIAGNOSED_LOG_BASE", DIAGNOSED_LOG_BASE)
    return os.path.join(diagnosed_log_base, log_name)


def get_improved_log_path(log_name):
    """Dynamically get the improved log path"""
    improved_log_base = os.getenv("IMPROVED_LOG_BASE", IMPROVED_LOG_BASE)
    return os.path.join(improved_log_base, log_name)


def get_diagnosis_record_path(log_name):
    """Dynamically get the diagnosis record path"""
    diagnosis_record_dir = os.getenv("DIAGNOSIS_RECORD_DIR", DIAGNOSIS_RECORD_DIR)
    return os.path.join(diagnosis_record_dir, log_name)


def get_current_round():
    """Dynamically get the current round number"""
    current_round = 3  # Default value
    diagnosed_log_base = os.getenv("DIAGNOSED_LOG_BASE", DIAGNOSED_LOG_BASE)

    if "round_2" in diagnosed_log_base:
        current_round = 2
    elif "round_1" in diagnosed_log_base:
        current_round = 1
    else:
        # Try to extract round number from path
        import re

        match = re.search(r"round_(\d+)", diagnosed_log_base)
        if match:
            current_round = int(match.group(1))
    return current_round


def load_previous_diagnosis_analyses(task_id: str) -> List[dict]:
    """Load all previous rounds' diagnosis analyses"""
    analyses = []

    # Load current round from environment variable
    current_round = 3  # Default value
    diagnosed_log_base = os.getenv("DIAGNOSED_LOG_BASE", "")

    if "round_3" in diagnosed_log_base:
        current_round = 3
    elif "round_2" in diagnosed_log_base:
        current_round = 2
    elif "round_1" in diagnosed_log_base:
        current_round = 1
    else:
        # Try to extract round number from path
        import re

        match = re.search(r"round_(\d+)", diagnosed_log_base)
        if match:
            current_round = int(match.group(1))

    # Iterate through all previous rounds
    for round_num in range(1, current_round):
        # Build the path for this round
        round_path = diagnosed_log_base.replace(
            f"round_{current_round}", f"round_{round_num}"
        )
        previous_path = os.path.join(round_path, f"{task_id.replace('/', '_')}.json")

        if os.path.exists(previous_path):
            try:
                with open(previous_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    diagnosis = data.get("diagnosis")
                    if diagnosis:
                        # Add round number info
                        diagnosis["round"] = round_num
                        analyses.append(diagnosis)
                        print(
                            f"[HISTORY] Loaded round {round_num} diagnosis analysis: step {diagnosis.get('mistake_step', 'Unknown')}"
                        )
            except Exception as e:
                print(
                    f"Error loading round {round_num} diagnosis from {previous_path}: {e}"
                )

    print(f"[HISTORY] Loaded {len(analyses)} previous round diagnosis analyses in total")
    return analyses


async def diagnose_with_meta_company(failed_log):
    """Diagnose using MetaGPT's software company"""
    history = []
    step_counter = {"step": 0}

    def log_step(content, role, name):
        print(f"[LOG_STEP] {role} {name}: {content}")
        history.append(
            {
                "step": step_counter["step"],
                "content": content,
                "role": role,
                "name": name,
            }
        )
        step_counter["step"] += 1

    # Build diagnosis prompt
    task_id = failed_log["task_id"]

    # Handle unified data structure
    if "original_log" in failed_log:
        # New format: data in original_log
        original_log = failed_log["original_log"]
        original_history = original_log.get("history", [])
        question = original_log.get("question", "")
        ground_truth = original_log.get("ground_truth", "")
        model_prediction = original_log.get("model_prediction", "")
    else:
        # Old format: data directly in failed_log
        original_log = failed_log
        original_history = failed_log.get("history", [])
        question = failed_log.get("question", "")
        ground_truth = failed_log.get("ground_truth", "")
        model_prediction = failed_log.get("model_prediction", "")

    history_str = ""
    for step in original_history:
        history_str += f"\nStep {step['step']} - {step['name']} ({step['role']}):\n{step['content']}\n"

    # Load previous diagnosis analyses
    # Prefer using framework-passed history
    if "previous_diagnosis_analyses" in failed_log:
        previous_diagnosis_analyses = failed_log["previous_diagnosis_analyses"]
        print(
            f"[FRAMEWORK_HISTORY] Using framework-passed previous diagnosis analyses: {len(previous_diagnosis_analyses)} rounds"
        )
    else:
        # Backward compatibility: use original loading method
        previous_diagnosis_analyses = load_previous_diagnosis_analyses(task_id)
        print(f"[FALLBACK_HISTORY] Using fallback method to load previous diagnosis analyses")

    previous_diagnosis_info = ""
    if previous_diagnosis_analyses and len(previous_diagnosis_analyses) > 0:
        print("------previous diagnosis analyses info added------\n")
        previous_diagnosis_info = "IMPORTANT - Previous Diagnosis Analysis History:\n"
        previous_diagnosis_info += f"Multiple diagnosis analyses have been attempted for this task ({len(previous_diagnosis_analyses)} rounds):\n\n"

        for i, prev_diag in enumerate(previous_diagnosis_analyses):
            round_num = prev_diag.get("round", i + 1)
            previous_diagnosis_info += f"Round {round_num}:\n"
            previous_diagnosis_info += (
                f"- Previous mistake_step: {prev_diag.get('mistake_step', 'Unknown')}\n"
            )
            previous_diagnosis_info += (
                f"- Previous reason: {prev_diag.get('reason', 'No reason provided')}\n"
            )
            previous_diagnosis_info += f"- Previous suggested_fix: {prev_diag.get('suggested_fix', 'No fix suggested')}\n\n"

        # Build excluded steps list
        excluded_steps = [
            str(prev_diag.get("mistake_step", ""))
            for prev_diag in previous_diagnosis_analyses
            if prev_diag.get("mistake_step")
        ]
        excluded_steps_str = ", ".join(excluded_steps) if excluded_steps else "none"

        previous_diagnosis_info += f"CRITICAL GUIDELINES FOR THIS DIAGNOSIS:\n"
        previous_diagnosis_info += f"1. DO NOT diagnose any of the following steps again - they have already been identified and attempted: {excluded_steps_str}\n"
        previous_diagnosis_info += f"2. Look for OTHER steps in the execution history that might contain the actual root cause.\n"
        previous_diagnosis_info += f"3. The previous diagnosis analyses may have been incorrect or incomplete.\n"
        previous_diagnosis_info += f"4. Focus on analyzing different aspects of the implementation or different steps in the process.\n"
        previous_diagnosis_info += f"5. Consider if the issue might be in the task understanding, algorithm design, or a different implementation step.\n"
        previous_diagnosis_info += f"6. Analyze the pattern of previous attempts to avoid similar ineffective approaches.\n"
    elif failed_log.get("previous_diagnosis"):
        # Backward compatibility for single previous_diagnosis format
        print("------previous diagnosis info added------\n")
        prev_diag = failed_log["previous_diagnosis"]
        previous_diagnosis_info = f"""
IMPORTANT - Previous Diagnosis Information:
A previous diagnosis was already attempted for this task:
- Previous mistake_step: {prev_diag['mistake_step']}
- Previous reason: {prev_diag['reason']}
- Previous suggested_fix: {prev_diag['suggested_fix']}

CRITICAL GUIDELINES FOR THIS DIAGNOSIS:
1. DO NOT diagnose step {prev_diag['mistake_step']} again - it has already been identified and attempted.
2. Look for OTHER steps in the execution history that might contain the actual root cause.
3. The previous diagnosis may have been incorrect or incomplete.
4. Focus on analyzing different aspects of the implementation or different steps in the process.
5. Consider if the issue might be in the task understanding, algorithm design, or a different implementation step.
"""

    # Build history constraint info
    history_constraint = ""
    if previous_diagnosis_analyses and len(previous_diagnosis_analyses) > 0:
        history_constraint = f"""
🚨 CRITICAL CONSTRAINT - YOU MUST FOLLOW THIS RULE:
You are FORBIDDEN from diagnosing step(s): {excluded_steps_str}
If you diagnose any of these steps again, your diagnosis will be REJECTED and considered a FAILURE.
You MUST choose a DIFFERENT step that has NOT been diagnosed before.

MANDATORY DIAGNOSIS RULES:
1. You MUST NOT diagnose step(s): {excluded_steps_str}
2. You MUST choose a DIFFERENT step that has NOT been diagnosed in previous rounds
3. If you repeat any previous diagnosis, your response will be considered INVALID
4. Focus on steps that have NOT been identified as problematic in previous attempts
5. Consider alternative error sources and different implementation aspects
6. Look for issues in different parts of the code or different agents' contributions

PREVIOUS ATTEMPTS SUMMARY:
"""
        for i, analysis in enumerate(previous_diagnosis_analyses):
            history_constraint += f"- Round {analysis.get('round', i+1)}: Step {analysis.get('mistake_step', 'Unknown')} ({analysis.get('mistake_agent', 'Unknown')})\n"

        history_constraint += f"""
YOUR TASK: Diagnose a DIFFERENT step than those listed above. Choose step numbers other than: {excluded_steps_str}

VALIDATION CHECK: Before submitting your diagnosis, verify that your mistake_step is NOT in this list: {excluded_steps_str}
If it is, you MUST choose a different step.
"""

    def validate_diagnosis(diagnosis, original_history):
        """Validate the correctness of the diagnosis result"""
        if not diagnosis:
            return False, "Diagnosis result is empty"

        mistake_step = diagnosis.get("mistake_step", "Unknown")
        mistake_agent = diagnosis.get("mistake_agent", "Unknown")

        # Check if mistake_step exists in history
        step_exists = False
        actual_agent = None
        for step in original_history:
            if str(step.get("step", "")) == str(mistake_step):
                step_exists = True
                actual_agent = step.get("name", "Unknown")
                break

        if not step_exists:
            return False, f"mistake_step {mistake_step} does not exist in history"

        # Check if mistake_agent matches the agent in history
        if actual_agent and actual_agent != mistake_agent:
            return (
                False,
                f"mistake_step {mistake_step} corresponds to agent '{actual_agent}', not '{mistake_agent}'",
            )

        # Check if mistake_agent exists in history
        agent_exists = False
        for step in original_history:
            if step.get("name") == mistake_agent:
                agent_exists = True
                break

        if not agent_exists:
            return False, f"mistake_agent '{mistake_agent}' does not exist in history"

        return True, "Validation passed"

    # Add validation and retry mechanism
    max_retries = 4
    for attempt in range(max_retries):
        print(f"[DIAGNOSIS_ATTEMPT] Round {attempt + 1} diagnosis attempt")

        # Build enhanced prompt, emphasize validation requirements
        validation_instruction = (
            f"""
CRITICAL VALIDATION REQUIREMENTS:
1. The mistake_step MUST exist in the execution history
2. The mistake_agent MUST match the agent that actually performed the step
3. The mistake_agent MUST exist in the execution history
4. Double-check your diagnosis before submitting

VALIDATION CHECKLIST:
- Verify that step {diagnosis.get('mistake_step', 'X')} exists in the history
- Verify that the agent you identified actually performed that step
- Verify that the agent exists in the history
- If any verification fails, choose a different step or agent

"""
            if attempt > 0
            else ""
        )

        prompt = f"""You are a software development team tasked with diagnosing a failed programming task. Your goal is to identify the critical error in the implementation.

Task Information:
Task ID: {task_id}
Question: {question}
Ground Truth: {ground_truth}
Model Prediction: {model_prediction}

{previous_diagnosis_info}

Original Task Execution History:
{history_str}

{history_constraint}

{validation_instruction}

Your diagnosis should be in the following JSON format:
{{
    "mistake_step": <step_number>,  // The step number where the error occurred
    "mistake_agent": "<the agent that made the mistake>",  // The agent that made the mistake (e.g., "Engineer", "Architect", "ProductManager", etc.)
    "reason": <detailed_explanation>,  // Detailed explanation of why this step is wrong
    "suggested_fix": <fix_guidance>  // Guidance on how to fix the error, NOT the complete solution
}}

Important Guidelines:
1. DO NOT provide the complete solution in the suggested_fix. Only provide guidance on how to fix the error.
2. Focus on identifying the root cause of the failure.
3. The 'mistake_step' should be a number corresponding to a step in the implementation process.
4. The 'mistake_agent' should be the specific agent that made the mistake (e.g., "Engineer", "Architect", "ProductManager", "TeamLeader", "DataAnalyst").
5. The 'reason' should be detailed and explain why the current implementation is incorrect.
6. The 'suggested_fix' should provide clear guidance without giving away the complete solution.
7. Analyze the original task execution history to understand the context and identify where things went wrong.
8. CRITICAL: Before submitting, verify that your mistake_step exists in the history and your mistake_agent matches the agent that actually performed that step.

IMPORTANT: To save the diagnosis result, you MUST use the Editor.create_file command with the following format:
First, create the file using Editor.create_file command
{{
    "command_name": "Editor.create_file",
    "args": {{
        "filename": "{task_id.replace('/', '_')}_diagnosis.json"
    }}
}}
Then use command to modify the content.
Please analyze the task and provide your diagnosis in the specified JSON format. The diagnosis result should be saved to a file named '{task_id.replace('/', '_')}_diagnosis.json' in the workspace directory."""

        try:
            team = await generate_repo(
                idea=prompt,
                n_round=3,  # Reduce number of rounds to speed up diagnosis
                code_review=True,
                run_tests=False,  # No need to run tests during diagnosis
                implement=False,  # No need to implement during diagnosis
                project_name=f"diagnose_{task_id.replace('/', '_')}",
                project_path=WORKSPACE_DIR,
                project_dir=WORKSPACE_DIR,
                log_step=log_step,
                use_async=True,
            )

            # Check if diagnosis result file is generated
            diagnosis_file = os.path.join(
                WORKSPACE_DIR, f"{task_id.replace('/', '_')}_diagnosis.json"
            )
            if os.path.exists(diagnosis_file):
                with open(diagnosis_file, "r") as f:
                    diagnosis = json.load(f)

                    # Validate diagnosis result
                    is_valid, validation_message = validate_diagnosis(
                        diagnosis, original_history
                    )

                    if is_valid:
                        print(f"[DIAGNOSIS_SUCCESS] Diagnosis validation passed: {validation_message}")
                        return diagnosis, history
                    else:
                        print(
                            f"[DIAGNOSIS_VALIDATION_FAILED] Round {attempt + 1} validation failed: {validation_message}"
                        )

                        # If not the last attempt, delete file and retry
                        if attempt < max_retries - 1:
                            try:
                                os.remove(diagnosis_file)
                                print(f"[DIAGNOSIS_RETRY] Deleted failed diagnosis file, preparing to retry")
                            except:
                                pass
                        else:
                            print(
                                f"[DIAGNOSIS_MAX_RETRIES] Maximum retry attempts reached, using last result"
                            )
                            return diagnosis, history
            else:
                print(f"[DIAGNOSIS_FILE_MISSING] Round {attempt + 1} did not generate diagnosis file")
                if attempt == max_retries - 1:
                    print(f"Warning: Diagnosis file not found for {task_id}")
                    return None, None

        except Exception as e:
            print(f"[DIAGNOSIS_ERROR] Round {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                print(f"Diagnosis failed after {max_retries} attempts: {e}")
                return None, None

    return None, None


async def generate_improved_solution_with_diagnosis_monitoring(failed_log, diagnosis):
    """Generate improved solution using diagnosis monitor"""
    history = []

    # Create unified step counter
    class StepCounter:
        def __init__(self):
            self.step = 0
            self.roles = {}  # Store references to all Role instances

        def increment(self):
            self.step += 1
            # Synchronize step counter for all Role instances
            for role in self.roles.values():
                if hasattr(role, "step_counter"):
                    role.step_counter = self.step

        def get_step(self):
            return self.step

        def reset(self):
            """Reset step counter"""
            self.step = 0
            # Synchronize step counter for all Role instances
            for role in self.roles.values():
                if hasattr(role, "step_counter"):
                    role.step_counter = 0

    step_counter = StepCounter()

    # Reset step counter to ensure each new task starts from 0
    step_counter.reset()

    # Debug info
    print(f"[DEBUG] Step counter reset: {step_counter.get_step()}")

    # Create diagnosis attack monitor
    monitor = DiagnosisAttackMonitor(
        failed_log["task_id"], diagnosis, original_failed_log=failed_log
    )
    interceptor = DiagnosisInjectionInterceptor(monitor)

    def log_step(content, role, name):
        print(f"[LOG_STEP] {role} {name}: {content}")

        # Standardize role name
        standardized_name = standardize_role_name(name)

        # Check if this is terminal output, if so merge with previous step
        if role == "Terminal" and content.startswith("Terminal output:"):
            # Merge terminal output into previous step
            if history:
                last_step = history[-1]
                # Add terminal output to content
                last_step["content"] += f"\n\n{content}"
                print(
                    f"[LOG_STEP] Terminal output merged into previous step: {last_step['step']}"
                )
            else:
                # If no previous step, create a new step
                current_step = step_counter.get_step()
                history.append(
                    {
                        "step": current_step,
                        "content": content,
                        "role": role,
                        "name": standardized_name,
                    }
                )
                step_counter.increment()
            return

        # Log monitor event
        monitor.log_event("step", role, standardized_name, content)

        # Check if the step before the error step is detected
        if not monitor.detected_pre_mistake_step:
            if monitor.check_pre_mistake_step(content, role, standardized_name):
                current_step = step_counter.get_step()
                monitor.record_pre_mistake_detection(current_step)
                print(
                    f"[DIAGNOSIS_MONITOR] Detected the step before the error step {monitor.mistake_step}! (actual step: {current_step})"
                )
                print(f"[DIAGNOSIS_MONITOR] Preparing to inject correction in the next step...")

                # Set waiting for injection state
                monitor.injection_pending = True
                monitor.injection_completed = False
                print(f"[DIAGNOSIS_MONITOR] Set waiting for injection state, system will wait for injection to complete...")

        # Record current step
        current_step = step_counter.get_step()
        history.append(
            {
                "step": current_step,
                "content": content,
                "role": role,
                "name": standardized_name,
            }
        )
        # Increment step counter
        step_counter.increment()

        # Synchronize step counter for all Role instances (only within reasonable range)
        current_step = step_counter.get_step()
        if current_step <= 1000:  # Only synchronize within reasonable range
            for r in step_counter.roles.values():
                if hasattr(r, "step_counter"):
                    r.step_counter = current_step
        else:
            # If step counter is abnormal, reset it
            print(f"[DEBUG] Detected abnormal step counter value: {current_step}, resetting...")
            step_counter.reset()
            for r in step_counter.roles.values():
                if hasattr(r, "step_counter"):
                    r.step_counter = 0

        # Synchronize monitor's step counter
        monitor.update_step(step_counter.get_step())

    # Build improved prompt - remove diagnosis info, let monitor inject dynamically
    task_id = failed_log["task_id"]

    # Get question info
    if "original_log" in failed_log:
        question = failed_log["original_log"].get("question", "")
    else:
        question = failed_log.get("question", "")

    # Build prompt containing diagnosis info
    diagnosis_info = f"""
CRITICAL DIAGNOSIS INFORMATION - MUST BE CONSIDERED:
A previous analysis identified a critical issue in the implementation:

MISTAKE STEP: {diagnosis.get('mistake_step', 'Unknown')}
REASON: {diagnosis.get('reason', 'No reason provided')}
SUGGESTED FIX: {diagnosis.get('suggested_fix', 'No fix suggested')}

IMPORTANT: The team MUST avoid the mistake identified in step {diagnosis.get('mistake_step', 'Unknown')} and follow the suggested fix when implementing the solution.
"""

    prompt = f"""Complete the following programming task, with a multi-agent team cooperation
    the final python code solution file you write should be named as "{task_id.replace('/', '_')}_improved_solution.py":

Task ID: {task_id}
Question: {question}

{diagnosis_info}

"""

    try:
        # Create custom generate_repo function to get Role instances
        from metagpt.software_company import generate_repo as original_generate_repo
        from metagpt.team import Team
        from metagpt.context import Context
        from metagpt.config2 import config
        from metagpt.roles import (
            Architect,
            DataAnalyst,
            Engineer2,
            ProductManager,
            TeamLeader,
        )

        # Custom generate_repo function
        def custom_generate_repo(idea, **kwargs):
            config.update_via_cli(
                kwargs.get("project_path", ""),
                kwargs.get("project_name", ""),
                kwargs.get("inc", False),
                kwargs.get("reqa_file", ""),
                kwargs.get("max_auto_summarize_code", 0),
            )
            ctx = Context(config=config)

            company = Team(context=ctx)
            roles = [
                TeamLeader(
                    project_dir=kwargs.get("project_dir"),
                    log_step=kwargs.get("log_step"),
                    attack_monitor=kwargs.get("diagnosis_monitor"),
                ),
                ProductManager(
                    project_dir=kwargs.get("project_dir"),
                    log_step=kwargs.get("log_step"),
                    attack_monitor=kwargs.get("diagnosis_monitor"),
                ),
                Architect(
                    project_dir=kwargs.get("project_dir"),
                    log_step=kwargs.get("log_step"),
                    attack_monitor=kwargs.get("diagnosis_monitor"),
                ),
                Engineer2(
                    project_dir=kwargs.get("project_dir"),
                    log_step=kwargs.get("log_step"),
                    attack_monitor=kwargs.get("diagnosis_monitor"),
                ),
                DataAnalyst(
                    project_dir=kwargs.get("project_dir"),
                    log_step=kwargs.get("log_step"),
                    attack_monitor=kwargs.get("diagnosis_monitor"),
                ),
            ]

            company.hire(roles)

            # Register Role instances to step counter
            for role in roles:
                step_counter.roles[role.name] = role
                # Ensure diagnosis_monitor is also set correctly
                if hasattr(role, "attack_monitor"):
                    role.attack_monitor = monitor

            company.invest(kwargs.get("investment", 3.0))
            return company.run(n_round=kwargs.get("n_round", 5), idea=idea)

        team_result = await custom_generate_repo(
            idea=prompt,
            n_round=5,
            code_review=True,
            run_tests=True,
            implement=True,
            project_name=f"improve_{task_id.replace('/', '_')}",
            project_path=WORKSPACE_DIR,
            project_dir=WORKSPACE_DIR,
            log_step=log_step,
            use_async=True,
            diagnosis_monitor=monitor,
        )

        # Check if improved solution is generated
        solution_file = os.path.join(
            WORKSPACE_DIR, f"{task_id.replace('/', '_')}_improved_solution.py"
        )
        if os.path.exists(solution_file):
            with open(solution_file, "r") as f:
                improved_solution = f.read()

                # Mark correction as successful (can be further refined based on test results)
                monitor.correction_successful = (
                    True  # Set to True for now, can be updated based on test results
                )

                return improved_solution, history, monitor
        else:
            print(f"Warning: Improved solution file not found for {task_id}")

            # Try to extract solution from history
            solution = extract_solution_from_history(history)
            if solution:
                print(f"Extracted solution from history for {task_id}")
                monitor.correction_successful = True
                return solution, history, monitor

            return None, None, monitor

    except Exception as e:
        print(f"Solution generation failed: {e}")
        return None, None, monitor


def extract_solution_from_history(history):
    """Extract solution from history records"""
    for step in reversed(history):
        content = step.get("content", "")

        # Look for Python code
        if "def " in content or "import " in content:
            lines = content.split("\n")
            code_lines = []
            in_code_block = False

            for line in lines:
                if "```python" in line or "```" in line:
                    in_code_block = not in_code_block
                    continue
                if in_code_block or line.strip().startswith(
                    ("def ", "import ", "class ", "from ")
                ):
                    code_lines.append(line)

            if code_lines:
                return "\n".join(code_lines)

    return None


async def process_failed_log(failed_log):
    """Process a single failed log file"""
    log_name = f"{failed_log['task_id'].replace('/', '_')}.json"

    # Resume from breakpoint: if log already exists, skip this task
    diagnosed_path = get_diagnosed_log_path(log_name)
    improved_path = get_improved_log_path(log_name)
    diagnosis_record_path = get_diagnosis_record_path(log_name)

    # Check if all required files exist
    all_files_exist = (
        os.path.exists(diagnosed_path)
        and os.path.exists(improved_path)
        and os.path.exists(diagnosis_record_path)
    )

    # Modify skip logic: only skip in round 1 if all files exist, in round 2 and above should regenerate diagnosis and improved solution
    # Dynamically get current round at runtime
    current_round = get_current_round()

    # Only skip if in round 1 and all files exist
    if current_round == 1 and all_files_exist:
        print(
            f"[INFO] All logs for task {failed_log['task_id']} already exist in round 1, skipping..."
        )
        return

    # Load previous diagnosis analyses
    # Prefer using framework-passed history
    if "previous_diagnosis_analyses" in failed_log:
        previous_diagnosis_analyses = failed_log["previous_diagnosis_analyses"]
        print(
            f"[FRAMEWORK_HISTORY] Using framework-passed previous diagnosis analyses: {len(previous_diagnosis_analyses)} rounds"
        )
    else:
        # Backward compatibility: use original loading method
        previous_diagnosis_analyses = load_previous_diagnosis_analyses(
            failed_log["task_id"]
        )
        print(f"[FALLBACK_HISTORY] Using fallback method to load previous diagnosis analyses")

    if previous_diagnosis_analyses:
        print(
            f"[PREVIOUS_DIAGNOSIS] Loaded previous diagnosis analyses for {failed_log['task_id']} from {len(previous_diagnosis_analyses)} rounds."
        )
        for i, prev_analysis in enumerate(previous_diagnosis_analyses):
            print(f"  Round {i+1}: step {prev_analysis.get('mistake_step', 'Unknown')}")
    else:
        print(
            f"[PREVIOUS_DIAGNOSIS] No previous diagnosis analyses found for {failed_log['task_id']}"
        )

    # 1. Diagnosis
    diagnosis, diagnosis_history = await diagnose_with_meta_company(failed_log)

    if diagnosis is None:
        print(f"Diagnosis failed for {log_name}")
        return

    # Save diagnosis result (including history)
    with open(diagnosed_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "diagnosis": diagnosis,
                "history": diagnosis_history,
                "previous_diagnosis_analyses": previous_diagnosis_analyses,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 2. Generate improved solution (with diagnosis monitoring)
    print(
        f"[DIAGNOSIS_MONITORING] Starting diagnosis monitoring for {failed_log['task_id']}"
    )
    improved_solution, improved_history, monitor = (
        await generate_improved_solution_with_diagnosis_monitoring(
            failed_log, diagnosis
        )
    )

    if improved_solution is None:
        print(f"Solution generation failed for {log_name}")
        # Even if failed, save monitor record
        with open(diagnosis_record_path, "w", encoding="utf-8") as f:
            json.dump(monitor.get_diagnosis_record(), f, ensure_ascii=False, indent=2)
        return

    # Get original data
    if "original_log" in failed_log:
        original_log = failed_log["original_log"]
        question = original_log.get("question", "")
        ground_truth = original_log.get("ground_truth", "")
    else:
        question = failed_log.get("question", "")
        ground_truth = failed_log.get("ground_truth", "")

    # Save improved solution and log
    with open(improved_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "question": question,
                "question_ID": failed_log["task_id"],
                "ground_truth": ground_truth,
                "model_prediction": improved_solution,
                "history": improved_history,
                "diagnosis": diagnosis,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Save diagnosis monitor record
    with open(diagnosis_record_path, "w", encoding="utf-8") as f:
        json.dump(monitor.get_diagnosis_record(), f, ensure_ascii=False, indent=2)

    print(
        f"[DIAGNOSIS_MONITORING_RESULT] Task {failed_log['task_id']}: "
        f"Detected={monitor.detected_pre_mistake_step}, "
        f"Injected={monitor.injected_correction}, "
        f"Successful={monitor.correction_successful}"
    )

    # Show injection history
    if monitor.injection_history:
        print(f"[INJECTION_HISTORY] Injection history:")
        for i, injection in enumerate(monitor.injection_history):
            print(f"  {i+1}. Strategy: {injection['strategy']}, Step: {injection['step']}")


async def main():
    # Add command line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=str, help="Specific task ID to process")
    args = parser.parse_args()

    # Load failed logs
    with open(FAILED_LOG_DIR, "r") as f:
        failed_logs = json.load(f)

    # If task_id is specified, only process that task
    if args.task_id:
        failed_logs = [log for log in failed_logs if log["task_id"] == args.task_id]
        if not failed_logs:
            print(f"Task ID {args.task_id} not found in failed logs")
            return
        print(f"Processing specific task: {args.task_id}")

    # Process each failed log
    for failed_log in failed_logs:
        await process_failed_log(failed_log)


if __name__ == "__main__":
    asyncio.run(main())
