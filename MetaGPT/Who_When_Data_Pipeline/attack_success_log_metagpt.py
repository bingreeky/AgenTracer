#!/usr/bin/env python3
"""
Attack Success Log Processing Module - Using MetaGPT framework for attack analysis
"""

import os
import sys
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from attack_monitor import AdvancedAttackMonitor, PromptInjectionInterceptor


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

# Import advanced monitor
from Who_When_Data_Pipeline.attack_monitor import (
    AdvancedAttackMonitor,
    PromptInjectionInterceptor,
)

# Directory configuration - Use environment variables or default paths
base_dir = os.path.dirname(os.path.abspath(__file__))

# Detect dataset type
dataset_type = "mbpp+"  # Default value
if "kodcode" in base_dir:
    dataset_type = "kodcode"
elif "mbpp" in base_dir:
    dataset_type = "mbpp+"

# Set default paths based on dataset type
if dataset_type == "kodcode":
    default_attacked_log_base = os.path.join(base_dir, "kodcode/attacked_logs_round_3")
    default_attack_suggestion_log_base = os.path.join(
        base_dir, "kodcode/attack_suggestions_round_3"
    )
    default_attack_record_dir = os.path.join(base_dir, "kodcode/attack_records_round_3")
    default_success_log_dir = os.path.join(
        base_dir, "kodcode/attacked_still_succeed_tasks_round_2.json"
    )
else:
    default_attacked_log_base = os.path.join(base_dir, "mbpp+/attacked_logs_round_3")
    default_attack_suggestion_log_base = os.path.join(
        base_dir, "mbpp+/attack_suggestions_round_3"
    )
    default_attack_record_dir = os.path.join(base_dir, "mbpp+/attack_records_round_3")
    default_success_log_dir = os.path.join(
        base_dir, "mbpp+/attacked_still_succeed_tasks_round_2.json"
    )

ATTACKED_LOG_BASE = os.getenv("ATTACKED_LOG_BASE", default_attacked_log_base)
ATTACK_SUGGESTION_LOG_BASE = os.getenv(
    "ATTACK_SUGGESTION_LOG_BASE", default_attack_suggestion_log_base
)
ATTACK_RECORD_DIR = os.getenv("ATTACK_RECORD_DIR", default_attack_record_dir)
WORKSPACE_DIR = os.getenv(
    "WORKSPACE_DIR", os.path.join(os.path.dirname(base_dir), "workspace")
)

# Create necessary directories
os.makedirs(ATTACKED_LOG_BASE, exist_ok=True)
os.makedirs(ATTACK_SUGGESTION_LOG_BASE, exist_ok=True)
os.makedirs(WORKSPACE_DIR, exist_ok=True)
os.makedirs(ATTACK_RECORD_DIR, exist_ok=True)


def get_attacked_log_path(log_name):
    """Dynamically get attack log path"""
    attacked_log_base = os.getenv("ATTACKED_LOG_BASE", ATTACKED_LOG_BASE)
    return os.path.join(attacked_log_base, log_name)


def get_attack_suggestion_log_path(log_name):
    """Dynamically get attack suggestion log path"""
    attack_suggestion_log_base = os.getenv(
        "ATTACK_SUGGESTION_LOG_BASE", ATTACK_SUGGESTION_LOG_BASE
    )
    return os.path.join(attack_suggestion_log_base, log_name)


def get_attack_record_path(log_name):
    """Dynamically get attack record path"""
    attack_record_dir = os.getenv("ATTACK_RECORD_DIR", ATTACK_RECORD_DIR)
    return os.path.join(attack_record_dir, log_name)


def load_previous_attack_analysis(task_id: str) -> List[dict]:
    """Load all round attack analysis history - Backward compatibility function"""
    analyses = []

    # Dynamically get current round and path
    current_round = 3  # Default value
    attack_suggestion_log_base = os.getenv(
        "ATTACK_SUGGESTION_LOG_BASE", ATTACK_SUGGESTION_LOG_BASE
    )

    if "round_2" in attack_suggestion_log_base:
        current_round = 2
    elif "round_1" in attack_suggestion_log_base:
        current_round = 1
    else:
        # Try to extract round from path
        import re

        match = re.search(r"round_(\d+)", attack_suggestion_log_base)
        if match:
            current_round = int(match.group(1))

    print(f"[HISTORY] Current round: {current_round}")

    # Iterate through all previous rounds
    for round_num in range(1, current_round):
        # Build path for this round
        round_path = attack_suggestion_log_base.replace(
            f"round_{current_round}", f"round_{round_num}"
        )
        previous_path = os.path.join(round_path, f"{task_id.replace('/', '_')}.json")

        print(f"[HISTORY] Trying to load Round {round_num} attack analysis: {previous_path}")

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
                            f"[HISTORY] Successfully loaded Round {round_num} attack analysis: step {attack_analysis.get('attack_step', 'Unknown')}"
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

    print(f"[HISTORY] Total loaded {len(analyses)} round historical attack analysis")
    return analyses


async def attack_with_meta_company(
    success_log, retry_count=0, previous_attack_analyses=None
):
    """Use MetaGPT's software company for attack analysis - Supports retry and historical attack analysis"""
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

    # Build attack prompt
    task_id = success_log["question_ID"]

    # Format original task history
    original_history = success_log.get("history", [])
    history_str = ""
    for step in original_history:
        history_str += f"\nStep {step['step']} - {step['name']} ({step['role']}):\n{step['content']}\n"

    # Build historical attack analysis information
    previous_attack_info = ""

    # Prioritize historical analysis information passed by framework
    if previous_attack_analyses is None and "previous_attack_analyses" in success_log:
        previous_attack_analyses = success_log["previous_attack_analyses"]
        print(
            f"[FRAMEWORK_HISTORY] Using historical attack analysis information passed by framework: {len(previous_attack_analyses)} rounds"
        )

    # If no historical information passed by framework, use original load method (backward compatibility)
    if previous_attack_analyses is None:
        print(f"[FALLBACK_HISTORY] Framework did not pass historical information, using original load method")
        previous_attack_analyses = load_previous_attack_analysis(task_id)

    if previous_attack_analyses and len(previous_attack_analyses) > 0:
        print("------previous attack analyses info added------\n")
        previous_attack_info = "IMPORTANT - Previous Attack Analysis History:\n"
        previous_attack_info += f"Multiple attack analyses have been attempted for this task ({len(previous_attack_analyses)} rounds):\n\n"

        for i, prev_attack in enumerate(previous_attack_analyses):
            round_num = prev_attack.get("round", i + 1)
            previous_attack_info += f"Round {round_num}:\n"
            previous_attack_info += (
                f"- Previous attack_step: {prev_attack.get('attack_step', 'Unknown')}\n"
            )
            previous_attack_info += f"- Previous mistake_agent: {prev_attack.get('mistake_agent', 'Unknown')}\n"
            previous_attack_info += f"- Previous original_content: {prev_attack.get('original_content', 'No content provided')}\n"
            previous_attack_info += f"- Previous attack_content: {prev_attack.get('attack_content', 'No attack content provided')}\n"
            previous_attack_info += f"- Previous reason: {prev_attack.get('reason', 'No reason provided')}\n\n"

        # Build excluded steps list
        excluded_steps = [
            str(prev_attack.get("attack_step", ""))
            for prev_attack in previous_attack_analyses
            if prev_attack.get("attack_step")
        ]
        excluded_steps_str = ", ".join(excluded_steps) if excluded_steps else "none"

        previous_attack_info += f"CRITICAL GUIDELINES FOR THIS ATTACK ANALYSIS:\n"
        previous_attack_info += f"1. DO NOT analyze any of the following steps again - they have already been identified and attempted: {excluded_steps_str}\n"
        previous_attack_info += f"2. Look for OTHER steps in the execution history that might contain the actual critical vulnerability.\n"
        previous_attack_info += f"3. Focus on analyzing different aspects of the implementation or different steps in the process.\n"
        previous_attack_info += f"4. Consider if the vulnerability might be in the task understanding, algorithm design, or a different implementation step.\n"
        previous_attack_info += f"5. Analyze the pattern of previous attempts to avoid similar ineffective approaches.\n"

    # Build historical constraint information
    history_constraint = ""
    if previous_attack_analyses and len(previous_attack_analyses) > 0:
        history_constraint = f"""
🚨 CRITICAL CONSTRAINT - YOU MUST FOLLOW THIS RULE:
You are FORBIDDEN from analyzing step(s): {excluded_steps_str}
If you analyze any of these steps again, your analysis will be REJECTED and considered a FAILURE.
You MUST choose a DIFFERENT step that has NOT been analyzed before.

MANDATORY ANALYSIS RULES:
1. You MUST NOT analyze step(s): {excluded_steps_str}
2. You MUST choose a DIFFERENT step that has NOT been analyzed in previous rounds
3. If you repeat any previous analysis, your response will be considered INVALID
4. Focus on steps that have NOT been identified as vulnerable in previous attempts
5. Consider alternative attack vectors and different implementation aspects
6. Look for vulnerabilities in different parts of the code or different agents' contributions

PREVIOUS ATTEMPTS SUMMARY:
"""
        for i, analysis in enumerate(previous_attack_analyses):
            history_constraint += f"- Round {analysis.get('round', i+1)}: Step {analysis.get('attack_step', 'Unknown')} ({analysis.get('mistake_agent', 'Unknown')})\n"

        history_constraint += f"""
YOUR TASK: Analyze a DIFFERENT step than those listed above. Choose step numbers other than: {excluded_steps_str}

VALIDATION CHECK: Before submitting your analysis, verify that your attack_step is NOT in this list: {excluded_steps_str}
If it is, you MUST choose a different step.
"""

    prompt = f"""You are an intelligent attack expert analyzing a successful MetaGPT execution for a programming task.

Your goal is to identify the most critical step in the execution history that, if modified, would lead to an incorrect final result.

Task Information:
Task ID: {task_id}
Question: {success_log['question']}
Ground Truth: {success_log['ground_truth']}
Model Prediction: {success_log['model_prediction']}

{previous_attack_info}

Original Task Execution History:
{history_str}

{history_constraint}

Your attack analysis should be in the following JSON format:
{{
    "attack_step": <step_number>,  // The step number where the attack should be applied
    "mistake_agent": "<the agent that made the mistake>",  // The agent that made the mistake
    "original_content": "<brief description of what this step does>",  // Brief description of the original step
    "attack_content": "<SPECIFIC, IMPLEMENTABLE attack instructions with concrete code changes>",  // SPECIFIC attack instructions
    "reason": "<detailed explanation of why this step is critical and how the attack would work>"  // Detailed explanation
}}

CRITICAL REQUIREMENTS FOR ATTACK_CONTENT:
1. The attack_content MUST be SPECIFIC and IMPLEMENTABLE
2. Include CONCRETE CODE CHANGES or MODIFICATIONS
3. Provide SPECIFIC instructions that can be directly executed
4. Focus on FUNCTIONALITY CHANGES that will cause the program to fail
5. Examples of good attack_content:
   - "Change the return statement from 'return tup + (dct,)' to 'return list(tup) + [dct]'"
   - "Modify the function to return None instead of the tuple"
   - "Add a bug: change 'return tup + (dct,)' to 'return tup + (dct, dct)' (duplicate the dictionary)"
   - "Change the function to ignore the dictionary: 'return tup'"
6. AVOID vague instructions like "return incorrect type" or "modify the function"

Important Guidelines:
1. Focus on identifying the root cause of potential failure, not just any step.
2. The 'attack_step' should be a number corresponding to a step in the implementation process.
3. The 'mistake_agent' should be the agent that made the mistake.
4. The 'original_content' should briefly describe what the step does.
5. The 'attack_content' MUST be SPECIFIC and IMPLEMENTABLE with concrete changes.
6. The 'reason' should be detailed and explain why this step is critical and how the attack would work.
7. Analyze the original task execution history to understand the context and identify where things could go wrong.
8. Focus on steps that involve code generation, implementation, or key algorithmic decisions.

CRITICAL REQUIREMENTS:
1. You MUST create the file FIRST using Editor.create_file
2. You MUST write the content SECOND using Editor.write
3. You MUST use the exact filename: "{task_id.replace('/', '_')}_attack_analysis.json"
4. You MUST NOT use the 'end' command until both file operations are completed
5. You MUST provide the attack analysis in valid JSON format

Step-by-step process:
1. First, create the file:
```json
[
    {{
        "command_name": "Editor.create_file",
        "args": {{
            "filename": "{task_id.replace('/', '_')}_attack_analysis.json"
        }}
    }}
]
```

2. Then, write the attack analysis content:
```json
[
    {{
        "command_name": "Editor.write",
        "args": {{
            "path": "{task_id.replace('/', '_')}_attack_analysis.json",
            "content": "{{"attack_step": "...", "original_content": "...", "attack_content": "SPECIFIC CODE CHANGES HERE", "reason": "..."}}"
        }}
    }}
]
```

3. Only after both file operations are successful, use the end command:
```json
[
    {{
        "command_name": "end"
    }}
]
```

Please analyze the task and provide your attack analysis."""

    try:
        # If it's a retry, add retry information to the prompt
        if retry_count > 0:
            prompt = f"ATTEMPT {retry_count + 1} (RETRY): " + prompt
            print(f"[ATTACK_ANALYSIS] Retry attempt {retry_count + 1} for {task_id}")

        team = await generate_repo(
            idea=prompt,
            n_round=5,
            code_review=True,
            run_tests=False,
            implement=False,
            project_name=f"attack_{task_id.replace('/', '_')}",
            project_path=WORKSPACE_DIR,
            project_dir=WORKSPACE_DIR,
            log_step=log_step,
            use_async=True,
        )

        # 检查攻击分析Result file是否生成
        possible_paths = [
            os.path.join(
                WORKSPACE_DIR, f"{task_id.replace('/', '_')}_attack_analysis.json"
            ),
            os.path.join(
                WORKSPACE_DIR,
                f"attack_{task_id.replace('/', '_')}",
                f"{task_id.replace('/', '_')}_attack_analysis.json",
            ),
            os.path.join(
                WORKSPACE_DIR,
                f"attack_{task_id.replace('/', '_')}",
                "workspace",
                f"{task_id.replace('/', '_')}_attack_analysis.json",
            ),
        ]

        attack_file = None
        for path in possible_paths:
            if os.path.exists(path):
                attack_file = path
                break

        if attack_file:
            with open(attack_file, "r") as f:
                attack_analysis = json.load(f)
                return attack_analysis, history
        else:
            print(f"Warning: Attack analysis file not found for {task_id}")
            print(f"Searched in paths: {possible_paths}")

            # 尝试LoadedHistorical记录中提取攻击分析
            attack_analysis = extract_attack_analysis_from_history(history)
            if attack_analysis:
                print(f"Extracted attack analysis from history for {task_id}")
                return attack_analysis, history

            return None, None

    except Exception as e:
        print(f"Attack analysis failed: {e}")
        return None, None


def extract_attack_analysis_from_history(history):
    """LoadedHistorical记录中提取攻击分析"""
    for step in reversed(history):
        content = step.get("content", "")

        # 查找JSON格式的攻击分析
        if '"attack_step"' in content and '"attack_content"' in content:
            try:
                # 提取JSON部分
                start_idx = content.find("{")
                end_idx = content.rfind("}") + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = content[start_idx:end_idx]
                    attack_analysis = json.loads(json_str)

                    # 验证必要的字段
                    required_fields = [
                        "attack_step",
                        "original_content",
                        "attack_content",
                        "reason",
                    ]
                    if all(field in attack_analysis for field in required_fields):
                        return attack_analysis
            except json.JSONDecodeError:
                continue

    return None


async def generate_attacked_solution_with_advanced_monitoring(
    success_log, attack_analysis
):
    """使用高级监控器生成攻击后的解决方案"""
    history = []

    # 创建统一的步骤计数器 - 使用类来管理状态
    class StepCounter:
        def __init__(self):
            self.step = 0
            self.roles = {}  # 存储所有Role实例的引用

        def increment(self):
            self.step += 1
            # 同步所有Role实例的步骤计数器
            for role in self.roles.values():
                if hasattr(role, "step_counter"):
                    role.step_counter = self.step

        def get_step(self):
            return self.step

        def reset(self):
            """重置步骤计数器"""
            self.step = 0
            # 同步所有Role实例的步骤计数器
            for role in self.roles.values():
                if hasattr(role, "step_counter"):
                    role.step_counter = 0

    step_counter = StepCounter()

    # 重置步骤计数器，确保每次新的Task执行都Loaded0开始
    step_counter.reset()

    # 添加调试信息
    print(f"[DEBUG] 步骤计数器重置: {step_counter.get_step()}")

    # 创建高级攻击监控器
    monitor = AdvancedAttackMonitor(
        success_log["question_ID"], attack_analysis, original_success_log=success_log
    )
    interceptor = PromptInjectionInterceptor(monitor)

    def log_step(content, role, name):
        print(f"[LOG_STEP] {role} {name}: {content}")

        # 标准化职责名称
        standardized_name = standardize_role_name(name)

        # 检查是否是terminal output，如果是则合并到上一个step
        if role == "Terminal" and content.startswith("Terminal output:"):
            # 将terminal output合并到上一个step
            if history:
                last_step = history[-1]
                # incontent中添加terminal output
                last_step["content"] += f"\n\n{content}"
                print(
                    f"[LOG_STEP] Terminal output merged into previous step: {last_step['step']}"
                )
            else:
                # 如果没有上一个step，创建一个新的step
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

        # 记录监控事件
        monitor.log_event("step", role, standardized_name, content)

        # 检查是否检测到攻击步骤的前一步 - 使用新的waiting机制
        if not monitor.detected_pre_attack_step:
            # 使用同步版本的检测方法
            if monitor.check_pre_attack_step(content, role, standardized_name):
                # 记录实际检测到的前一步step编号（当前步骤就是前一步）
                current_step = step_counter.get_step()
                monitor.record_pre_attack_detection(current_step)
                print(
                    f"[ADVANCED_ATTACK_MONITOR] 检测到攻击步骤 {monitor.attack_step} 的前一步! (实际step: {current_step})"
                )
                print(f"[ADVANCED_ATTACK_MONITOR] 准备in下一步注入攻击prompt...")

                # 新增：设置waiting注入状态
                monitor.injection_pending = True
                monitor.injection_completed = False
                print(
                    f"[ADVANCED_ATTACK_MONITOR] 设置waiting注入状态，系统将waiting注入完成..."
                )

        # 先记录当前步骤
        current_step = step_counter.get_step()
        history.append(
            {
                "step": current_step,
                "content": content,
                "role": role,
                "name": standardized_name,
            }
        )
        # 增加步骤计数器
        step_counter.increment()

        # 同步所有Role实例的步骤计数器（仅in合理范围内）
        current_step = step_counter.get_step()
        if current_step <= 1000:  # 只in合理范围内同步
            for r in step_counter.roles.values():
                if hasattr(r, "step_counter"):
                    r.step_counter = current_step
        else:
            # 如果步骤计数器异常，重置它
            print(f"[DEBUG] 检测到步骤计数器异常值: {current_step}，正in重置...")
            step_counter.reset()
            for r in step_counter.roles.values():
                if hasattr(r, "step_counter"):
                    r.step_counter = 0

        # 同步监控器的步骤计数器
        monitor.update_step(step_counter.get_step())

    # 构建攻击后的解决方案生成提示
    task_id = success_log["question_ID"]

    prompt = f"""Complete the following programming task, with a multi-agent team cooperation
    the solution file you write should be named as "{task_id.replace('/', '_')}_attacked_solution.py":

Task ID: {task_id}
Question: {success_log['question']}

"""

    try:
        # 修改：创建一个自定义的generate_repo函数来获取Role实例
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

        # 创建自定义的generate_repo函数
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
                    attack_monitor=kwargs.get("attack_monitor"),
                ),
                ProductManager(
                    project_dir=kwargs.get("project_dir"),
                    log_step=kwargs.get("log_step"),
                    attack_monitor=kwargs.get("attack_monitor"),
                ),
                Architect(
                    project_dir=kwargs.get("project_dir"),
                    log_step=kwargs.get("log_step"),
                    attack_monitor=kwargs.get("attack_monitor"),
                ),
                Engineer2(
                    project_dir=kwargs.get("project_dir"),
                    log_step=kwargs.get("log_step"),
                    attack_monitor=kwargs.get("attack_monitor"),
                ),
                DataAnalyst(
                    project_dir=kwargs.get("project_dir"),
                    log_step=kwargs.get("log_step"),
                    attack_monitor=kwargs.get("attack_monitor"),
                ),
            ]

            company.hire(roles)

            # 将Role实例注册到步骤计数器
            for role in roles:
                step_counter.roles[role.name] = role
                # 确保attack_monitor也被正确设置
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
            project_name=f"attack_solution_{task_id.replace('/', '_')}",
            project_path=WORKSPACE_DIR,
            project_dir=WORKSPACE_DIR,
            log_step=log_step,
            use_async=True,
            attack_monitor=monitor,
        )

        # 检查攻击后的解决方案是否生成
        possible_paths = [
            os.path.join(
                WORKSPACE_DIR, f"{task_id.replace('/', '_')}_attacked_solution.py"
            ),
            os.path.join(
                WORKSPACE_DIR,
                f"attack_solution_{task_id.replace('/', '_')}",
                f"{task_id.replace('/', '_')}_attacked_solution.py",
            ),
            os.path.join(
                WORKSPACE_DIR,
                f"attack_solution_{task_id.replace('/', '_')}",
                "workspace",
                f"{task_id.replace('/', '_')}_attacked_solution.py",
            ),
        ]

        solution_file = None
        for path in possible_paths:
            if os.path.exists(path):
                solution_file = path
                break

        if solution_file:
            with open(solution_file, "r") as f:
                attacked_solution = f.read()

                # 判断攻击是否Success
                monitor.attack_successful = (
                    attacked_solution != success_log["model_prediction"]
                )

                return attacked_solution, history, monitor
        else:
            print(f"Warning: Attacked solution file not found for {task_id}")
            print(f"Searched in paths: {possible_paths}")

            # 尝试LoadedHistorical记录中提取解决方案
            solution = extract_solution_from_history(history)
            if solution:
                print(f"Extracted solution from history for {task_id}")
                monitor.attack_successful = solution != success_log["model_prediction"]
                return solution, history, monitor

            return None, None, monitor

    except Exception as e:
        print(f"Attacked solution generation failed: {e}")
        return None, None, monitor


def extract_solution_from_history(history):
    """LoadedHistorical记录中提取解决方案"""
    for step in reversed(history):
        content = step.get("content", "")

        # 查找Python代码
        if "def " in content or "import " in content:
            # 提取代码部分
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


async def process_success_log(success_log):
    """处理单个Success的log文件 - 被universal框架调用"""
    log_name = f"{success_log['question_ID'].replace('/', '_')}.json"
    task_id = success_log["question_ID"]

    # 断点续跑：检查是否已完成所有步骤
    attacked_path = get_attacked_log_path(log_name)
    attack_suggestion_path = get_attack_suggestion_log_path(log_name)
    attack_record_path = get_attack_record_path(log_name)

    # 检查是否所有必需的文件都存in
    all_files_exist = (
        os.path.exists(attacked_path)
        and os.path.exists(attack_suggestion_path)
        and os.path.exists(attack_record_path)
    )

    # 修改Skipped逻辑：只有inRound一round时才Skipped，Round二round应该重新生成攻击后的解决方案
    # in运行时动态获取当前round次
    current_round = 3  # 默认值
    attack_suggestion_log_base = os.getenv(
        "ATTACK_SUGGESTION_LOG_BASE", ATTACK_SUGGESTION_LOG_BASE
    )

    if "round_2" in attack_suggestion_log_base:
        current_round = 2
    elif "round_1" in attack_suggestion_log_base:
        current_round = 1
    else:
        # 尝试Loaded路径中提取round次
        import re

        match = re.search(r"round_(\d+)", attack_suggestion_log_base)
        if match:
            current_round = int(match.group(1))

    # 只有inRound一round且所有文件都存in时才Skipped
    if current_round == 1 and all_files_exist:
        print(
            f"[INFO] All logs for task {task_id} already exist in round 1, skipping..."
        )
        return

    # Failed to loadprevious attack analysis
    # 优先使用框架传递的Historical分析信息
    if "previous_attack_analyses" in success_log:
        previous_attack_analyses = success_log["previous_attack_analyses"]
        print(
            f"[FRAMEWORK_HISTORY] 使用框架传递的Historical攻击分析信息: {len(previous_attack_analyses)} round"
        )
    else:
        # 向后兼容：使用原有的Failed to load方式
        previous_attack_analyses = load_previous_attack_analysis(task_id)
        print(f"[FALLBACK_HISTORY] 使用原有方式Failed to loadHistorical攻击分析信息")

    if previous_attack_analyses:
        print(
            f"[PREVIOUS_ATTACK] Loaded previous attack analyses for {task_id} from {len(previous_attack_analyses)} rounds."
        )
        for i, prev_analysis in enumerate(previous_attack_analyses):
            print(f"  Round {i+1}: step {prev_analysis.get('attack_step', 'Unknown')}")
    else:
        print(f"[PREVIOUS_ATTACK] No previous attack analyses found for {task_id}")

    # 1. 攻击分析
    print(f"[ATTACK_ANALYSIS] Starting attack analysis for {task_id}")
    attack_analysis, attack_history = await attack_with_meta_company(
        success_log, 0, previous_attack_analyses
    )

    if attack_analysis is None:
        print(f"Attack analysis failed for {log_name}")
        return

    print(f"[ATTACK_ANALYSIS] Generated attack analysis for {task_id}")

    # 保存攻击分析结果
    with open(attack_suggestion_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "attack_analysis": attack_analysis,
                "history": attack_history,
                "previous_attack_analysis": previous_attack_analyses,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[ATTACK_ANALYSIS] Attack analysis completed for {task_id}")

    # 2. 生成攻击后的解决方案（带高级监控）
    print(
        f"[ADVANCED_ATTACK_IMPLEMENTATION] Starting advanced attack implementation for {success_log['question_ID']}"
    )
    attacked_solution, attacked_history, monitor = (
        await generate_attacked_solution_with_advanced_monitoring(
            success_log, attack_analysis
        )
    )

    if attacked_solution is None:
        print(f"Attacked solution generation failed for {log_name}")
        # 即使failed也要保存监控记录
        with open(attack_record_path, "w", encoding="utf-8") as f:
            json.dump(monitor.get_attack_record(), f, ensure_ascii=False, indent=2)
        return

    # 保存攻击后的解决方案and日志
    with open(attacked_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "question": success_log["question"],
                "question_ID": success_log["question_ID"],
                "ground_truth": success_log["ground_truth"],
                "model_prediction": attacked_solution,
                "history": attacked_history,
                "attack_analysis": attack_analysis,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 保存攻击监控记录
    with open(attack_record_path, "w", encoding="utf-8") as f:
        json.dump(monitor.get_attack_record(), f, ensure_ascii=False, indent=2)

    print(
        f"[ADVANCED_ATTACK_RESULT] Task {success_log['question_ID']}: "
        f"Detected={monitor.detected_pre_attack_step}, "
        f"Injected={monitor.injected_prompt}, "
        f"Successful={monitor.attack_successful}"
    )

    # 显示注入Historical
    if monitor.injection_history:
        print(f"[INJECTION_HISTORY] 注入Historical:")
        for i, injection in enumerate(monitor.injection_history):
            print(f"  {i+1}. 策略: {injection['strategy']}, 步骤: {injection['step']}")
