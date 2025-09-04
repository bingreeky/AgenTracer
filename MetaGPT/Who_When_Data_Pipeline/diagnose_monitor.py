#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagnosis Attack Monitor - Specifically for dynamic injection in diagnosis + correction process
Based on AdvancedAttackMonitor, but optimized for diagnosis scenarios
"""

import time
import asyncio
from datetime import datetime
from typing import Dict


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


class DiagnosisAttackMonitor:
    """Diagnosis Attack Monitor, supports dynamic injection of correction information in diagnosis process"""

    def __init__(self, task_id: str, diagnosis: dict, original_failed_log: dict = None):
        self.task_id = task_id
        self.diagnosis = diagnosis
        self.mistake_step = diagnosis.get("mistake_step", -1)
        self.suggested_fix = diagnosis.get("suggested_fix", "")
        self.reason = diagnosis.get("reason", "")

        # Original failed log, used for dynamic generation of matching indicators
        self.original_failed_log = original_failed_log

        # Monitoring status
        self.detected_pre_mistake_step = False  # Detected the step before error step
        self.injected_correction = False
        self.correction_successful = False
        self.monitor_log = []
        self.injection_history = []

        # Step counter synchronization
        self.current_step = 0
        self.last_detected_step = -1
        self.detection_history = []

        # Actual correction records
        self.actual_correction_step = -1  # Actual step number that was corrected
        self.actual_pre_mistake_step = -1  # Actual detected previous step number

        # Waiting mechanism related
        self.injection_pending = False
        self.injection_completed = False
        self.injection_lock = asyncio.Lock()

        # Dynamically generated matching indicators
        self.dynamic_pre_mistake_indicators = (
            self._generate_dynamic_pre_mistake_indicators()
        )

        # Injection strategies
        self.injection_strategies = {
            "correction_injection": self._generate_correction_injection_prompt,
        }

    def _generate_dynamic_pre_mistake_indicators(self) -> list:
        """Dynamically generate previous step detection indicators based on original failed log"""
        indicators = []

        if self.original_failed_log:
            # Extract previous step specific content from original log
            mistake_step = self.mistake_step

            # Handle unified format data structure
            if "original_log" in self.original_failed_log:
                # New format: data in original_log
                original_history = self.original_failed_log["original_log"].get(
                    "history", []
                )
            else:
                # Old format: data directly in original_failed_log
                original_history = self.original_failed_log.get("history", [])

            if mistake_step > 1 and len(original_history) >= mistake_step - 1:
                # Get previous step content from original log
                pre_mistake_step_data = original_history[mistake_step - 2]
                pre_mistake_content = pre_mistake_step_data.get("content", "")
                pre_mistake_role = pre_mistake_step_data.get("role", "")
                pre_mistake_name = pre_mistake_step_data.get("name", "")

                print(
                    f"[DIAGNOSIS_MATCHING] Original previous step content: {pre_mistake_content[:100]}..."
                )
                print(
                    f"[DIAGNOSIS_MATCHING] Original previous step role: {pre_mistake_role} - {pre_mistake_name}"
                )

                # Generate precise matching indicators based on original content
                if pre_mistake_content:
                    # 1. Complete content matching (fuzzy matching)
                    indicators.append(
                        lambda content: self._fuzzy_content_match(
                            content, pre_mistake_content
                        )
                    )

                    # 2. Keyword matching
                    keywords = self._extract_keywords(pre_mistake_content)
                    if keywords:
                        indicators.append(
                            lambda content: self._keyword_match(content, keywords)
                        )

                    # 3. Semantic similarity matching
                    indicators.append(
                        lambda content: self._semantic_similarity_match(
                            content, pre_mistake_content
                        )
                    )

                    # 4. Specific pattern matching (based on role type)
                    if "leader" in pre_mistake_role.lower():
                        indicators.append(
                            lambda content: "assign" in content.lower()
                            or "task" in content.lower()
                        )
                    elif "manager" in pre_mistake_role.lower():
                        indicators.append(
                            lambda content: "requirement" in content.lower()
                            or "define" in content.lower()
                        )
                    elif "architect" in pre_mistake_role.lower():
                        indicators.append(
                            lambda content: "design" in content.lower()
                            or "architecture" in content.lower()
                        )
                    elif "engineer" in pre_mistake_role.lower():
                        indicators.append(
                            lambda content: "implement" in content.lower()
                            or "code" in content.lower()
                            or "function" in content.lower()
                        )

        return indicators

    def _fuzzy_content_match(
        self, content: str, original_content: str, threshold: float = 0.4
    ) -> bool:
        """Fuzzy content matching"""
        import difflib

        similarity = difflib.SequenceMatcher(
            None, content.lower(), original_content.lower()
        ).ratio()
        return similarity >= threshold

    def _extract_keywords(self, content: str) -> list:
        """Extract keywords from content"""
        import re

        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
        }
        words = re.findall(r"\b[a-zA-Z]+\b", content.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        return keywords[:10]

    def _keyword_match(
        self, content: str, keywords: list, min_matches: int = 1
    ) -> bool:
        """Keyword matching"""
        content_lower = content.lower()
        matches = sum(1 for keyword in keywords if keyword in content_lower)
        return matches >= min_matches

    def _semantic_similarity_match(
        self, content: str, original_content: str, threshold: float = 0.2
    ) -> bool:
        """Semantic similarity matching"""
        content_words = set(content.lower().split())
        original_words = set(original_content.lower().split())
        if not original_words:
            return False
        overlap = len(content_words.intersection(original_words))
        similarity = overlap / len(original_words)
        return similarity >= threshold

    def log_event(self, event_type: str, role: str, name: str, content: str):
        """Record monitoring events"""
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "role": role,
            "name": name,
            "content": content[:200] + "..." if len(content) > 200 else content,
        }
        self.monitor_log.append(event)

    def update_step(self, step_number: int):
        """Update current step"""
        old_step = self.current_step
        self.current_step = step_number
        if old_step != step_number:
            print(f"[DIAGNOSIS_MONITOR] Step counter updated: {old_step} -> {step_number}")

    def check_pre_mistake_step(
        self, step_content: str, role: str = "", name: str = ""
    ) -> bool:
        """Check if it's the step before error step"""
        if self.detected_pre_mistake_step:
            print(
                f"[DIAGNOSIS_MONITOR] Skipped previous step detection: Already detected previous step (step {self.actual_pre_mistake_step})"
            )
            return False

        # Check if current step is close to error step (at most 2 steps ahead)
        if hasattr(self, "current_step") and self.current_step < self.mistake_step - 2:
            return False

        # Avoid duplicate detection of same content
        content_hash = self._content_hash(role, name, step_content)
        if content_hash in self.detection_history:
            print(f"[DIAGNOSIS_MONITOR] Skipped duplicate detection: {role} {name}")
            return False

        # Use same logic as error step detection
        is_pre_mistake = self._check_content_matches_pre_mistake_step(
            step_content, role, name
        )

        if is_pre_mistake:
            # Record detection history
            self.detection_history.append(content_hash)
            self.last_detected_step = self.current_step
            print(
                f"[DIAGNOSIS_MONITOR] Detected previous step: Step {self.current_step}, Role {role}, Name {name}"
            )
            return True

        return False

    def _check_content_matches_pre_mistake_step(
        self, content: str, role: str, name: str
    ) -> bool:
        """Check if current step content matches previous step features"""
        print(
            f"[DIAGNOSIS_MONITOR] Check previous step content matching: Role {role}, Name {name}, Content length {len(content)}"
        )

        if not self.dynamic_pre_mistake_indicators:
            print(f"[DIAGNOSIS_MONITOR] No dynamic previous step indicators, using semantic matching")
            if self.original_failed_log:
                mistake_step = self.mistake_step
                original_history = self.original_failed_log.get("log_data", {}).get(
                    "history", []
                )
                if mistake_step > 1 and len(original_history) >= mistake_step - 1:
                    pre_mistake_content = original_history[mistake_step - 2].get(
                        "content", ""
                    )
                    return self._improved_semantic_match(
                        content, pre_mistake_content, threshold=0.1
                    )
            return False

        match_count = 0
        total_indicators = len(self.dynamic_pre_mistake_indicators)

        for i, indicator in enumerate(self.dynamic_pre_mistake_indicators):
            try:
                if indicator(content):
                    match_count += 1
                    print(f"[DIAGNOSIS_MONITOR] Previous step indicator {i+1} matched successfully")
                else:
                    print(f"[DIAGNOSIS_MONITOR] Previous step indicator {i+1} matching failed")
            except Exception as e:
                print(f"[DIAGNOSIS_MONITOR] Previous step indicator {i+1} execution exception: {e}")

        match_ratio = match_count / total_indicators if total_indicators > 0 else 0
        print(
            f"[DIAGNOSIS_MONITOR] Previous step matching ratio: {match_count}/{total_indicators} = {match_ratio:.2f}"
        )

        if match_ratio >= 0.25:
            print(f"[DIAGNOSIS_MONITOR] ✅ Content matches previous step features: Role {role}, Name {name}")
            return True
        else:
            print(f"[DIAGNOSIS_MONITOR] Previous step indicator matching failed, trying semantic matching")
            if self.original_failed_log:
                mistake_step = self.mistake_step
                original_history = self.original_failed_log.get("log_data", {}).get(
                    "history", []
                )
                if mistake_step > 1 and len(original_history) >= mistake_step - 1:
                    pre_mistake_content = original_history[mistake_step - 2].get(
                        "content", ""
                    )
                    if self._improved_semantic_match(
                        content, pre_mistake_content, threshold=0.15
                    ):
                        print(
                            f"[DIAGNOSIS_MONITOR] ✅ Previous step semantic matching successful: Role {role}, Name {name}"
                        )
                        return True

            print(
                f"[DIAGNOSIS_MONITOR] ❌ Content does not match previous step features: Role {role}, Name {name}"
            )
            return False

    def _content_hash(self, role: str, name: str, content: str) -> int:
        """Generate combined hash of role+name+content"""
        content_preview = content[:300] if content else ""
        hash_string = f"{role}|{name}|{content_preview}"
        return hash(hash_string)

    def should_inject_at_step(
        self, step_number: int, step_content: str = "", role: str = "", name: str = ""
    ) -> bool:
        """Determine if should inject correction information in current step"""
        self.update_step(step_number)

        print(
            f"[DIAGNOSIS_INJECTION] Injection decision: step_number={step_number}, detected_pre_mistake_step={self.detected_pre_mistake_step}, actual_pre_mistake_step={self.actual_pre_mistake_step}, current_step={self.current_step}, role={role}, name={name}"
        )

        # First check if previous step has been detected - this is a necessary condition
        if not self.detected_pre_mistake_step:
            print(f"[DIAGNOSIS_INJECTION] ❌ Not injecting: Previous step not detected")
            return False

        # Check if already injected
        if self.injected_correction:
            print(f"[DIAGNOSIS_INJECTION] ❌ Not injecting: Already injected, avoiding duplicate injection")
            return False

        # Check if previous step number is recorded
        if self.actual_pre_mistake_step == -1:
            print(f"[DIAGNOSIS_INJECTION] ❌ Not injecting: Previous step number not recorded")
            return False

        # Check if current step is greater than previous step
        if step_number <= self.actual_pre_mistake_step:
            print(
                f"[DIAGNOSIS_INJECTION] ❌ Not injecting: Current step ({step_number}) is not greater than previous step ({self.actual_pre_mistake_step})"
            )
            return False

        # Check if it's a think step
        is_think_step = (
            "thinking" in step_content.lower()
            or "thinking:" in step_content.lower()
            or step_content.strip().startswith("thinking")
        )

        if not is_think_step:
            print(f"[DIAGNOSIS_INJECTION] ❌ Not injecting: Current step is not a think step")
            return False

        # Check if it's a terminalOutput step
        is_terminal_output = (
            "terminal output:" in step_content.lower()
            or "command output:" in step_content.lower()
            or "command executed:" in step_content.lower()
        )

        if is_terminal_output:
            print(f"[DIAGNOSIS_INJECTION] ❌ 不注入: 当前步骤是terminalOutput，Skipped")
            return False

        # 基于diagnosis内容判断是否应该in此步骤注入
        should_inject_based_on_content = self._should_inject_based_on_diagnosis(
            step_content, role, name
        )

        if not should_inject_based_on_content:
            print(
                f"[DIAGNOSIS_INJECTION] ❌ 不注入: 基于diagnosis内容判断不应该in此步骤注入"
            )
            return False

        # 所有条件都满足，可以注入
        self.actual_correction_step = step_number
        print(
            f"[DIAGNOSIS_INJECTION] ✅ 准备in步骤 {step_number} 注入纠错信息 (前一步: {self.actual_pre_mistake_step}, 当前Role: {name})"
        )
        print(
            f"[DIAGNOSIS_INJECTION] 纠错信息: suggested_fix='{self.suggested_fix[:50]}...'"
        )
        return True

    def _improved_semantic_match(
        self, content: str, original_content: str, threshold: float = 0.25
    ) -> bool:
        """改进的语义匹配"""
        if not original_content:
            return False

        content_tokens = self._tokenize_content(content)
        original_tokens = self._tokenize_content(original_content)

        if not original_tokens:
            return False

        overlap = len(content_tokens.intersection(original_tokens))
        similarity = overlap / len(original_tokens)

        print(
            f"[DIAGNOSIS_INJECTION] 语义匹配: 重叠词汇{overlap}/{len(original_tokens)} = {similarity:.3f}"
        )

        return similarity >= threshold

    def _should_inject_based_on_diagnosis(
        self, step_content: str, role: str, name: str
    ) -> bool:
        """基于diagnosis判断是否应该in此步骤注入"""
        if not self.diagnosis:
            return True

        mistake_step = self.diagnosis.get("mistake_step", -1)
        reason = self.diagnosis.get("reason", "")
        suggested_fix = self.diagnosis.get("suggested_fix", "")

        # 如果diagnosis指定了具体的Task内容，检查当前步骤内容是否相关
        if reason or suggested_fix:
            content_lower = step_content.lower()
            reason_lower = reason.lower()
            fix_lower = suggested_fix.lower()

            # 提取关键词进行匹配
            keywords = self._extract_keywords(reason_lower + " " + fix_lower)
            if keywords:
                matches = sum(1 for keyword in keywords if keyword in content_lower)
                if matches >= 1:
                    print(
                        f"[DIAGNOSIS_INJECTION] ✅ 当前步骤内容匹配diagnosis关键词({matches}个)"
                    )
                    return True

            # 语义相似度匹配
            if self._improved_semantic_match(
                content_lower, reason_lower, threshold=0.1
            ) or self._improved_semantic_match(content_lower, fix_lower, threshold=0.1):
                print(f"[DIAGNOSIS_INJECTION] ✅ 当前步骤内容与diagnosis语义相似")
                return True

        # 如果没有明确的匹配，但当前步骤包含代码生成相关的内容，也考虑注入
        code_generation_indicators = [
            "create",
            "write",
            "implement",
            "function",
            "def ",
            "class ",
            "return",
            "import",
            "file",
            "code",
            "program",
            "algorithm",
            "fix",
            "correct",
        ]

        content_lower = step_content.lower()
        code_indicators = sum(
            1 for indicator in code_generation_indicators if indicator in content_lower
        )

        if code_indicators >= 2:
            print(
                f"[DIAGNOSIS_INJECTION] ✅ 当前步骤包含代码生成相关内容({code_indicators}个指标)"
            )
            return True

        print(f"[DIAGNOSIS_INJECTION] ❌ 当前步骤内容与diagnosis不匹配")
        return False

    def _tokenize_content(self, content: str) -> set:
        """分词处理"""
        import re

        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "thinking",
            "will",
            "create",
            "file",
            "and",
            "then",
            "write",
            "content",
            "function",
        }
        words = re.findall(r"\b[a-zA-Z0-9-]+\b", content.lower())
        meaningful_words = {
            word for word in words if word not in stop_words and len(word) > 2
        }
        return meaningful_words

    def record_pre_mistake_detection(self, step_number: int):
        """记录检测到前一步的step编号"""
        self.actual_pre_mistake_step = step_number
        self.detected_pre_mistake_step = True
        print(f"[DIAGNOSIS_MONITOR] 记录前一步检测: 步骤{step_number}")
        print(
            f"[DIAGNOSIS_MONITOR] 状态更新: detected_pre_mistake_step=True, actual_pre_mistake_step={step_number}"
        )

    def _generate_correction_injection_prompt(self) -> str:
        """生成纠错注入prompt - in检测到前一步时注入更具体的纠错信息"""
        return f"""
🚨 CRITICAL CORRECTION ALERT - SPECIFIC IMPLEMENTATION GUIDANCE 🚨

You are currently approaching step {self.mistake_step}, which is where a critical error was identified in the previous implementation.

PREVIOUS ERROR ANALYSIS:
- Error occurred at: Step {self.mistake_step}
- Root cause: {self.reason}
- Recommended fix: {self.suggested_fix}

IMMEDIATE ACTION REQUIRED:
1. Review your current implementation approach
2. Ensure you are NOT making the same mistake identified in step {self.mistake_step}
3. Apply the suggested fix: {self.suggested_fix}
4. Double-check your logic before proceeding

SPECIFIC GUIDANCE:
{self.suggested_fix}

CRITICAL: This is not just a suggestion - it's a specific correction for a known error. Please implement the fix as described above.
"""

    def generate_correction_prompt(self, strategy: str = "correction_injection") -> str:
        """生成纠错prompt"""
        if strategy in self.injection_strategies:
            return self.injection_strategies[strategy]()
        else:
            return self._generate_correction_injection_prompt()

    def inject_correction(self, strategy: str = "correction_injection") -> Dict:
        """执行纠错注入"""
        prompt = self.generate_correction_prompt(strategy)

        injection_record = {
            "timestamp": time.time(),
            "strategy": strategy,
            "prompt": prompt,
            "step": self.current_step,
            "success": True,
        }

        self.injection_history.append(injection_record)
        self.injected_correction = True

        return injection_record

    def get_diagnosis_record(self) -> dict:
        """获取诊断记录"""
        return {
            "task_id": self.task_id,
            "diagnosis": self.diagnosis,
            "detected_pre_mistake_step": self.detected_pre_mistake_step,
            "injected_correction": self.injected_correction,
            "correction_successful": self.correction_successful,
            "monitor_log": self.monitor_log,
            "injection_history": self.injection_history,
            "actual_correction_step": self.actual_correction_step,
            "actual_pre_mistake_step": self.actual_pre_mistake_step,
            "expected_mistake_step": self.mistake_step,
            "timestamp": datetime.now().isoformat(),
        }


class DiagnosisInjectionInterceptor:
    """诊断注入拦截器"""

    def __init__(self, monitor: DiagnosisAttackMonitor):
        self.monitor = monitor
        self.injection_points = []

    def intercept_prompt(
        self, original_prompt: str, role: str, step_number: int
    ) -> str:
        """拦截并修改prompt"""
        if self.monitor.should_inject_at_step(step_number):
            print(f"[DIAGNOSIS_INJECTION] in步骤 {step_number} 注入纠错信息!")

            strategy = self._select_injection_strategy(role, step_number)
            injection_record = self.monitor.inject_correction(strategy)

            modified_prompt = self._modify_prompt(
                original_prompt, injection_record["prompt"]
            )

            self.injection_points.append(
                {
                    "step": step_number,
                    "role": role,
                    "strategy": strategy,
                    "original_length": len(original_prompt),
                    "modified_length": len(modified_prompt),
                }
            )

            return modified_prompt

        return original_prompt

    def _select_injection_strategy(self, role: str, step_number: int) -> str:
        """选择注入策略"""
        return "correction_injection"

    def _modify_prompt(self, original_prompt: str, injection_prompt: str) -> str:
        """修改原始prompt"""
        return f"{injection_prompt}\n\n{original_prompt}"


def create_diagnosis_monitoring_log_step(
    monitor: DiagnosisAttackMonitor, interceptor: DiagnosisInjectionInterceptor
):
    """创建支持纠错注入的log_step函数"""

    def log_step(content: str, role: str, name: str):
        # 标准化职责名称
        standardized_name = standardize_role_name(name)

        # 检查是否是terminal output，如果是则合并到上一个step
        if role == "Terminal" and content.startswith("Terminal output:"):
            # 这里我们只记录事件，不创建新的step
            # terminal output会inuniversal framework的log_step函数中处理
            monitor.log_event("terminal_output", role, standardized_name, content)
            return

        monitor.log_event("step", role, standardized_name, content)

        if not monitor.detected_pre_mistake_step and monitor.check_pre_mistake_step(
            content, role, standardized_name
        ):
            monitor.record_pre_mistake_detection(monitor.current_step)
            print(
                f"[DIAGNOSIS_MONITOR] 检测到Error步骤 {monitor.mistake_step} 的前一步!"
            )
            print(f"[DIAGNOSIS_MONITOR] 准备in下一步注入纠错信息...")

    return log_step


class DiagnosisMonitoringConfig:
    """诊断监控配置"""

    def __init__(self):
        self.enable_real_injection = True
        self.injection_strategies = ["correction_injection"]
        self.detection_threshold = 0.7
        self.max_injections_per_task = 1
        self.log_detailed_events = True
