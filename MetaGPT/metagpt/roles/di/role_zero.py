from __future__ import annotations

import inspect
import json
import re
import traceback
from datetime import datetime
from typing import Annotated, Callable, Dict, List, Literal, Optional, Tuple
from pathlib import Path

from pydantic import Field, model_validator

from metagpt.actions import Action, UserRequirement
from metagpt.actions.di.run_command import RunCommand
from metagpt.actions.search_enhanced_qa import SearchEnhancedQA
from metagpt.const import IMAGES, DEFAULT_WORKSPACE_ROOT
from metagpt.exp_pool import exp_cache
from metagpt.exp_pool.context_builders import RoleZeroContextBuilder
from metagpt.exp_pool.serializers import RoleZeroSerializer
from metagpt.logs import logger
from metagpt.memory.role_zero_memory import RoleZeroLongTermMemory
from metagpt.prompts.di.role_zero import (
    ASK_HUMAN_COMMAND,
    ASK_HUMAN_GUIDANCE_FORMAT,
    CMD_PROMPT,
    DETECT_LANGUAGE_PROMPT,
    END_COMMAND,
    JSON_REPAIR_PROMPT,
    QUICK_RESPONSE_SYSTEM_PROMPT,
    QUICK_THINK_EXAMPLES,
    QUICK_THINK_PROMPT,
    QUICK_THINK_SYSTEM_PROMPT,
    QUICK_THINK_TAG,
    REGENERATE_PROMPT,
    REPORT_TO_HUMAN_PROMPT,
    ROLE_INSTRUCTION,
    SUMMARY_PROBLEM_WHEN_DUPLICATE,
    SUMMARY_PROMPT,
    SYSTEM_PROMPT,
)
from metagpt.roles import Role
from metagpt.schema import AIMessage, Message, UserMessage
from metagpt.strategy.experience_retriever import DummyExpRetriever, ExpRetriever
from metagpt.strategy.planner import Planner
from metagpt.tools.libs.browser import Browser
from metagpt.tools.libs.editor import Editor
from metagpt.tools.tool_recommend import BM25ToolRecommender, ToolRecommender
from metagpt.tools.tool_registry import register_tool
from metagpt.utils.common import CodeParser, any_to_str, extract_and_encode_images
from metagpt.utils.repair_llm_raw_output import (
    RepairType,
    repair_escape_error,
    repair_llm_raw_output,
)
from metagpt.utils.report import ThoughtReporter
import os


@register_tool(include_functions=["ask_human", "reply_to_human"])
class RoleZero(Role):
    """A role who can think and act dynamically"""

    # Basic Info
    name: str = "Zero"
    profile: str = "RoleZero"
    goal: str = ""
    system_msg: Optional[list[str]] = (
        None  # Use None to conform to the default value at llm.aask
    )
    system_prompt: str = (
        SYSTEM_PROMPT  # Use None to conform to the default value at llm.aask
    )
    cmd_prompt: str = CMD_PROMPT
    cmd_prompt_current_state: str = ""
    instruction: str = ROLE_INSTRUCTION
    task_type_desc: Optional[str] = None

    # React Mode
    react_mode: Literal["react"] = "react"
    max_react_loop: int = 50  # used for react mode

    # Tools
    tools: list[str] = (
        []
    )  # Use special symbol ["<all>"] to indicate use of all registered tools
    tool_recommender: Optional[ToolRecommender] = None
    tool_execution_map: Annotated[dict[str, Callable], Field(exclude=True)] = {}
    special_tool_commands: list[str] = [
        "Plan.finish_current_task",
        "end",
        "Terminal.run_command",
        "RoleZero.ask_human",
    ]
    # List of exclusive tool commands.
    # If multiple instances of these commands appear, only the first occurrence will be retained.
    exclusive_tool_commands: list[str] = [
        "Editor.edit_file_by_replace",
        "Editor.insert_content_at_line",
        "Editor.append_file",
        "Editor.open_file",
    ]
    # Equipped with three basic tools by default for optional use
    editor: Editor = Editor(enable_auto_lint=True)
    browser: Browser = Browser()

    # Experience
    experience_retriever: Annotated[ExpRetriever, Field(exclude=True)] = (
        DummyExpRetriever()
    )

    # Others
    observe_all_msg_from_buffer: bool = True
    command_rsp: str = ""  # the raw string containing the commands
    commands: list[dict] = []  # commands to be executed
    memory_k: int = 200  # number of memories (messages) to use as historical context
    use_fixed_sop: bool = False
    respond_language: str = (
        ""  # Language for responding humans and publishing messages.
    )
    use_summary: bool = True  # whether to summarize at the end

    def __init__(
        self, *args, project_dir=None, log_step=None, attack_monitor=None, **kwargs
    ):
        super().__init__(*args, project_dir=project_dir, **kwargs)
        self.log_step = log_step
        self.attack_monitor = attack_monitor
        self.step_counter = 0
        if project_dir:
            self.editor.working_dir = Path(project_dir)

    @model_validator(mode="after")
    def set_plan_and_tool(self) -> "RoleZero":
        # We force using this parameter for DataAnalyst
        assert self.react_mode == "react"

        # Roughly the same part as DataInterpreter.set_plan_and_tool
        self._set_react_mode(
            react_mode=self.react_mode, max_react_loop=self.max_react_loop
        )
        if self.tools and not self.tool_recommender:
            self.tool_recommender = BM25ToolRecommender(tools=self.tools, force=True)
        self.set_actions([RunCommand])

        # HACK: Init Planner, control it through dynamic thinking; Consider formalizing as a react mode
        self.planner = Planner(
            goal="", working_memory=self.rc.working_memory, auto_run=True
        )

        return self

    @model_validator(mode="after")
    def set_tool_execution(self) -> "RoleZero":
        # default map
        self.tool_execution_map = {
            "Plan.append_task": self.planner.plan.append_task,
            "Plan.reset_task": self.planner.plan.reset_task,
            "Plan.replace_task": self.planner.plan.replace_task,
            "RoleZero.ask_human": self.ask_human,
            "RoleZero.reply_to_human": self.reply_to_human,
        }
        if self.config.enable_search:
            self.tool_execution_map["SearchEnhancedQA.run"] = SearchEnhancedQA().run
        self.tool_execution_map.update(
            {
                f"Browser.{i}": getattr(self.browser, i)
                for i in [
                    "click",
                    "close_tab",
                    "go_back",
                    "go_forward",
                    "goto",
                    "hover",
                    "press",
                    "scroll",
                    "tab_focus",
                    "type",
                ]
            }
        )
        self.tool_execution_map.update(
            {
                f"Editor.{i}": getattr(self.editor, i)
                for i in [
                    "append_file",
                    "create_file",
                    "edit_file_by_replace",
                    "find_file",
                    "goto_line",
                    "insert_content_at_line",
                    "open_file",
                    "read",
                    "scroll_down",
                    "scroll_up",
                    "search_dir",
                    "search_file",
                    "similarity_search",
                    # "set_workdir",
                    "write",
                ]
            }
        )
        # can be updated by subclass
        self._update_tool_execution()
        return self

    @model_validator(mode="after")
    def set_longterm_memory(self) -> "RoleZero":
        """Set up long-term memory for the role if enabled in the configuration.

        If `enable_longterm_memory` is True, set up long-term memory.
        The role name will be used as the collection name.
        """

        if self.config.role_zero.enable_longterm_memory:
            # Use config.role_zero to initialize long-term memory
            self.rc.memory = RoleZeroLongTermMemory(
                **self.rc.memory.model_dump(),
                persist_path=self.config.role_zero.longterm_memory_persist_path,
                collection_name=self.name.replace(" ", ""),
                memory_k=self.config.role_zero.memory_k,
                similarity_top_k=self.config.role_zero.similarity_top_k,
                use_llm_ranker=self.config.role_zero.use_llm_ranker,
            )
            logger.info(f"Long-term memory set for role '{self.name}'")

        return self

    def _update_tool_execution(self):
        pass

    async def _think(self) -> bool:
        """Useful in 'react' mode. Use LLM to decide whether and what to do next."""
        # Compatibility
        if self.use_fixed_sop:
            return await super()._think()

        ### 0. Preparation ###
        if not self.rc.todo:
            return False

        if not self.planner.plan.goal:
            self.planner.plan.goal = self.get_memories()[-1].content
            detect_language_prompt = DETECT_LANGUAGE_PROMPT.format(
                requirement=self.planner.plan.goal
            )
            self.respond_language = await self.llm.aask(detect_language_prompt)
        ### 1. Experience ###
        example = self._retrieve_experience()

        ### 2. Plan Status ###
        plan_status, current_task = self._get_plan_status()

        ### 3. Tool/Command Info ###
        tools = await self.tool_recommender.recommend_tools()
        tool_info = json.dumps({tool.name: tool.schemas for tool in tools})

        ### Role Instruction ###
        instruction = self.instruction.strip()

        # 在system_prompt前插入Editor命令参数说明和内容规范要求
        EDITOR_COMMANDS_DOC = """
You can use the following Editor commands (use exact parameter names!):

- Editor.create_file(filename: str, content: str)
- Editor.insert_content_at_line(file_name: str, line_number: int, insert_content: str)
- Editor.edit_file_by_replace(file_name: str, line_number: int, new_content: str)
- Editor.open_file(path: str)

Do NOT use parameters like 'line', 'content', 'file_path', or 'file_name' unless specified above. Always use the exact parameter names and types as listed.

When generating code to insert, always:
- Output a complete, standalone Python file (including function definition, test cases, and necessary imports).
- Start the file with a comment or an empty line before any function/class definition.
- Ensure the code is properly indented and syntactically correct.
- Do not insert only a function body; always include the full context.
"""

        system_prompt = (
            EDITOR_COMMANDS_DOC
            + "\n"
            + self.system_prompt.format(
                role_info=self._get_prefix(),
                task_type_desc=self.task_type_desc,
                available_commands=tool_info,
                example=example,
                instruction=instruction,
            )
        )

        # 攻击监控逻辑 - 在system_prompt构建后注入攻击内容
        if hasattr(self, "attack_monitor") and self.attack_monitor:
            # 获取当前步骤的内容（从最近的memory中提取）
            current_content = ""
            if self.rc.memory.get(1):  # 获取最近的1条消息
                current_content = (
                    self.rc.memory.get(1)[-1].content if self.rc.memory.get(1) else ""
                )

        ### Make Decision Dynamically ###
        prompt = self.cmd_prompt.format(
            current_state=self.cmd_prompt_current_state,
            plan_status=plan_status,
            current_task=current_task,
            respond_language=self.respond_language,
        )

        # 监控逻辑 - 在system_prompt构建后注入内容
        if hasattr(self, "attack_monitor") and self.attack_monitor:
            # 新增：检查是否有待注入的prompt
            if (
                self.attack_monitor.injection_pending
                and not self.attack_monitor.injection_completed
            ):
                print(
                    f"[MONITOR_INJECTION] 检测到待注入状态，检查是否应该在当前步骤注入..."
                )

            # 构建当前步骤的内容描述（用于判断是否是think步骤）
            step_content = f"{self.profile} thinking: {prompt}"

            # 检查是否是攻击监控器还是诊断监控器
            if (
                hasattr(self.attack_monitor, "attack_content")
                and self.attack_monitor.attack_content
            ):
                # 这是攻击监控器
                if self.attack_monitor.should_inject_at_step(
                    self.step_counter, step_content, self.name, self.profile
                ):
                    strategy = self._select_injection_strategy()
                    injection_record = self.attack_monitor.inject_prompt(strategy)

                    if injection_record:
                        attack_prompt = injection_record.get("prompt", "")
                        system_prompt = f"{attack_prompt}\n\n{system_prompt}"
                        print(
                            f"[ATTACK_INJECTION] 已注入攻击prompt到system_prompt: {attack_prompt[:100]}..."
                        )
                        print(
                            f"[ATTACK_INJECTION] 策略: {strategy}, 步骤: {self.step_counter}"
                        )

                        self.attack_monitor.injection_completed = True
                        self.attack_monitor.injection_pending = False
                        print(f"[ATTACK_INJECTION] 注入完成，系统继续执行...")

            elif (
                hasattr(self.attack_monitor, "suggested_fix")
                and self.attack_monitor.suggested_fix
            ):
                # 这是诊断监控器
                if self.attack_monitor.should_inject_at_step(
                    self.step_counter, step_content, self.name, self.profile
                ):
                    strategy = "correction_injection"
                    injection_record = self.attack_monitor.inject_correction(strategy)

                    if injection_record:
                        correction_prompt = injection_record.get("prompt", "")
                        system_prompt = f"{correction_prompt}\n\n{system_prompt}"
                        print(
                            f"[DIAGNOSIS_INJECTION] 已注入纠错prompt到system_prompt: {correction_prompt[:100]}..."
                        )
                        print(
                            f"[DIAGNOSIS_INJECTION] 策略: {strategy}, 步骤: {self.step_counter}"
                        )

                        self.attack_monitor.injection_completed = True
                        self.attack_monitor.injection_pending = False
                        print(f"[DIAGNOSIS_INJECTION] 注入完成，系统继续执行...")

        ### Recent Observation ###
        memory = self.rc.memory.get(self.memory_k)
        memory = await self.parse_browser_actions(memory)
        memory = await self.parse_editor_result(memory)
        memory = self.parse_images(memory)

        req = self.llm.format_msg(memory + [UserMessage(content=prompt)])
        state_data = dict(
            plan_status=plan_status,
            current_task=current_task,
            instruction=instruction,
        )

        # 新增：记录思考过程到log_step
        if hasattr(self, "log_step") and self.log_step:
            thinking_content = f"{self.profile} thinking: {prompt}"
            self.log_step(thinking_content, "assistant", self.profile)

        async with ThoughtReporter(enable_llm_stream=True) as reporter:
            await reporter.async_report({"type": "react"})
            self.command_rsp = await self.llm_cached_aask(
                req=req, system_msgs=[system_prompt], state_data=state_data
            )
        self.command_rsp = await self._check_duplicates(req, self.command_rsp)

        # 更新步骤计数器 - 注意：这里不再直接增加，而是由外部统一管理
        # self.step_counter += 1  # 注释掉这行，由StepCounter类统一管理
        return True

    @exp_cache(
        context_builder=RoleZeroContextBuilder(), serializer=RoleZeroSerializer()
    )
    async def llm_cached_aask(
        self, *, req: list[dict], system_msgs: list[str], **kwargs
    ) -> str:
        """Use `exp_cache` to automatically manage experiences.

        The `RoleZeroContextBuilder` attempts to add experiences to `req`.
        The `RoleZeroSerializer` extracts essential parts of `req` for the experience pool, trimming lengthy entries to retain only necessary parts.
        """
        return await self.llm.aask(req, system_msgs=system_msgs)

    async def parse_browser_actions(self, memory: list[Message]) -> list[Message]:
        if not self.browser.is_empty_page:
            pattern = re.compile(r"Command Browser\.(\w+) executed")
            for index, msg in zip(range(len(memory), 0, -1), memory[::-1]):
                if pattern.search(msg.content):
                    memory.insert(
                        index,
                        UserMessage(
                            cause_by="browser", content=await self.browser.view()
                        ),
                    )
                    break
        return memory

    async def parse_editor_result(
        self, memory: list[Message], keep_latest_count=5
    ) -> list[Message]:
        """Retain the latest result and remove outdated editor results."""
        pattern = re.compile(r"Command Editor\.(\w+?) executed")
        new_memory = []
        i = 0
        for msg in reversed(memory):
            matches = pattern.findall(msg.content)
            if matches:
                i += 1
                if i > keep_latest_count:
                    new_content = msg.content[: msg.content.find("Command Editor")]
                    new_content += "\n".join(
                        [f"Command Editor.{match} executed." for match in matches]
                    )
                    msg = UserMessage(content=new_content)
            new_memory.append(msg)
        # Reverse the new memory list so the latest message is at the end
        new_memory.reverse()
        return new_memory

    def parse_images(self, memory: list[Message]) -> list[Message]:
        if not self.llm.support_image_input():
            return memory
        for msg in memory:
            if IMAGES in msg.metadata or msg.role != "user":
                continue
            images = extract_and_encode_images(msg.content)
            if images:
                msg.add_metadata(IMAGES, images)
        return memory

    def _get_prefix(self) -> str:
        time_info = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return super()._get_prefix() + f" The current time is {time_info}."

    async def _act(self) -> Tuple[Message, list]:
        if self.use_fixed_sop:
            return await super()._act(), []
        commands, ok, self.command_rsp = await self._parse_commands(self.command_rsp)
        self.rc.memory.add(AIMessage(content=self.command_rsp))
        if not ok:
            # 检查是否是错误命令
            if commands and len(commands) == 1 and "error" in commands[0]:
                error_msg = commands[0]["error"]
            else:
                error_msg = str(commands)
            self.rc.memory.add(UserMessage(content=error_msg, cause_by=RunCommand))
            # 修复：返回Message对象而不是字符串
            error_message = AIMessage(
                content=f"Command parsing failed: {error_msg}",
                sent_from=self.name,
                cause_by=RunCommand,
            )
            return error_message, []
        logger.info(f"Commands: \n{commands}")
        outputs = await self._run_commands(commands)
        logger.info(f"Commands outputs: \n{outputs}")
        self.rc.memory.add(UserMessage(content=outputs, cause_by=RunCommand))
        msg = AIMessage(
            content=f"I have finished the task, please mark my task as finished. Outputs: {outputs}",
            sent_from=self.name,
            cause_by=RunCommand,
        )
        return msg, commands

    async def _react(self) -> Message:
        # NOTE: Diff 1: Each time landing here means news is observed, set todo to allow news processing in _think
        self._set_state(0)

        # problems solvable by quick thinking doesn't need to a formal think-act cycle
        quick_rsp, _ = await self._quick_think()
        if quick_rsp:
            return quick_rsp

        actions_taken = 0
        rsp = AIMessage(
            content="No actions taken yet", cause_by=Action
        )  # will be overwritten after Role _act
        while actions_taken < self.rc.max_react_loop:
            # NOTE: Diff 2: Keep observing within _react, news will go into memory, allowing adapting to new info
            plan = getattr(self.planner, "plan", None)
            if (
                actions_taken > 0
                and plan
                and getattr(plan, "tasks", None)
                and plan.is_plan_finished()
            ):
                logger.info("[RoleZero] Plan is finished, breaking react loop.")
                break
            await self._observe()

            # think
            has_todo = await self._think()
            if not has_todo:
                break
            # act
            logger.debug(f"{self._setting}: {self.rc.state=}, will do {self.rc.todo}")
            rsp, commands = await self._act()
            actions_taken += 1

            # === 新增：只要有end命令，立即break ===
            if any(cmd.get("command_name") == "end" for cmd in commands):
                logger.info("[RoleZero] Detected end command, breaking react loop.")
                break
            # ================================

            # post-check
            if self.rc.max_react_loop >= 10 and actions_taken >= self.rc.max_react_loop:
                # If max_react_loop is a small value (e.g. < 10), it is intended to be reached and make the agent stop
                logger.warning(f"reached max_react_loop: {actions_taken}")
                # human_rsp = await self.ask_human(
                #     "I have reached my max action rounds, do you want me to continue? Yes or no"
                # )
                actions_taken = 0
                # if "yes" in human_rsp.lower():
                #     actions_taken = 0
        return rsp  # return output from the last action

    def format_quick_system_prompt(self) -> str:
        """Format the system prompt for quick thinking."""
        return QUICK_THINK_SYSTEM_PROMPT.format(
            examples=QUICK_THINK_EXAMPLES, role_info=self._get_prefix()
        )

    async def _quick_think(self) -> Tuple[Message, str]:
        answer = ""
        rsp_msg = None
        if self.rc.news[-1].cause_by != any_to_str(UserRequirement):
            # Agents themselves won't generate quick questions, use this rule to reduce extra llm calls
            return rsp_msg, ""

        # routing
        memory = self.get_memories(k=self.memory_k)
        context = self.llm.format_msg(
            memory + [UserMessage(content=QUICK_THINK_PROMPT)]
        )
        async with ThoughtReporter() as reporter:
            await reporter.async_report({"type": "classify"})
            intent_result = await self.llm.aask(
                context, system_msgs=[self.format_quick_system_prompt()]
            )

        if (
            "QUICK" in intent_result or "AMBIGUOUS" in intent_result
        ):  # llm call with the original context
            async with ThoughtReporter(enable_llm_stream=True) as reporter:
                await reporter.async_report({"type": "quick"})
                answer = await self.llm.aask(
                    self.llm.format_msg(memory),
                    system_msgs=[
                        QUICK_RESPONSE_SYSTEM_PROMPT.format(
                            role_info=self._get_prefix()
                        )
                    ],
                )
            # If the answer contains the substring '[Message] from A to B:', remove it.
            pattern = r"\[Message\] from .+? to .+?:\s*"
            answer = re.sub(pattern, "", answer, count=1)
            if "command_name" in answer:
                # an actual TASK intent misclassified as QUICK, correct it here, FIXME: a better way is to classify it correctly in the first place
                answer = ""
                intent_result = "TASK"
        elif "SEARCH" in intent_result:
            query = "\n".join(str(msg) for msg in memory)
            answer = await SearchEnhancedQA().run(query)

        if answer:
            self.rc.memory.add(AIMessage(content=answer, cause_by=QUICK_THINK_TAG))
            await self.reply_to_human(content=answer)
            rsp_msg = AIMessage(
                content=answer,
                sent_from=self.name,
                cause_by=QUICK_THINK_TAG,
            )

        return rsp_msg, intent_result

    async def _check_duplicates(
        self, req: list[dict], command_rsp: str, check_window: int = 10
    ):
        past_rsp = [mem.content for mem in self.rc.memory.get(check_window)]
        if command_rsp in past_rsp and '"command_name": "end"' not in command_rsp:
            # Normal response with thought contents are highly unlikely to reproduce
            # If an identical response is detected, it is a bad response, mostly due to LLM repeating generated content
            # In this case, ask human for help and regenerate
            # TODO: switch to llm_cached_aask

            #  Hard rule to ask human for help
            if past_rsp.count(command_rsp) >= 2:
                if '"command_name": "Plan.finish_current_task",' in command_rsp:
                    # Detect the duplicate of the 'Plan.finish_current_task' command, and use the 'end' command to finish the task.
                    logger.warning(f"Duplicate response detected: {command_rsp}")
                    return END_COMMAND
                problem = await self.llm.aask(
                    req
                    + [
                        UserMessage(
                            content=SUMMARY_PROBLEM_WHEN_DUPLICATE.format(
                                language=self.respond_language
                            )
                        )
                    ]
                )
                ASK_HUMAN_COMMAND[0]["args"]["question"] = (
                    ASK_HUMAN_GUIDANCE_FORMAT.format(problem=problem).strip()
                )
                ask_human_command = (
                    "```json\n"
                    + json.dumps(ASK_HUMAN_COMMAND, indent=4, ensure_ascii=False)
                    + "\n```"
                )
                return ask_human_command
            # Try correction by self
            logger.warning(f"Duplicate response detected: {command_rsp}")
            regenerate_req = req + [UserMessage(content=REGENERATE_PROMPT)]
            regenerate_req = self.llm.format_msg(regenerate_req)
            command_rsp = await self.llm.aask(regenerate_req)
        return command_rsp

    async def _parse_commands(self, command_rsp) -> Tuple[List[Dict], bool, str]:
        """Retrieves commands from the Large Language Model (LLM).

        This function attempts to retrieve a list of commands from the LLM by
        processing the response (`self.command_rsp`). It handles potential errors
        during parsing and LLM response formats.

        Returns:
            A tuple containing:
                - List of commands (or error message as list with single dict if failed)
                - A boolean flag indicating success (True) or failure (False).
                - The original command response string
        """
        try:
            commands = CodeParser.parse_code(block=None, lang="json", text=command_rsp)
            if commands.endswith("]") and not commands.startswith("["):
                commands = "[" + commands
            commands = json.loads(
                repair_llm_raw_output(
                    output=commands, req_keys=[None], repair_type=RepairType.JSON
                )
            )
        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse JSON for: {command_rsp}. Trying to repair..."
            )
            commands = await self.llm.aask(
                msg=JSON_REPAIR_PROMPT.format(
                    json_data=command_rsp, json_decode_error=str(e)
                )
            )
            try:
                commands = json.loads(
                    CodeParser.parse_code(block=None, lang="json", text=commands)
                )
            except json.JSONDecodeError:
                # repair escape error of code and math
                commands = CodeParser.parse_code(
                    block=None, lang="json", text=command_rsp
                )
                new_command = repair_escape_error(commands)
                commands = json.loads(
                    repair_llm_raw_output(
                        output=new_command, req_keys=[None], repair_type=RepairType.JSON
                    )
                )
        except Exception as e:
            tb = traceback.format_exc()
            print(tb)
            error_msg = str(e)
            # 修复：返回错误信息作为列表中的单个字典，而不是字符串
            return [{"error": error_msg}], False, command_rsp

        # 为了对LLM不按格式生成进行容错
        if isinstance(commands, dict):
            commands = commands["commands"] if "commands" in commands else [commands]

        # 验证命令格式
        for cmd in commands:
            if not isinstance(cmd, dict) or "command_name" not in cmd:
                error_msg = f"Invalid command format. Each command must be a dictionary with a 'command_name' field. Got: {cmd}"
                # 修复：返回错误信息作为列表中的单个字典，而不是字符串
                return [{"error": error_msg}], False, command_rsp

        # Set the exclusive command flag to False.
        command_flag = [
            command["command_name"] not in self.exclusive_tool_commands
            for command in commands
        ]
        if command_flag.count(False) > 1:
            # Keep only the first exclusive command
            index_of_first_exclusive = command_flag.index(False)
            commands = commands[: index_of_first_exclusive + 1]
            command_rsp = (
                "```json\n"
                + json.dumps(commands, indent=4, ensure_ascii=False)
                + "\n```"
            )
            logger.info(
                "exclusive command more than one in current command list. change the command list.\n"
                + command_rsp
            )
        return commands, True, command_rsp

    async def _run_commands(self, commands) -> str:
        outputs = []
        for cmd in commands:
            output = f"Command {cmd['command_name']} executed"
            args = cmd.get("args", {})

            # 参数兼容处理
            if cmd["command_name"] == "Editor.insert_content_at_line":
                if "line" in args and "line_number" not in args:
                    args["line_number"] = args.pop("line")
                if "content" in args and "insert_content" not in args:
                    args["insert_content"] = args.pop("content")
                args = {
                    k: v
                    for k, v in args.items()
                    if k in {"file_name", "line_number", "insert_content"}
                }
                print(f"[DEBUG] insert_content_at_line args after processing: {args}")

                if not args.get("file_name") or not str(args["file_name"]).strip():
                    outputs.append(
                        f"ERROR: Editor.insert_content_at_line requires the 'file_name' parameter, but it is missing. No file will be written. Please check your command."
                    )
                    continue

                # insert_content 缺失容错计数
                if not hasattr(self, "_insert_content_missing_counter"):
                    self._insert_content_missing_counter = 0
                if (
                    "insert_content" not in args
                    or not str(args["insert_content"]).strip()
                ):
                    self._insert_content_missing_counter += 1
                    outputs.append(
                        f"ERROR: Editor.insert_content_at_line requires the 'insert_content' parameter, but it is missing. No content will be inserted. Please check your command."
                    )
                    continue
                else:
                    self._insert_content_missing_counter = 0
                dir_name = os.path.dirname(args["file_name"])
                if not dir_name:
                    project_dir = getattr(self, "project_dir", None)
                    if project_dir:
                        args["file_name"] = os.path.join(project_dir, args["file_name"])
                    else:
                        args["file_name"] = os.path.join(
                            DEFAULT_WORKSPACE_ROOT, args["file_name"]
                        )
                    dir_name = os.path.dirname(args["file_name"])
                os.makedirs(dir_name, exist_ok=True)
                try:
                    with open(args["file_name"], "w", encoding="utf-8") as f:
                        f.write(args["insert_content"])
                    outputs.append(
                        f"Command Editor.insert_content_at_line executed: wrote to {args['file_name']}"
                    )
                except Exception as e:
                    outputs.append(f"Error writing to {args['file_name']}: {e}")
                continue
            if cmd["command_name"] == "Editor.write":
                if "file_name" in args and "path" not in args:
                    args["path"] = args.pop("file_name")
                if "filename" in args and "path" not in args:
                    args["path"] = args.pop("filename")
                args = {k: v for k, v in args.items() if k in {"path", "content"}}
                print(f"[DEBUG] Editor.write args after processing: {args}")
                if not args.get("path") or not str(args["path"]).strip():
                    outputs.append(
                        "ERROR: Editor.write requires the 'path' parameter, but it is missing. No file will be written. Please check your command."
                    )
                    continue
                if "content" not in args or not str(args["content"]).strip():
                    outputs.append(
                        "ERROR: Editor.write requires the 'content' parameter, but it is missing. No content will be written. Please check your command."
                    )
                    continue
                dir_name = os.path.dirname(args["path"])
                if not dir_name:
                    project_dir = getattr(self, "project_dir", None)
                    if project_dir:
                        args["path"] = os.path.join(project_dir, args["path"])
                    else:
                        args["path"] = os.path.join(
                            DEFAULT_WORKSPACE_ROOT, args["path"]
                        )
                    dir_name = os.path.dirname(args["path"])
                os.makedirs(dir_name, exist_ok=True)
                try:
                    with open(args["path"], "w", encoding="utf-8") as f:
                        f.write(args["content"])
                    outputs.append(
                        f"Command Editor.write executed: wrote to {args['path']}"
                    )
                except Exception as e:
                    outputs.append(f"Error writing to {args['path']}: {e}")
                continue
            if cmd["command_name"] == "Editor.create_file":
                if "file_name" in args and "filename" not in args:
                    args["filename"] = args.pop("file_name")
                filename = args.get("filename")
                content = args.get("content", "")
                if not filename or not str(filename).strip():
                    outputs.append(
                        "ERROR: Editor.create_file requires the 'filename' parameter, but it is missing. No file will be created."
                    )
                    continue
                if content and content.strip():
                    outputs.append(
                        "ERROR: Editor.create_file can only create empty files, cannot write content. Please use Editor.write or Editor.insert_content_at_line instead."
                    )
                    continue
                dir_name = os.path.dirname(filename)
                if not dir_name:
                    # 自动补全到工作目录
                    project_dir = getattr(self, "project_dir", None)
                    if project_dir:
                        filename = os.path.join(project_dir, filename)
                    else:
                        filename = os.path.join(DEFAULT_WORKSPACE_ROOT, filename)
                    dir_name = os.path.dirname(filename)
                os.makedirs(dir_name, exist_ok=True)
                try:
                    open(filename, "w").close()
                    outputs.append(
                        f"Command Editor.create_file executed: created {filename}"
                    )
                except Exception as e:
                    outputs.append(f"Error creating file {filename}: {e}")
                continue

            # handle special command first
            if self._is_special_command(cmd):
                special_command_output = await self._run_special_command(cmd)
                outputs.append(output + ":" + special_command_output)
                continue
            # run command as specified by tool_execute_map
            if cmd["command_name"] in self.tool_execution_map:
                tool_obj = self.tool_execution_map[cmd["command_name"]]
                try:
                    if inspect.iscoroutinefunction(tool_obj):
                        tool_output = await tool_obj(**args)
                    else:
                        tool_output = tool_obj(**args)
                    if tool_output:
                        output += f": {str(tool_output)}"
                    outputs.append(output)
                except Exception as e:
                    tb = traceback.format_exc()
                    logger.exception(str(e) + tb)
                    outputs.append(output + f": {tb}")
                    break  # Stop executing if any command fails
            else:
                outputs.append(f"Command {cmd['command_name']} not found.")
                break
        outputs = "\n\n".join(outputs)

        # 将错误信息添加到memory中,确保agent能看到
        if any("ERROR:" in output for output in outputs.split("\n\n")):
            self.rc.memory.add(UserMessage(content=outputs, cause_by=RunCommand))

        return outputs

    def _is_special_command(self, cmd) -> bool:
        return cmd["command_name"] in self.special_tool_commands

    async def _run_special_command(self, cmd) -> str:
        """command requiring special check or parsing"""
        command_output = ""

        if cmd["command_name"] == "Plan.finish_current_task":
            if self.planner.plan.is_plan_finished():
                # Plan已完成，直接转end
                command_output = "Plan is already finished. Switching to end command."
                end_output = await self._run_special_command(
                    {"command_name": "end", "args": {}}
                )
                return command_output + "\n" + "Command end executed: " + end_output
            else:
                self.planner.plan.finish_current_task()
                command_output = "Current task is finished. If you no longer need to take action, use the command 'end' to stop."

        elif cmd["command_name"] == "end":
            command_output = await self._end()
        elif cmd["command_name"] == "RoleZero.ask_human":
            human_response = await self.ask_human(**cmd["args"])
            if human_response.strip().lower().endswith(("stop", "<stop>")):
                human_response += "The user has asked me to stop because I have encountered a problem."
                self.rc.memory.add(
                    UserMessage(content=human_response, cause_by=RunCommand)
                )
                end_output = "\nCommand end executed:"
                end_output += await self._end()
                return end_output
            return human_response
        # output from bash.run may be empty, add decorations to the output to ensure visibility.
        elif cmd["command_name"] == "Terminal.run_command":
            tool_obj = self.tool_execution_map[cmd["command_name"]]
            tool_output = await tool_obj(**cmd["args"])
            if len(tool_output) <= 10:
                command_output += f"\n[command]: {cmd['args']['cmd']} \n[command output] : {tool_output} (pay attention to this.)"
            else:
                command_output += f"\n[command]: {cmd['args']['cmd']} \n[command output] : {tool_output}"

            # 记录terminal output，让run_rollout.py中的log_step函数处理合并
            if self.log_step:
                terminal_content = f"Terminal output: [command]: {cmd['args']['cmd']} \n[command output] : {tool_output}"
                self.log_step(terminal_content, "Terminal", self.profile)
        return command_output

    def _get_plan_status(self) -> Tuple[str, str]:
        plan_status = self.planner.plan.model_dump(include=["goal", "tasks"])
        current_task = (
            self.planner.plan.current_task.model_dump(
                exclude=["code", "result", "is_success"]
            )
            if self.planner.plan.current_task
            else ""
        )
        # format plan status
        # Example:
        # [GOAL] create a 2048 game
        # [TASK_ID 1] (finished) Create a Product Requirement Document (PRD) for the 2048 game. This task depends on tasks[]. [Assign to Alice]
        # [TASK_ID 2] (        ) Design the system architecture for the 2048 game. This task depends on tasks[1]. [Assign to Bob]
        formatted_plan_status = f"[GOAL] {plan_status['goal']}\n"
        if len(plan_status["tasks"]) > 0:
            formatted_plan_status += "[Plan]\n"
            for task in plan_status["tasks"]:
                formatted_plan_status += f"[TASK_ID {task['task_id']}] ({'finished' if task['is_finished'] else '    '}){task['instruction']} This task depends on tasks{task['dependent_task_ids']}. [Assign to {task['assignee']}]\n"
        else:
            formatted_plan_status += "No Plan \n"
        return formatted_plan_status, current_task

    def _retrieve_experience(self) -> str:
        """Default implementation of experience retrieval. Can be overwritten in subclasses."""
        context = [str(msg) for msg in self.rc.memory.get(self.memory_k)]
        context = "\n\n".join(context)
        example = self.experience_retriever.retrieve(context=context)
        return example

    async def ask_human(self, question: str) -> str:
        """Use this when you fail the current task or if you are unsure of the situation encountered. Your response should contain a brief summary of your situation, ended with a clear and concise question."""
        # NOTE: Can be overwritten in remote setting
        from metagpt.environment.mgx.mgx_env import MGXEnv  # avoid circular import

        if not isinstance(self.rc.env, MGXEnv):
            return "Not in MGXEnv, command will not be executed."
        return await self.rc.env.ask_human(question, sent_from=self)

    async def reply_to_human(self, content: str) -> str:
        """Reply to human user with the content provided. Use this when you have a clear answer or solution to the user's question."""
        # NOTE: Can be overwritten in remote setting
        from metagpt.environment.mgx.mgx_env import MGXEnv  # avoid circular import

        if not isinstance(self.rc.env, MGXEnv):
            return "Not in MGXEnv, command will not be executed."
        return await self.rc.env.reply_to_human(content, sent_from=self)

    async def _end(self, **kwarg):
        self._set_state(-1)
        memory = self.rc.memory.get(self.memory_k)
        # Ensure reply to the human before the "end" command is executed. Hard code k=5 for checking.
        if not any(
            ["reply_to_human" in memory.content for memory in self.get_memories(k=5)]
        ):
            logger.info("manually reply to human")
            reply_to_human_prompt = REPORT_TO_HUMAN_PROMPT.format(
                respond_language=self.respond_language
            )
            async with ThoughtReporter(enable_llm_stream=True) as reporter:
                await reporter.async_report({"type": "quick"})
                reply_content = await self.llm.aask(
                    self.llm.format_msg(memory + [UserMessage(reply_to_human_prompt)])
                )
            await self.reply_to_human(content=reply_content)
            self.rc.memory.add(AIMessage(content=reply_content, cause_by=RunCommand))
        outputs = ""
        # Summary of the Completed Task and Deliverables
        if self.use_summary:
            logger.info("end current run and summarize")
            outputs = await self.llm.aask(
                self.llm.format_msg(memory + [UserMessage(SUMMARY_PROMPT)])
            )
        return outputs

    def _select_injection_strategy(self) -> str:
        """选择注入策略 - 简化为直接攻击策略"""
        # 统一使用direct_attack策略，不再根据角色或步骤选择不同策略
        return "direct_attack"
