#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 14:43
@Author  : alexanderwu
@File    : product_manager.py
@Modified By: liushaojie, 2024/10/17.
"""
from metagpt.actions import UserRequirement, WritePRD
from metagpt.actions.prepare_documents import PrepareDocuments
from metagpt.actions.search_enhanced_qa import SearchEnhancedQA
from metagpt.prompts.product_manager import PRODUCT_MANAGER_INSTRUCTION
from metagpt.roles.di.role_zero import RoleZero
from metagpt.roles.role import RoleReactMode
from metagpt.tools.libs.browser import Browser
from metagpt.tools.libs.editor import Editor
from metagpt.utils.common import any_to_name, any_to_str, tool2name
from metagpt.utils.git_repository import GitRepository


class ProductManager(RoleZero):
    """
    Represents a Product Manager role responsible for product development and management.

    Attributes:
        name (str): Name of the product manager.
        profile (str): Role profile, default is 'Product Manager'.
        goal (str): Goal of the product manager.
        constraints (str): Constraints or limitations for the product manager.
    """

    name: str = "Alice"
    profile: str = "Product Manager"
    goal: str = (
        "Create a Product Requirement Document or market research/competitive product research."
    )
    constraints: str = (
        "utilize the same language as the user requirements for seamless communication"
    )
    instruction: str = PRODUCT_MANAGER_INSTRUCTION
    tools: list[str] = [
        "RoleZero",
        Browser.__name__,
        Editor.__name__,
        SearchEnhancedQA.__name__,
    ]

    todo_action: str = any_to_name(WritePRD)

    def __init__(
        self, *args, project_dir=None, log_step=None, attack_monitor=None, **kwargs
    ):
        super().__init__(
            *args,
            project_dir=project_dir,
            log_step=log_step,
            attack_monitor=attack_monitor,
            **kwargs,
        )
        if self.use_fixed_sop:
            self.enable_memory = False
            self.set_actions([PrepareDocuments(send_to=any_to_str(self)), WritePRD])
            self._watch([UserRequirement, PrepareDocuments])
            self.rc.react_mode = RoleReactMode.BY_ORDER

    def _update_tool_execution(self):
        wp = WritePRD()
        self.tool_execution_map.update(tool2name(WritePRD, ["run"], wp.run))

    async def _run_special_command(self, cmd) -> str:
        """Override to handle special commands for ProductManager role."""
        if cmd["command_name"] == "Terminal.run_command":
            return "ProductManager does not have the capability to run terminal commands directly. Please assign this task to Engineer2 (Alex), DataAnalyst (David), or Architect (Bob) who have the appropriate technical capabilities to execute code and run commands."

        # Call parent method for other special commands
        return await super()._run_special_command(cmd)

    async def _think(self) -> bool:
        """Decide what to do"""
        if not self.use_fixed_sop:
            res = await super()._think()
            if self.log_step:
                self.log_step(
                    f"{self.profile} thinking: {self.command_rsp}",
                    role="assistant",
                    name=self.profile,
                )
            return res

        if (
            GitRepository.is_git_dir(self.config.project_path)
            and not self.config.git_reinit
        ):
            self._set_state(1)
        else:
            self._set_state(0)
            self.config.git_reinit = False
            self.todo_action = any_to_name(WritePRD)
        return bool(self.rc.todo)
