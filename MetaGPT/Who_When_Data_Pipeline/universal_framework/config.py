#!/usr/bin/env python3
"""
Unified Configuration Management - Supporting configurations for different datasets
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum
from pathlib import Path


class DatasetType(Enum):
    """Dataset type enumeration"""
    MBPP_PLUS = "mbpp_plus"
    KODCODE = "kodcode"
    HUMANEVAL_PLUS = "humaneval_plus"
    # Can continue adding other dataset types


class ProcessType(Enum):
    """Process type enumeration"""
    ATTACK = "attack"
    DIAGNOSE = "diagnose"


@dataclass
class RetryConfig:
    """Retry configuration"""
    max_retries: int = 3
    retry_delay: int = 2
    backoff_factor: float = 1.5


@dataclass
class EvalConfig:
    """Evaluation configuration"""
    evalplus_path: str = "coding/run_eval.py"
    eval_result_file: str = "eval_result.json"
    timeout: int = 300


@dataclass
class FrameworkConfig:
    """Framework configuration class"""
    # Basic configuration
    dataset_type: DatasetType
    workspace_dir: str
    output_base_dir: str
    
    # Processing configuration
    max_rounds: int = 3
    max_tasks_per_round: Optional[int] = None
    enable_concurrent: bool = False
    concurrent_limit: int = 5
    
    # Retry configuration
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    
    # Evaluation configuration
    eval_config: EvalConfig = field(default_factory=EvalConfig)
    
    # Dataset-specific configuration
    dataset_specific: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self):
        """Validate configuration"""
        if not os.path.exists(self.workspace_dir):
            raise ValueError(f"Workspace directory not found: {self.workspace_dir}")
        
        if not os.path.exists(self.output_base_dir):
            os.makedirs(self.output_base_dir, exist_ok=True)
        
        # Validate dataset-specific configuration
        self._validate_dataset_specific()
    
    def _validate_dataset_specific(self):
        """Validate dataset-specific configuration"""
        if self.dataset_type == DatasetType.KODCODE:
            required_keys = ["task_id_format", "log_file_pattern"]
            for key in required_keys:
                if key not in self.dataset_specific:
                    raise ValueError(f"KodCode configuration missing required field: {key}")
        
        elif self.dataset_type == DatasetType.MBPP_PLUS:
            required_keys = ["task_id_format", "log_file_pattern"]
            for key in required_keys:
                if key not in self.dataset_specific:
                    raise ValueError(f"MBPP+ configuration missing required field: {key}")
    
    def get_directory_structure(self) -> Dict[str, str]:
        """Get directory structure"""
        base_dir = self.output_base_dir
        
        return {
            "attack_suggestions": os.path.join(base_dir, "attack_suggestions_round_{}"),
            "attacked_logs": os.path.join(base_dir, "attacked_logs_round_{}"),
            "attack_records": os.path.join(base_dir, "attack_records_round_{}"),
            "attacked_still_succeed": os.path.join(base_dir, "attacked_still_succeed_tasks_round_{}.json"),
            "diagnosed_logs": os.path.join(base_dir, "diagnosed_logs_round_{}"),
            "improved_logs": os.path.join(base_dir, "improved_logs_round_{}"),
            "diagnosis_records": os.path.join(base_dir, "diagnosis_records_round_{}"),
            "replayed_still_failed": os.path.join(base_dir, "replayed_still_failed_tasks_round_{}.json"),
            "final_results": os.path.join(base_dir, "final_results"),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "dataset_type": self.dataset_type.value,
            "workspace_dir": self.workspace_dir,
            "output_base_dir": self.output_base_dir,
            "max_rounds": self.max_rounds,
            "max_tasks_per_round": self.max_tasks_per_round,
            "enable_concurrent": self.enable_concurrent,
            "concurrent_limit": self.concurrent_limit,
            "retry_config": {
                "max_retries": self.retry_config.max_retries,
                "retry_delay": self.retry_config.retry_delay,
                "backoff_factor": self.retry_config.backoff_factor,
            },
            "eval_config": {
                "evalplus_path": self.eval_config.evalplus_path,
                "eval_result_file": self.eval_config.eval_result_file,
                "timeout": self.eval_config.timeout,
            },
            "dataset_specific": self.dataset_specific,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FrameworkConfig':
        """Create configuration from dictionary"""
        return cls(
            dataset_type=DatasetType(data["dataset_type"]),
            workspace_dir=data["workspace_dir"],
            output_base_dir=data["output_base_dir"],
            max_rounds=data.get("max_rounds", 3),
            max_tasks_per_round=data.get("max_tasks_per_round"),
            enable_concurrent=data.get("enable_concurrent", False),
            concurrent_limit=data.get("concurrent_limit", 5),
            retry_config=RetryConfig(**data.get("retry_config", {})),
            eval_config=EvalConfig(**data.get("eval_config", {})),
            dataset_specific=data.get("dataset_specific", {}),
        )
    
    def save_to_file(self, file_path: str):
        """Save configuration to file"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'FrameworkConfig':
        """Load configuration from file"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_eval_result_file(self, round_num: int, eval_type: str = None, difficulty: str = "hard") -> str:
        """Get evalplus result file path for specified round"""
        if self.dataset_type == DatasetType.KODCODE:
            # KodCode evalplus result file path
            eval_results_dir = os.path.join(self.workspace_dir, "coding", "eval_results")
            
            # Build filename based on eval_type and difficulty
            if eval_type == "attack":
                filename = f"kodcode-{difficulty}-attack-round-{round_num}.json"
            elif eval_type == "diagnose":
                filename = f"kodcode-{difficulty}-diagnose-round-{round_num}.json"
            else:
                # Default to filename without type identifier
                filename = f"kodcode-{difficulty}-round-{round_num}.json"
            
            return os.path.join(eval_results_dir, filename)
        else:
            # MBPP+ evalplus result file path - to be implemented
            print("Warning: MBPP+ evalplus result file path retrieval not implemented yet")
            return os.path.join(
                self.output_base_dir,
                f"evalplus_attacked_result_round_{round_num}.json"
            )

    def get_eval_command(self, round_num: int, eval_type: str = None, difficulty: str = "hard") -> str:
        """Get evaluation command"""
        if self.dataset_type == DatasetType.KODCODE:
            # Build run_eval.py command
            run_eval_path = os.path.join(self.workspace_dir, "coding", "run_eval.py")
            
            # Dynamically determine log directory based on round and eval_type
            if eval_type == "attack":
                # Post-attack eval: read from attacked logs directory
                log_dir = os.path.join(self.output_base_dir, f"attacked_logs_round_{round_num}")
            elif eval_type == "diagnose":
                # Post-diagnosis eval: read from improved logs directory
                log_dir = os.path.join(self.output_base_dir, f"improved_logs_round_{round_num}")
            else:
                # Original eval or unspecified type: use default directory
                log_dir = os.path.join(self.workspace_dir, "coding", "logs", "kodcode")
            
            # Select corresponding dataset file based on difficulty
            dataset_file = f"coding/datasets/code/kodcode-light-rl-10k-{difficulty}.parquet"
            
            # Set environment variables
            env_vars = f"EVAL_ROUND={round_num} EVAL_TYPE={eval_type or 'general'} EVAL_DIFFICULTY={difficulty} LOG_ROOT={log_dir}"
            
            command = f"cd {self.workspace_dir} && {env_vars} python {run_eval_path} --dataset {dataset_file} --concurrency 10"
            
            return command
        else:
            # MBPP+ evaluation command - to be implemented
            print("Warning: MBPP+ evaluation command generation not implemented yet")
            return "echo 'MBPP+ evaluation not implemented yet'"

    def get_original_eval_result_file(self, difficulty: str = "hard") -> str:
        """Get original evalplus result file path"""
        if self.dataset_type == DatasetType.KODCODE:
            # KodCode original evalplus result file path
            return os.path.join(
                self.workspace_dir,
                "coding",
                "eval_results",
                f"kodcode-{difficulty}.json"
            )
        else:
            # MBPP+ original evalplus result file path
            return os.path.join(
                self.workspace_dir,
                "coding",
                "eval_results",
                "mbpp-plus.json"
            )

    def get_all_original_eval_result_files(self) -> Dict[str, str]:
        """Get all difficulty levels' original evalplus result file paths"""
        if self.dataset_type == DatasetType.KODCODE:
            # KodCode all difficulty result files
            eval_results_dir = os.path.join(self.workspace_dir, "coding", "eval_results")
            difficulties = ["easy", "medium", "hard"]
            result_files = {}
            
            for difficulty in difficulties:
                file_path = os.path.join(eval_results_dir, f"kodcode-{difficulty}.json")
                if os.path.exists(file_path):
                    result_files[difficulty] = file_path
            
            return result_files
        else:
            # MBPP+ has only one result file
            return {
                "mbpp_plus": self.get_original_eval_result_file()
            }

    def get_logs_dir(self) -> str:
        """Get logs directory path"""
        if self.dataset_type == DatasetType.KODCODE:
            # KodCode logs directory
            return os.path.join(
                self.workspace_dir,
                "coding",
                "logs",
                "kodcode"
            )
        else:
            # MBPP+ logs directory
            return os.path.join(
                self.workspace_dir,
                "MBPP+_dataset",
                "plus_0.2.0",
                "logs"
            )


def create_kodcode_config(workspace_dir: str, output_dir: str) -> FrameworkConfig:
    """Create KodCode configuration"""
    return FrameworkConfig(
        dataset_type=DatasetType.KODCODE,
        workspace_dir=workspace_dir,
        output_base_dir=output_dir,
        dataset_specific={
            "task_id_format": "Category_{num}_I",
            "log_file_pattern": "Category_{num}_I.json",
            "eval_result_format": "simple_boolean",
        }
    )


def create_mbpp_plus_config(workspace_dir: str, output_dir: str) -> FrameworkConfig:
    """Create MBPP+ configuration"""
    return FrameworkConfig(
        dataset_type=DatasetType.MBPP_PLUS,
        workspace_dir=workspace_dir,
        output_base_dir=output_dir,
        dataset_specific={
            "task_id_format": "Mbpp/{num}",
            "log_file_pattern": "mbpp_Mbpp_{num}.json",
            "eval_result_format": "evalplus",
        }
    )


def create_config_from_env() -> FrameworkConfig:
    """Create configuration from environment variables"""
    dataset_type = os.getenv("DATASET_TYPE", "kodcode")
    workspace_dir = os.getenv("WORKSPACE_DIR", "/path/to/workspace")
    output_dir = os.getenv("OUTPUT_DIR", "/path/to/output")
    
    if dataset_type == "kodcode":
        return create_kodcode_config(workspace_dir, output_dir)
    elif dataset_type == "mbpp_plus":
        return create_mbpp_plus_config(workspace_dir, output_dir)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}") 