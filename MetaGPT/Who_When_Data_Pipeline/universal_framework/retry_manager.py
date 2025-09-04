#!/usr/bin/env python3
"""
Retry Manager - Unified retry logic management
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RetryStats:
    """Retry statistics"""
    total_retries: int = 0
    successful_retries: int = 0
    failed_retries: int = 0
    retry_counts: Dict[str, int] = None
    loaded_existing: int = 0
    
    def __post_init__(self):
        if self.retry_counts is None:
            self.retry_counts = {}
    
    def record_retry(self, task_id: str, success: bool):
        """Record retry"""
        self.total_retries += 1
        if success:
            self.successful_retries += 1
        else:
            self.failed_retries += 1
    
    def record_retry_count(self, task_id: str, count: int):
        """Record task retry count"""
        self.retry_counts[task_id] = count
    
    def record_loaded_existing(self, task_id: str):
        """Record tasks loaded from files"""
        self.loaded_existing += 1
        self.retry_counts[task_id] = 0
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "total_retries": self.total_retries,
            "successful_retries": self.successful_retries,
            "failed_retries": self.failed_retries,
            "retry_counts": self.retry_counts,
            "loaded_existing": self.loaded_existing,
        }


class RetryManager:
    """Retry manager"""
    
    def __init__(self, max_retries: int = 3, retry_delay: int = 2, backoff_factor: float = 1.5):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
        self.stats = RetryStats()
    
    async def retry_with_backoff(self, func: Callable, *args, **kwargs):
        """Retry mechanism with backoff"""
        for attempt in range(self.max_retries):
            try:
                result = await func(*args, **kwargs)
                # Record successful retry
                if attempt > 0:
                    self.stats.record_retry("", True)
                return result
            except Exception as e:
                # Record retry count
                self.stats.record_retry("", False)
                
                if attempt == self.max_retries - 1:
                    raise e
                
                delay = self.retry_delay * (self.backoff_factor ** attempt)
                print(f"Retry {attempt + 1}/{self.max_retries}, waiting {delay} seconds...")
                await asyncio.sleep(delay)
    
    def check_analysis_file_exists(self, task_id: str, analysis_type: str, workspace_dir: str) -> bool:
        """Check if analysis file already exists"""
        safe_task_id = task_id.replace('/', '_')
        
        # Build possible paths based on analysis type
        if analysis_type == "attack":
            possible_paths = [
                os.path.join(workspace_dir, f"{safe_task_id}_attack_analysis.json"),
                os.path.join(workspace_dir, f"attack_{safe_task_id}", f"{safe_task_id}_attack_analysis.json"),
                os.path.join(workspace_dir, f"attack_{safe_task_id}", "workspace", f"{safe_task_id}_attack_analysis.json"),
            ]
        elif analysis_type == "diagnose":
            possible_paths = [
                os.path.join(workspace_dir, f"{safe_task_id}_diagnosis.json"),
                os.path.join(workspace_dir, f"diagnose_{safe_task_id}", f"{safe_task_id}_diagnosis.json"),
                os.path.join(workspace_dir, f"diagnose_{safe_task_id}", "workspace", f"{safe_task_id}_diagnosis.json"),
            ]
        else:
            return False

        for path in possible_paths:
            if os.path.exists(path):
                return True
        return False
    
    def load_existing_analysis(self, task_id: str, analysis_type: str, workspace_dir: str) -> Optional[Dict[str, Any]]:
        """Load existing analysis file"""
        safe_task_id = task_id.replace('/', '_')
        
        # Build possible paths based on analysis type
        if analysis_type == "attack":
            possible_paths = [
                os.path.join(workspace_dir, f"{safe_task_id}_attack_analysis.json"),
                os.path.join(workspace_dir, f"attack_{safe_task_id}", f"{safe_task_id}_attack_analysis.json"),
                os.path.join(workspace_dir, f"attack_{safe_task_id}", "workspace", f"{safe_task_id}_attack_analysis.json"),
            ]
        elif analysis_type == "diagnose":
            possible_paths = [
                os.path.join(workspace_dir, f"{safe_task_id}_diagnosis.json"),
                os.path.join(workspace_dir, f"diagnose_{safe_task_id}", f"{safe_task_id}_diagnosis.json"),
                os.path.join(workspace_dir, f"diagnose_{safe_task_id}", "workspace", f"{safe_task_id}_diagnosis.json"),
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
        self.stats.record_retry_count(task_id, retry_count)
    
    def record_loaded_existing(self, task_id: str):
        """Record tasks loaded from files"""
        self.stats.record_loaded_existing(task_id)
    
    def print_stats(self):
        """Print retry statistics"""
        if self.stats.total_retries > 0 or self.stats.loaded_existing > 0:
            print(f"\n{'='*60}")
            print("📊 Retry Statistics")
            print(f"{'='*60}")
            print(f"Total retries: {self.stats.total_retries}")
            print(f"Successful retries: {self.stats.successful_retries}")
            print(f"Failed retries: {self.stats.failed_retries}")
            print(f"Loaded from files: {self.stats.loaded_existing}")

            if self.stats.retry_counts:
                retry_distribution = {}
                for task_id, count in self.stats.retry_counts.items():
                    retry_distribution[count] = retry_distribution.get(count, 0) + 1

                print(f"\nRetry count distribution:")
                for retry_count, task_count in sorted(retry_distribution.items()):
                    if retry_count == 0:
                        print(f"  0 retries (loaded from files): {task_count} tasks")
                    else:
                        print(f"  {retry_count} retries: {task_count} tasks")
    
    def save_stats(self, file_path: str):
        """Save statistics to file"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.stats.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"Retry statistics saved to: {file_path}")


# Global retry manager instance
_global_retry_manager = None

def get_retry_manager(max_retries: int = 3, retry_delay: int = 2, backoff_factor: float = 1.5) -> RetryManager:
    """Get global retry manager instance"""
    global _global_retry_manager
    if _global_retry_manager is None:
        _global_retry_manager = RetryManager(max_retries, retry_delay, backoff_factor)
    return _global_retry_manager

def reset_retry_manager():
    """Reset global retry manager"""
    global _global_retry_manager
    _global_retry_manager = None 