"""Track per-seed generation quotas and durable generation statistics."""

import copy
import json
import os
import random
import threading
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional


def _synchronized(method):
    """Serialize state mutations across worker threads."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class SeedManager:
    """Persist successful generations and reserve work without exceeding quotas."""

    def __init__(self, state_file="seed_state.json", target_count_per_seed=3, target_counts=None):
        self.state_file = state_file
        self.target_count_per_seed = max(0, int(target_count_per_seed))
        self.target_counts = target_counts or {}
        self._lock = threading.RLock()
        self._in_flight: Dict[str, int] = {}
        self.seed_states = self._load_state()

    def get_seed_id(self, seed_question: dict) -> str:
        """Return the stable identifier used for a seed question."""
        value = (
            seed_question.get("name")
            or seed_question.get("Name")
            or seed_question.get("id")
            or seed_question.get("ID")
            or seed_question.get("image_path")
            or seed_question.get("image")
            or "unknown"
        )
        return str(value)

    def get_target_count(self, seed_question: dict) -> int:
        """Return the configured quota for a seed."""
        seed_id = self.get_seed_id(seed_question)
        return max(0, int(self.target_counts.get(seed_id, self.target_count_per_seed)))

    def _load_state(self) -> Dict[str, Any]:
        if not os.path.exists(self.state_file):
            return {}
        try:
            with open(self.state_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_state(self) -> None:
        """Atomically persist state so an interrupted write cannot corrupt it."""
        state_snapshot = copy.deepcopy(self.seed_states)
        parent_dir = os.path.dirname(os.path.abspath(self.state_file))
        os.makedirs(parent_dir, exist_ok=True)
        temp_path = self.state_file + ".tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump(state_snapshot, handle, indent=2, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.state_file)
        except OSError:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def _ensure_seed(self, seed_id: str) -> Dict[str, Any]:
        return self.seed_states.setdefault(
            seed_id,
            {
                "total_attempts": 0,
                "success_count": 0,
                "consecutive_failures": 0,
                "last_success_time": None,
                "last_failure_time": None,
                "failure_history": [],
            },
        )

    @_synchronized
    def record_success(self, seed_question: dict) -> None:
        """Record a completed generation and release its reservation."""
        seed_id = self.get_seed_id(seed_question)
        state = self._ensure_seed(seed_id)
        self._in_flight[seed_id] = max(0, self._in_flight.get(seed_id, 0) - 1)
        state["total_attempts"] += 1
        state["success_count"] += 1
        state["consecutive_failures"] = 0
        state["last_success_time"] = datetime.now().isoformat()
        self._save_state()

    @_synchronized
    def record_failure(self, seed_question: dict, failure_reason: str = "") -> None:
        """Record a failed generation and release its reservation."""
        seed_id = self.get_seed_id(seed_question)
        state = self._ensure_seed(seed_id)
        self._in_flight[seed_id] = max(0, self._in_flight.get(seed_id, 0) - 1)
        state["total_attempts"] += 1
        state["consecutive_failures"] += 1
        state["last_failure_time"] = datetime.now().isoformat()
        state["failure_history"].append(
            {"time": datetime.now().isoformat(), "reason": failure_reason}
        )
        state["failure_history"] = state["failure_history"][-10:]
        self._save_state()

    @_synchronized
    def get_seed_stats(self, seed_question: dict) -> Dict[str, Any]:
        """Return generation statistics for one seed."""
        seed_id = self.get_seed_id(seed_question)
        state = self.seed_states.get(seed_id, {})
        attempts = state.get("total_attempts", 0)
        return {
            "seed_id": seed_id,
            "total_attempts": attempts,
            "success_count": state.get("success_count", 0),
            "consecutive_failures": state.get("consecutive_failures", 0),
            "success_rate": state.get("success_count", 0) / attempts if attempts else 0,
            "last_success_time": state.get("last_success_time"),
            "last_failure_time": state.get("last_failure_time"),
        }

    @_synchronized
    def get_all_stats(self) -> Dict[str, Any]:
        """Return aggregate statistics for the persisted state."""
        total_attempts = sum(s.get("total_attempts", 0) for s in self.seed_states.values())
        total_successes = sum(s.get("success_count", 0) for s in self.seed_states.values())
        return {
            "total_seeds": len(self.seed_states),
            "total_attempts": total_attempts,
            "total_successes": total_successes,
            "overall_success_rate": total_successes / total_attempts if total_attempts else 0,
        }

    @_synchronized
    def get_next_seed_to_generate(self, seed_questions: List[dict]) -> Optional[dict]:
        """Reserve and return the least-complete seed that still has quota."""
        available = []
        for seed in seed_questions:
            seed_id = self.get_seed_id(seed)
            state = self.seed_states.get(seed_id, {})
            success_count = state.get("success_count", 0)
            in_flight = self._in_flight.get(seed_id, 0)
            target_count = self.get_target_count(seed)
            if success_count + in_flight < target_count:
                available.append((seed, success_count, in_flight, target_count))
        if not available:
            return None

        min_progress = min(
            (success + in_flight) / target for _, success, in_flight, target in available
        )
        candidates = [
            seed
            for seed, success, in_flight, target in available
            if (success + in_flight) / target == min_progress
        ]
        selected = random.choice(candidates)
        seed_id = self.get_seed_id(selected)
        self._in_flight[seed_id] = self._in_flight.get(seed_id, 0) + 1
        return selected

    @_synchronized
    def get_generation_progress(self, seed_questions: List[dict]) -> Dict[str, Any]:
        """Return exact target and completion counts for the supplied seeds."""
        total_target = sum(self.get_target_count(seed) for seed in seed_questions)
        seed_progress = []
        for seed in seed_questions:
            seed_id = self.get_seed_id(seed)
            success_count = self.seed_states.get(seed_id, {}).get("success_count", 0)
            target_count = self.get_target_count(seed)
            seed_progress.append(
                {
                    "seed_id": seed_id,
                    "success_count": success_count,
                    "target_count": target_count,
                    "progress": f"{success_count}/{target_count}",
                    "completed": success_count >= target_count,
                }
            )
        total_generated = sum(item["success_count"] for item in seed_progress)
        return {
            "total_target": total_target,
            "total_generated": total_generated,
            "progress_percentage": total_generated / total_target * 100 if total_target else 0,
            "seed_progress": seed_progress,
            "completed_seeds": sum(item["completed"] for item in seed_progress),
            "total_seeds": len(seed_questions),
        }
