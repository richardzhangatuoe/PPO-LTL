# src/safety_gym_wrapper.py
from __future__ import annotations
from typing import Any, List, Tuple, Set

import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperObsType


class SafetyGymWrapper(gym.Wrapper):
    """
    purpose:
      - expose the color AP list in obs["goal"]["propositions"] (for LDBA construction and consistency check)
      - expose the "current true" color propositions in info["propositions"] (inferred from info['cost_zones_*'])
      - expose whether the wall is hit in info["hit_wall"] (inferred from info['cost_ltl_walls']/info['cost_walls'])
    note:
      - do not do safety penalty reward shaping here (let PPO-Lagrange through cost constraints)
      - if you need to terminate immediately when hitting the wall, you can set terminated=True here
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.colors: List[str] = []
        # only enable the heuristic mapping when the underlying environment is Safety*; ZoneEnv does not enable
        try:
            env_id = getattr(getattr(self.env, "spec", None), "id", "") or ""
        except Exception:
            env_id = ""
        self._enable_safetygym_mapping = env_id.startswith("Safety")

    # ---------- tools ----------
    @staticmethod
    def _infer_colors_from_info(info: dict) -> List[str]:
        keys = [k for k in info.keys() if k.startswith("cost_zones_")]
        return sorted(k.replace("cost_zones_", "", 1) for k in keys)

    # ---------- Gym API ----------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)

        # 1) infer the color AP list (use default fallback if inference fails)
        if not self.colors:
            inferred = self._infer_colors_from_info(info)
            if not inferred and hasattr(self.env, "colors"):
                try:
                    inferred = list(getattr(self.env, "colors"))
                except Exception:
                    inferred = []
            if not inferred:
                inferred = ["blue", "green", "yellow", "red"]
            self.colors = inferred

        # 2) ensure the observation space is a dictionary format, and expose the AP list in obs["goal"]["propositions"]
        if not isinstance(obs, dict):
            obs = {"obs": obs}
        
        # ensure the goal field exists and contains propositions
        if "goal" not in obs:
            obs["goal"] = {}
        goal = dict(obs["goal"])
        goal["propositions"] = list(self.colors)
        obs["goal"] = goal
        
        # ensure the wall_sensor field exists
        if "wall_sensor" not in obs:
            obs["wall_sensor"] = np.array([0, 0, 0, 0], dtype=np.int64)

        # 3) initialize the info fields
        info["propositions"] = set()
        info.setdefault("hit_wall", False)
        return obs, info

    def step(self, action):
        out = self.env.step(action)
        # compatible with two return formats: 5-tuple and 6-tuple (obs, reward[, cost], terminated, truncated, info)
        if len(out) == 6:
            obs, reward, cost, terminated, truncated, info = out
            # unify the cost into info
            try:
                info["cost"] = float(cost)
            except Exception:
                info["cost"] = 0.0
        else:
            obs, reward, terminated, truncated, info = out
            info.setdefault("cost", 0.0)

        # force the reward to be a scalar
        try:
            import numpy as _np
            if not _np.isscalar(reward):
                reward = float(_np.asarray(reward).squeeze().item())
            else:
                reward = float(reward)
        except Exception:
            reward = float(reward)

        # ensure the observation space is a dictionary format
        if not isinstance(obs, dict):
            obs = {"obs": obs}
        
        # ensure the goal field exists and contains propositions
        if "goal" not in obs:
            obs["goal"] = {}
        goal = dict(obs["goal"])
        goal["propositions"] = list(self.colors)
        obs["goal"] = goal
        
        # ensure the wall_sensor field exists
        if "wall_sensor" not in obs:
            obs["wall_sensor"] = np.array([0, 0, 0, 0], dtype=np.int64)

        # (A) the active propositions (entering a color circle)
        active: Set[str] = set()
        # priority: the cost_zones_* signal originally from ZoneEnv
        if self.colors:
            for c in self.colors:
                if info.get(f"cost_zones_{c}", 0) > 0:
                    active.add(c)
        # only enable the heuristic mapping when the underlying environment is Safety*
        if self._enable_safetygym_mapping:
            # - any cost_* > 0 is considered blue (unsafe)
            try:
                any_cost = any((k.startswith("cost_") and float(info.get(k, 0)) > 0) for k in info.keys())
            except Exception:
                any_cost = False
            # explicitly compatible with common keys: cost_hazards / cost_sum
            try:
                any_cost = any_cost or float(info.get("cost_hazards", 0)) > 0 or float(info.get("cost_sum", 0)) > 0
            except Exception:
                pass
            if any_cost:
                active.add("blue")
            # - success/goal_met is considered green (safe)
            goal_hit = bool(info.get("success", False) or info.get("goal_met", False))
            goal_close = False
            try:
                gd = info.get("goal_dist", None)
                if gd is not None:
                    goal_close = float(gd) < 0.2
            except Exception:
                goal_close = False
            if goal_hit or goal_close:
                active.add("green")
        info["propositions"] = active

        # (B) hard safety: hit the wall (or general collision)
        hit_wall = bool(
            info.get("cost_ltl_walls", 0) > 0
            or info.get("cost_walls", 0) > 0
            or (self._enable_safetygym_mapping and (
                info.get("cost_collision", 0) > 0 or info.get("collision", 0) > 0
            ))
            or info.get("cost_hazards", 0) > 0
            or info.get("cost_sum", 0) > 0
        )
        info["hit_wall"] = hit_wall

        return obs, reward, terminated, truncated, info

    def get_propositions(self) -> list[str]:
        return sorted(self.colors)

    def get_possible_assignments(self):
        props = self.get_propositions()
        return [{p} for p in props]