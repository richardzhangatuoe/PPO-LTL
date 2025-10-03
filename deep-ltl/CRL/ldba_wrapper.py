# src/envs/ldba_wrapper.py
from __future__ import annotations
from typing import Any, Tuple, Set, List

import re
import gymnasium as gym

# project: the LTL->LDBA constructive function (keep the original path/import)
# if there is a joblib.memory cache, keep it
from ltl.automata import ltl2ldba


# ---- extract the atomic propositions from the formula (exclude operators) ----
_ATOM_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_LTL_OPS = {"X","F","G","U","R","W","true","false","and","or","not","AND","OR","NOT"}
def _parse_atoms(formula: str) -> List[str]:
    tokens = set(_ATOM_RE.findall(formula))
    return sorted(t for t in tokens if t not in _LTL_OPS)


class LDBAWrapper(gym.Wrapper):
    """LTL constraint wrapper using LDBA."""

    def __init__(self, env: gym.Env, *, formula_str: str, finite: bool = False,
                 w_logic: float = 1.0, w_wall: float = 1.0):
        super().__init__(env)
        self.formula_str = formula_str
        self.finite = bool(finite)
        # cost weights (default 1, 1)
        self.w_logic = float(w_logic)
        self.w_wall = float(w_wall)

        # runtime state
        self.propositions: List[str] = []
        self.ldba = None
        self.terminate_on_acceptance = False
        self.ldba_state = None
        self.num_accepting_visits = 0

    # ---------- the original interface for completing the observation/advancing the automaton (keep or implement as needed) ----------
    def complete_observation(self, obs, info):
        # if there is a specific implementation, you can overwrite/replace this method
        pass

    def advance_automaton(self, active_props: Set[str]):
        """
        advance the automaton state according to active_props.
        compatible with multiple common LDBA interfaces: use ldba.step(state, label_set) first;
        then try ldba.delta(state, label_set) / ldba.transition(...)/ ldba.get_next_state(...)/ ldba.next(...)
        if the interface is not found, keep the original state (make the logic cost only depend on the wall)
        """
        if self.ldba is None:
            return
        label = set(active_props) if active_props else set()

        candidates = [
            "step", "delta", "transition", "get_next_state", "next"
        ]
        next_state = None
        for name in candidates:
            fn = getattr(self.ldba, name, None)
            if callable(fn):
                try:
                    # common signature: (state, label_set)
                    next_state = fn(self.ldba_state, label)
                except TypeError:
                    try:
                        # alternative signature: (label_set, state)
                        next_state = fn(label, self.ldba_state)
                    except Exception:
                        next_state = None
                except Exception:
                    next_state = None
            if next_state is not None:
                break

        if next_state is not None:
            # accept state statistics (if the interface is available)
            prev_state = self.ldba_state
            self.ldba_state = next_state
            try:
                in_accepting = False
                if hasattr(self.ldba, "is_accepting") and callable(getattr(self.ldba, "is_accepting")):
                    in_accepting = bool(self.ldba.is_accepting(self.ldba_state))
                elif hasattr(self.ldba, "accepting_states"):
                    in_accepting = self.ldba_state in getattr(self.ldba, "accepting_states")
                if in_accepting:
                    self.num_accepting_visits += 1
            except Exception:
                pass
            # mark the automaton state change (for logging)
            if prev_state is not next_state:
                # mark the automaton state change in the info (only update the internal flag)
                setattr(self, "_ldba_state_changed_step", True)
        else:
            # if the interface is not found, keep the original state
            setattr(self, "_ldba_state_changed_step", False)

    # ---------- the A^- of the current LTL step (negative propositions) ----------
    def _current_negative_sets(self) -> List[Set[str]]:
        """
        return the A^- of the current LTL step: list[set[str]]
        example: [{'yellow'}, {'blue','red'}] means: avoid yellow or avoid blue & red.
        first try to read from the LDBA exposed interface; if there is no available information, return empty (only keep the wall cost).
        """
        if self.ldba is None or self.ldba_state is None:
            return []

        # common conventions:
        # 1) ldba.get_negative_sets(state) -> Iterable[Iterable[str]]
        # 2) state.negative_sets / state.A_neg / state.forbidden / state.forbidden_sets
        # 3) ldba.labels.get(state, {}).get("A_neg", ...)
        try:
            fn = getattr(self.ldba, "get_negative_sets", None)
            if callable(fn):
                neg = fn(self.ldba_state)
                return [set(s) for s in neg] if neg is not None else []
        except Exception:
            pass

        # try to read from the state object
        for attr in ("negative_sets", "A_neg", "forbidden", "forbidden_sets"):
            try:
                neg = getattr(self.ldba_state, attr, None)
                if neg:
                    return [set(s) for s in neg]
            except Exception:
                continue

        # try to read from the ldba metadata dictionary
        try:
            labels = getattr(self.ldba, "labels", None)
            if isinstance(labels, dict):
                meta = labels.get(self.ldba_state, {})
                for key in ("A_neg", "negative_sets", "forbidden", "forbidden_sets"):
                    if key in meta and meta[key]:
                        return [set(s) for s in meta[key]]
        except Exception:
            pass

        return []

    # ---------- Gym API ----------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> Tuple[Any, dict]:
        obs, info = self.env.reset(seed=seed, options=options)

        # 1) read the AP list (ZoneEnv: colors)
        props = (obs.get("goal", {}) or {}).get("propositions", None) if isinstance(obs, dict) else None
        if not props:
            raise AssertionError("Env must provide obs['goal']['propositions'] (list[str]).")
        self.propositions = list(props)

        # 2) formula-AP consistency check (avoid HOA obscure errors)
        atoms = _parse_atoms(self.formula_str)
        missing = [a for a in atoms if a not in self.propositions]
        if missing:
            raise ValueError(
                f"Formula uses AP {missing} not in env propositions {self.propositions}. "
                f"For ZoneEnv, APs must be color names from this list."
            )

        # 3) construct the LDBA (keep your original cache/implementation)
        self.ldba = ltl2ldba(self.formula_str, self.propositions, simplify_labels=False)

        # 4) initialize the automaton runtime state (according to your project logic)
        #    the following two lines are typical; if your LDBA interface is different, please replace with equivalent call
        self.terminate_on_acceptance = getattr(self.ldba, "is_finite_specification", lambda: self.finite)()
        self.ldba_state = getattr(self.ldba, "initial_state", None)
        self.num_accepting_visits = 0

        # 5) optional: complete the observation
        self.complete_observation(obs, info)

        # 6) mark: the automaton state changed in this round
        info["ldba_state_changed"] = True
        info.setdefault("cost", 0.0)
        info.setdefault("cost_logic", 0.0)
        info.setdefault("cost_wall", 0.0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # A) the active propositions
        active = info.get("propositions", set())
        if not isinstance(active, (set, frozenset)):
            try:
                active = set(active)
            except Exception:
                active = set()

        # B) optional: advance the automaton
        setattr(self, "_ldba_state_changed_step", False)
        self.advance_automaton(active)

        # C) determine the A^- violation
        neg_sets = self._current_negative_sets()
        violated_neg = any(neg.issubset(active) for neg in neg_sets)

        # D) hard safety: hit the wall
        hit_wall = bool(info.get("hit_wall", False))

        # E) synthesize the cost (including weights and component logs)
        c_logic = 1.0 if violated_neg else 0.0
        c_wall = 1.0 if hit_wall else 0.0
        cost = self.w_logic * c_logic + self.w_wall * c_wall
        info["cost_logic"] = float(c_logic)
        info["cost_wall"] = float(c_wall)
        info["cost"] = float(cost)

        # F) mark the automaton state change (for upstream record)
        if getattr(self, "_ldba_state_changed_step", False):
            info["ldba_state_changed"] = True

        # G) optional: if in finite specification, terminate if accepting is reached, you can terminate here
        # if self.terminate_on_acceptance and self._in_accepting_state():
        #     terminated = True

        # H) optional: complete the observation
        self.complete_observation(obs, info)

        return obs, reward, terminated, truncated, info