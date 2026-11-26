"""Language-model adapter: chat template, tool-call parser, multi-turn rollouts, GRPO groups.

The environment does not train models (the task does not ask for it, and the official check has no
GPU or paid API). This module is the boundary a trainer plugs into; the model itself is any
callable `generate(messages) -> str` (a local model, a server, or a scripted stand-in for tests).

    from auditenv.llm import run_episode, collect_group
    from auditenv.training import make_worker_env
    env = make_worker_env("runs/workers", worker_id=0)
    rollout = run_episode(my_generate, env, {"seed": 300001})
    group = collect_group(my_generate, env, episode_dir, k=8)     # GRPO: same episode, k samples

What it gives a trainer:
* system_prompt / render_observation: one stable chat template (compact JSON observations).
* parse_action: tolerant tool-call parser (code fences, prose around the JSON). A reply that holds
  no action becomes {"tool": "unparseable_output"}: the env answers with a bounded error and the
  step is spent, exactly as for an external agent, so parse failures are visible in the reward.
* action_json_schema(episode_id): JSON Schema for constrained decoding of ONE action, with the
  current episode_id pinned as a const in submit, so a decoder cannot produce the -1 format errors
  (float amount, a missing key). The pinned episode_id only keeps transcripts tidy: the grader treats
  episode_id as informational.
* run_episode: multi-turn loop; returns the transcript, the official reward, the experimental
  reward and the per-finding feedback codes (train mode).
* collect_group: k rollouts of the same episode with group-normalized advantages (GRPO).
  KL to a reference model, the optimizer and the model belong to the trainer (TRL, verl, ...).
* ScriptedTextPolicy: a scripted baseline behind the text interface (tests, SFT data).

Loss masking: train only on assistant turns; user turns are environment observations.
"""

from __future__ import annotations

import copy
import json
import re
import statistics
from pathlib import Path
from typing import Any, Callable

from auditenv.common import ERROR_TYPES, read_json
from auditenv.env.core import MAX_ACTION_BYTES, SCHEMA_DIR

Messages = list[dict]
Generate = Callable[[Messages], str]

SYSTEM_PROMPT = (
    "You are a financial audit agent working inside a sandboxed environment. Every turn, reply with "
    "exactly ONE JSON object and nothing else: {\"tool\": <name>, \"args\": {...}}. Tools:\n{tools}\n"
    "Rules: read records only through the tools; amounts are integer minor units; never compare amounts "
    "in different currencies without converting at the policy rate; report only real discrepancies "
    "with the evidence ids that prove them; finish with the submit tool before the step budget runs out "
    "(an episode that hits the limit counts as an empty answer). Submit exactly "
    "{\"episode_id\": <the episode_id of the manifest>, \"findings\": [{\"type\", \"affected_ids\", "
    "\"claimed_amount_minor\" (integer), \"currency\", \"evidence_ids\", \"explanation\"}]}."
)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def system_prompt(tools: dict) -> str:
    return SYSTEM_PROMPT.replace("{tools}", json.dumps(tools, separators=(",", ":"), sort_keys=True))


def render_observation(obs: dict) -> str:
    """Compact, deterministic JSON. Every observation is bounded by env.core.OBS_BYTE_BUDGET."""
    return json.dumps(obs, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def parse_action(text) -> tuple[dict, str | None]:
    """Extract the first JSON object with a string "tool". Returns (action, error-or-None)."""
    if not isinstance(text, str):
        return {"tool": "unparseable_output", "args": {}}, "model output is not text"
    text = text[: MAX_ACTION_BYTES * 2]
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    decoder = json.JSONDecoder()
    for chunk in candidates:
        for start in [i for i, ch in enumerate(chunk) if ch == "{"][:200]:
            try:
                obj, _ = decoder.raw_decode(chunk, start)
            except (ValueError, RecursionError):
                continue
            if isinstance(obj, dict) and isinstance(obj.get("tool"), str):
                args = obj.get("args", {})
                return {"tool": obj["tool"], "args": args if isinstance(args, dict) else {}}, None
    return {"tool": "unparseable_output", "args": {}}, "no JSON object with a 'tool' field"


def findings_json_schema(episode_id: str) -> dict:
    """The official findings schema, narrowed for constrained decoding: episode_id pinned, type limited
    to the four types, currency to three upper-case letters, no extra keys. Everything it admits is
    valid officially (the official schema accepts any string there and scores unknown types / other
    spellings as FP, and ignores extra keys)."""
    schema = copy.deepcopy(read_json(SCHEMA_DIR / "findings.schema.json"))
    schema["properties"]["episode_id"] = {"const": episode_id}
    props = schema["properties"]["findings"]["items"]["properties"]
    props["type"] = {"enum": list(ERROR_TYPES)}
    props["currency"] = {"type": "string", "pattern": "^[A-Z]{3}$"}
    # extra keys are valid officially (ignored), but a decoder should not spend tokens on them
    schema["additionalProperties"] = False
    schema["properties"]["findings"]["items"]["additionalProperties"] = False
    return schema


def action_json_schema(episode_id: str) -> dict:
    """Schema of one action for constrained decoding; submit carries the pinned findings schema."""
    schema = copy.deepcopy(read_json(SCHEMA_DIR / "action.schema.json"))
    findings = findings_json_schema(episode_id)
    for branch in schema["oneOf"]:
        if branch["properties"]["tool"].get("const") == "submit":
            branch["properties"]["args"] = {"type": "object", "additionalProperties": False,
                                            "required": ["findings"], "properties": {"findings": findings}}
    schema.pop("$id", None)
    return schema


def run_episode(generate: Generate, env, reset_kwargs: dict, max_parse_errors: int | None = None) -> dict:
    """One multi-turn episode. The env enforces the 100-step budget and the time limit."""
    obs, info = env.reset(**reset_kwargs)
    messages: Messages = [{"role": "system", "content": system_prompt(obs["tools"])},
                          {"role": "user", "content": render_observation(obs)}]
    parse_errors = 0
    reward, terminated, truncated, step_info = 0.0, False, False, info
    while not (terminated or truncated):
        text = generate(messages)
        action, problem = parse_action(text)
        parse_errors += problem is not None
        messages.append({"role": "assistant", "content": text if isinstance(text, str) else ""})
        obs, reward, terminated, truncated, step_info = env.step(action)
        messages.append({"role": "user", "content": render_observation(obs)})
        if max_parse_errors is not None and parse_errors > max_parse_errors and not (terminated or truncated):
            # Early stop for hopeless samples: spend the rest of the budget deterministically.
            while not (terminated or truncated):
                obs, reward, terminated, truncated, step_info = env.step({"tool": "unparseable_output", "args": {}})
    metrics = step_info.get("metrics") or {}
    return {"episode_id": env.episode_id, "messages": messages, "reward": reward,
            "experimental_reward": metrics.get("experimental_reward"), "feedback": step_info.get("feedback"),
            "terminated": terminated, "truncated": truncated, "steps": step_info.get("steps_used"),
            "parse_errors": parse_errors, "infra_error": step_info.get("infra_error")}


def collect_group(generate: Generate, env, episode_dir: str | Path, k: int,
                  reward_key: str = "reward") -> list[dict]:
    """GRPO-style group: k rollouts of the SAME episode; advantage = (r - mean) / (std + 1e-6).
    reward_key="reward" (official) or "experimental_reward" (training channel). Rollouts with an
    infrastructure error are dropped from the group (requeue them), never scored as -1."""
    rollouts = [run_episode(generate, env, {"episode_dir": episode_dir}) for _ in range(k)]
    rollouts = [r for r in rollouts if not r["infra_error"]]
    if not rollouts:
        return []
    rewards = [r[reward_key] if r[reward_key] is not None else r["reward"] for r in rollouts]
    mean, std = statistics.fmean(rewards), statistics.pstdev(rewards)
    for r, value in zip(rollouts, rewards):
        r["group_reward"] = value
        r["advantage"] = (value - mean) / (std + 1e-6)
    return rollouts


class ScriptedTextPolicy:
    """A scripted policy behind the text interface: reads the last observation from the messages and
    answers with the policy's action as JSON. Stateless between episodes: a new system message
    starts a new episode."""

    def __init__(self, factory: Callable[[dict], object]):
        """factory(first_observation) -> object with reset(obs, info), act(obs), observe(obs)."""
        self._factory = factory
        self._policy: Any = None

    def __call__(self, messages: Messages) -> str:
        obs = json.loads(messages[-1]["content"])
        if "manifest" in obs or self._policy is None:  # the reset observation starts a new episode
            self._policy = self._factory(obs)
            self._policy.reset(obs, {"episode_id": obs["manifest"]["episode_id"]})
        else:
            self._policy.observe(obs)
        return json.dumps(self._policy.act(obs), sort_keys=True)
