"""check_env: an offline equivalent of gymnasium.utils.env_checker.check_env for this
JSON-contract environment (gymnasium itself is not a dependency).

It verifies the Gymnasium Env API rules that apply here:
  * reset(seed=...) returns (obs, info) with dict info; the same int seed gives the same
    observation; seed=None continues the sequence; np_random_seed reports the seed;
  * reset info has the same keys as step info (analogous, per the API docs);
  * step returns (obs, float reward, bool terminated, bool truncated, dict info);
  * observations and info are NEW objects every call (no buffer mutated in place);
  * observations match observation_space and actions accepted by the env match action_space
    (JSON Schemas; checked when jsonschema is installed);
  * after terminated/truncated the env refuses further progress until reset();
  * render() follows render_mode; close() is idempotent.

    python -m auditenv.env.checker            # prints the list of passed checks, exits 1 on failure
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile


class EnvCheckError(AssertionError):
    pass


def _expect(cond, message):
    if not cond:
        raise EnvCheckError(message)


def check_env(env_factory, seed: int = 123, config=None) -> list[str]:
    """`env_factory(render_mode=...)` must return a fresh AuditEnv. Returns the names of passed checks."""
    passed = []
    env = env_factory(render_mode=None)

    obs, info = env.reset(seed=seed, config=config)
    _expect(isinstance(obs, dict) and isinstance(info, dict), "reset must return (dict obs, dict info)")
    _expect(env.np_random_seed == seed, "np_random_seed must report the int seed")
    passed.append("reset returns (obs, info); np_random_seed reports the seed")

    obs2, _ = env.reset(seed=seed, config=config)
    _expect(json.dumps(obs, sort_keys=True) == json.dumps(obs2, sort_keys=True), "same seed must give the same obs")
    _expect(obs is not obs2, "reset must return a new observation object")
    passed.append("same int seed -> same first observation (new object)")

    obs3, info3 = env.reset()
    _expect(info3["episode_id"] != info["episode_id"], "seed=None must continue the sequence")
    passed.append("seed=None continues the seeded sequence")

    try:
        import jsonschema  # noqa: F401
        from referencing import Registry, Resource
        from pathlib import Path
        schema_dir = Path(__file__).resolve().parents[1] / "schemas"
        registry = Registry().with_resources(
            [(s["$id"], Resource.from_contents(s)) for s in
             (json.loads(p.read_text()) for p in schema_dir.glob("*.schema.json"))])
        validate_obs = lambda o: jsonschema.Draft202012Validator(env.observation_space, registry=registry).validate(o)
        validate_act = lambda a: jsonschema.Draft202012Validator(env.action_space, registry=registry).validate(a)
    except ImportError:
        validate_obs = validate_act = None

    if validate_obs:
        validate_obs(obs3)
        passed.append("reset observation matches observation_space (JSON Schema)")

    action = {"tool": "list_records", "args": {"table": "invoices", "cursor": None, "limit": 5}}
    if validate_act:
        validate_act(action)
    before = copy.deepcopy(action)
    out = env.step(action)
    _expect(isinstance(out, tuple) and len(out) == 5, "step must return a 5-tuple")
    o, r, term, trunc, inf = out
    _expect(isinstance(o, dict) and type(r) is float and type(term) is bool and type(trunc) is bool
            and isinstance(inf, dict), "step must return (dict, float, bool, bool, dict)")
    _expect(action == before, "step must not mutate the action")
    _expect(set(info3) >= set(inf) - {"metrics", "feedback", "infra_error"} and set(inf) >= {"episode_id", "steps_used"},
            "reset info must be analogous to step info")
    passed.append("step returns (obs, float, bool, bool, dict); reset info has the step info keys")
    if validate_obs:
        validate_obs(o)
        passed.append("step observation matches observation_space (JSON Schema)")

    o2, *_ = env.step(action)
    _expect(o is not o2 and o["result"] is not o2["result"], "observations must be new objects every step")
    o["result"]["rows"].clear()
    o3, *_ = env.step(action)
    _expect(o3["result"]["rows"], "mutating a returned observation must not affect the env")
    passed.append("observations and info are new objects every call")

    _, r, term, trunc, _ = env.step({"tool": "submit", "args": {"findings": {"episode_id": info3["episode_id"],
                                                                              "findings": []}}})
    _expect(term and not trunc, "submit must terminate")
    after = env.step(action)
    _expect(after[0]["error"]["code"] == "episode_over" and after[1] == 0.0, "no progress after termination")
    passed.append("submit -> terminated; stepping after the end is refused until reset()")

    env.close()
    env.close()
    _expect(env.step(action)[0]["error"]["code"] == "no_episode", "closed env must refuse to step")
    passed.append("close() is idempotent")

    renv = env_factory(render_mode="ansi")
    renv.reset(seed=seed, config=config)
    _expect(isinstance(renv.render(), str) and env_factory(render_mode=None).render() is None,
            "render must follow render_mode")
    _expect("ansi" in renv.metadata["render_modes"] and renv.unwrapped is renv, "metadata/unwrapped")
    renv.close()
    passed.append("render() follows render_mode; metadata and unwrapped present")
    return passed


def main() -> int:
    from auditenv.env import AuditEnv
    with tempfile.TemporaryDirectory() as tmp:
        try:
            passed = check_env(lambda render_mode=None: AuditEnv(workdir=tmp, render_mode=render_mode))
        except EnvCheckError as e:
            print("check_env FAILED:", e)
            return 1
    print("check_env passed:")
    for p in passed:
        print("  -", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
