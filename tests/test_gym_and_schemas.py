"""Gymnasium-style semantics and JSON Schema conformance of every artifact."""

import csv
import json
from pathlib import Path

import pytest

from auditenv.common import read_json, read_jsonl
from auditenv.env import AuditEnv, Recorder
from auditenv.generator import load_config
from tests.test_isolation import run

SCHEMAS = Path(__file__).resolve().parents[1] / "auditenv" / "schemas"


# ----------------------------------------------------------------- gymnasium
def test_seed_semantics_follow_gymnasium(tmp_path):
    def sequence(workdir):
        env = AuditEnv(workdir=workdir)
        ids = [env.reset(seed=11, config=load_config())[1]["episode_id"]]
        ids += [env.reset()[1]["episode_id"] for _ in range(2)]  # seed=None continues the sequence
        ids.append(env.reset(seed=11)[1]["episode_id"])             # an int seed restarts it
        return ids

    a, b = sequence(tmp_path / "a"), sequence(tmp_path / "b")
    assert a == b and a[0] == a[3] and len(set(a[:3])) == 3


def test_unseeded_reset_works_and_seed_type_is_checked(tmp_path):
    env = AuditEnv(workdir=tmp_path)
    obs, info = env.reset()
    assert obs["steps_remaining"] == 100 and info["episode_id"].startswith("ep-")
    with pytest.raises(TypeError):
        env.reset(seed="42")


def test_options_episode_dir_render_and_close(episodes):
    env = AuditEnv(render_mode="ansi")
    env.reset(options={"episode_dir": str(episodes["dirty"][0])})
    assert "reset" in env.render()
    env.step({"tool": "calculator", "args": {"expression": "2*2"}})
    assert "calculator -> ok" in env.render()
    assert AuditEnv().render() is None
    env.close()
    env.close()  # idempotent
    obs, reward, term, trunc, _ = env.step({"tool": "calculator", "args": {"expression": "1"}})
    assert obs["error"]["code"] == "no_episode" and reward == 0.0 and trunc and not term
    obs, *_ = AuditEnv().step(None)  # before any reset
    assert obs["error"]["code"] == "no_episode"
    assert env.unwrapped is env and env.metadata["render_modes"] == ["ansi"]
    assert env.action_space["$id"] == "action.schema.json"
    assert env.observation_space["$id"] == "observation.schema.json"


# ------------------------------------------------------------------- schemas
@pytest.fixture(scope="module")
def validator():
    jsonschema = pytest.importorskip("jsonschema")
    referencing = pytest.importorskip("referencing")
    resources = []
    for path in SCHEMAS.glob("*.schema.json"):
        schema = read_json(path)
        jsonschema.Draft202012Validator.check_schema(schema)
        resources.append((schema["$id"], referencing.Resource.from_contents(schema)))
    registry = referencing.Registry().with_resources(resources)

    def check(schema_id, instance, ref=None):
        schema = registry.contents(schema_id)
        if ref:
            schema = {"$ref": f"{schema_id}#/$defs/{ref}"}
        jsonschema.Draft202012Validator(schema, registry=registry).validate(instance)

    return check


def test_episode_files_match_schemas(episodes, validator):
    for ep in episodes["dirty"] + episodes["clean"]:
        validator("manifest.schema.json", read_json(ep / "public" / "manifest.json"))
        validator("truth.schema.json", read_json(ep / "private" / "truth.json"))
        for inv in read_jsonl(ep / "public" / "invoices.jsonl"):
            validator("records.schema.json", inv, "invoice")
        for pay in read_jsonl(ep / "public" / "payments.jsonl"):
            validator("records.schema.json", pay, "payment")
        with open(ep / "public" / "fx.csv") as f:
            for row in csv.DictReader(f):
                validator("records.schema.json", row, "fx_row")


def test_trajectories_actions_and_observations_match_schemas(episodes, validator, tmp_path):
    path = tmp_path / "traj.jsonl"
    rec = Recorder(path)
    for ep in episodes["dirty"][:2] + episodes["clean"][:1]:
        for policy in ("random", "reconcile", "flag_all", "oracle"):
            run(policy, ep, recorder=rec)
    rec.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        validator("trajectory_step.schema.json", row)
        if row["action"] is not None:
            validator("action.schema.json", row["action"])
        result = (row["observation"].get("result") or {})
        for r in result.get("rows", []):
            table = result["table"]
            ref = {"ledger": "ledger_row", "invoices": "invoice_summary", "payments": "payment", "fx": "fx_row"}[table]
            validator("records.schema.json", r, ref)
        if "document" in result:
            doc = {k: v for k, v in result["document"].items() if k != "kind"}
            validator("records.schema.json", doc, "invoice" if "line_items" in doc else "payment")
