"""Genesis Audit Env: RL environment for training financial-audit agents.

Layout:
    generator/  builds a synthetic company, injects errors, writes public + private files
    env/        Gymnasium-style reset/step over the public files only
    grader/     scores findings against private truth; runs in a separate process
    policies/   baseline policies (random, reconcile, flag_all, oracle)
"""

# Versions (see README, "Версии"):
#   ENV_VERSION     episode content: generator, data and truth. Same seed + config + ENV_VERSION
#                   -> identical records. Written to every manifest.
#   SEED_SALT       part of every generation seed. Frozen, so a generator change that only touches
#                   edge configurations keeps the data of every other configuration identical
#                   across ENV_VERSIONs. Change it only to deliberately reshuffle all data.
#   API_VERSION     the step contract: tools, observations, info keys, reward channels. Same
#                   API_VERSION + episode + actions -> identical transitions and rewards.
#   SCHEMA_VERSION  the JSON schemas (manifest, findings, observation, ...).
#   package version (pyproject.toml) is the release of this repository.
ENV_VERSION = "1.5.1"
SEED_SALT = "1.2.0"
API_VERSION = "1.3.0"
SCHEMA_VERSION = "1.2"
