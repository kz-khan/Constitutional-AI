import os

DATA_DIR = "data"

CONSTITUTION_FILE_PATH = os.path.join(
    DATA_DIR, "constitution", "standard_rules.txt"
)
OUTPUT_PATH = "output"

SNAPSHOT_HASH = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
SNAPSHOT_DIR = f"/cache/models--Qwen--Qwen2.5-3B-Instruct/snapshots/{SNAPSHOT_HASH}"


MAX_REVISIONS = 3
# ---------------- generate and save files using this config, keep one true at a time ---------
INIT_GENERATION_MODE = True
CRITIQUE_MODE = False
REVISION_MODE = False
CURRENT_ITERATION = 0