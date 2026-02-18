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

# Uncomment and use one block

# --------- Iteration 0 initial generation ---------------
INIT_GENERATION_MODE = True
CRITIQUE_MODE = False
REVISION_MODE = False
CURRENT_ITERATION = 0

# --------- Iteration 0 initial critique ---------------

# INIT_GENERATION_MODE = False
# CRITIQUE_MODE = True
# REVISION_MODE = False
# CURRENT_ITERATION = 0

# --------- Iteration 0 initial revision ---------------

# INIT_GENERATION_MODE = False
# CRITIQUE_MODE = False
# REVISION_MODE = True
# CURRENT_ITERATION = 0

# --------- Iteration 1 critique ---------------

# INIT_GENERATION_MODE = False
# CRITIQUE_MODE = True
# REVISION_MODE = False
# CURRENT_ITERATION = 1

# --------- Iteration 1 revision ---------------

# INIT_GENERATION_MODE = False
# CRITIQUE_MODE = False
# REVISION_MODE = True
# CURRENT_ITERATION = 1

# --------- Iteration 2 critic ---------------

# INIT_GENERATION_MODE = False
# CRITIQUE_MODE = True
# REVISION_MODE = False
# CURRENT_ITERATION = 2

# --------- Iteration 2 revision ---------------

# INIT_GENERATION_MODE = False
# CRITIQUE_MODE = False
# REVISION_MODE = True
# CURRENT_ITERATION = 2

# --------- Iteration 3 critic ---------------

# INIT_GENERATION_MODE = False
# CRITIQUE_MODE = True
# REVISION_MODE = False
# CURRENT_ITERATION = 3

# --------- Iteration 3 revision ---------------

# INIT_GENERATION_MODE = False
# CRITIQUE_MODE = False
# REVISION_MODE = True
# CURRENT_ITERATION = 3