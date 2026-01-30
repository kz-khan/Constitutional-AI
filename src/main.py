from config import SNAPSHOT_DIR, CONSTITUTION_FILE_PATH, INIT_GENERATION_MODE,CRITIQUE_MODE,REVISION_MODE, CURRENT_ITERATION
from model_wrapper import ModelWrapper
from constitutional_critic import ConstitutionalCritic
from constitutional_ai_pipeline import ConstitutionalAIPipeline
from sampling import SamplingParams
import os

def main():
    # -------------------------------------------------
    # Load constitution
    # -------------------------------------------------
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    constitution_file = os.path.join(base_dir, CONSTITUTION_FILE_PATH)
    if os.path.exists(constitution_file):
        with open(constitution_file, "r", encoding="utf-8") as f:
            constitution = f.read()
        print(f"[INFO] Loaded constitution from {constitution_file}")
    else:
        print(f"[WARNING] Constitution file not found at {constitution_file}. Using default system prompt.")
        constitution = "You are a helpful, harmless, and honest AI assistant."

    # -------------------------------------------------
    # Load model + tokenizer (via ModelWrapper)
    # -------------------------------------------------
    model_name_or_path = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    # model_name_or_path = os.path.join(base_dir, SNAPSHOT_DIR)
    dtype = os.getenv("VLLM_DTYPE", "half")  # half for A40, bf16 for A100
    max_model_len = int(os.getenv("MAX_MODEL_LEN", "8192"))
    gpu_mem_util = float(os.getenv("GPU_MEM_UTIL", "0.90"))
    tp = int(os.getenv("TP", "1"))
    enforce_eager = os.getenv("ENFORCE_EAGER", "1") == "1"
    cache_dir=os.getenv("VLLM_DOWNLOAD_DIR", "./cache")

    model_wrapper = ModelWrapper(
        model_name_or_path=model_name_or_path,
        cache_dir=cache_dir,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_mem_util,
        dtype=dtype,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
    )

    # -------------------------------------------------
    # Sampling parameters
    # -------------------------------------------------
    sampling_params = SamplingParams(
        max_tokens=200,
        temperature=0.2,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.0,
    )

    # -------------------------------------------------
    # CAI components
    # -------------------------------------------------
    critic = ConstitutionalCritic(
        constitution=constitution,
        model_wrapper=model_wrapper,
    )

    pipeline = ConstitutionalAIPipeline(
        critic=critic,
        sampling_params=sampling_params
    )

    print("[INFO] Starting Constitutional AI generation...")
    print(f"[INFO] Model: {model_name_or_path}")
    print(f"[INFO] MODE: Initial generation: {INIT_GENERATION_MODE}")
    print(f"[INFO] MODE: Critique: {CRITIQUE_MODE}")
    print(f"[INFO] MODE: Revision: {REVISION_MODE}")
    print(f"[INFO] Current Iteration: {CURRENT_ITERATION}")

    if INIT_GENERATION_MODE and CURRENT_ITERATION==0:
        pipeline.run_initial_generation()
    elif CRITIQUE_MODE:
        pipeline.run_critique()
    elif REVISION_MODE:
        pipeline.run_revision()

    print("[INFO] Constitutional AI generation completed.")

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

if __name__ == "__main__":
    main()



