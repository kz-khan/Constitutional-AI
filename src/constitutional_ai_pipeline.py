import json
from pathlib import Path
from typing import Dict, Iterable, List
from constitutional_critic import ConstitutionalCritic
from config import OUTPUT_PATH, CURRENT_ITERATION
from data_manager import DataManager
from datetime import datetime
import os

class ConstitutionalAIPipeline:
    def __init__(
        self,
        critic: ConstitutionalCritic,
        sampling_params
    ):
        self.critic = critic
        self.sampling_params = sampling_params

    def run(self, user_prompt):
        print("[CAI] Generating initial response...")
        initial_response = self.critic.generate_initial(
            user_prompt, self.sampling_params
        )

        if not self.enable_critique:
            return {
                "initial_response": initial_response,
                "final_response": initial_response,
                "critiques": [],
                "revisions": [],
            }

        current_response = initial_response
        critiques, revisions = [], []

        for i in range(self.max_revisions):
            print(f"[CAI] Critique iteration {i + 1}/{self.max_revisions}")
            critique = self.critic.critique_response(
                user_prompt, current_response, self.sampling_params
            )
            critiques.append(critique)

            print(f"[CAI] Revision iteration {i + 1}/{self.max_revisions}")
            revised = self.critic.revise_response(
                user_prompt, current_response, critique, self.sampling_params
            )
            revisions.append(revised)

            current_response = revised

        return {
            "initial_response": initial_response,
            "final_response": current_response,
            "critiques": critiques,
            "revisions": revisions,
        }
    
    def run_initial_generation(self):       
        print("[CAI] Generating initial response...")
        # -------------------------------------------------
        # Load dataset
        # -------------------------------------------------
        data_mgr = DataManager()
        data_mgr.prepare(count=1960,force_download=False)

        dataset = data_mgr.load_local_prompts(count=None) # we will work with 100 samples
        print(f"[INFO] Loaded dataset with {len(dataset)} samples.")        

        rows = []
        for item in dataset:            
            pid = item["id"]
            prompt = item["prompt"]

            print(f"[CAI] Processing sample {prompt}...")

            # Construct user prompt
            user_prompt = (
                f"""Please answer the following question as directly and concisely as possible.
                Question:
                {prompt}"""
            )

            initial_response = self.critic.generate_initial(
                user_prompt, self.sampling_params
            )

            rows.append({
                "id": pid,
                "prompt": prompt,
                "candidate": initial_response,
                "meta": {
                    "model": os.getenv("MODEL_NAME", ""),
                    "ts": datetime.now().isoformat()
                }
            })

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"output_{timestamp}.jsonl"
        output_path = os.path.join(base_dir, OUTPUT_PATH, self.critic.model.model_name_or_path, "iteration_0", "initial_generations", output_file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.write_jsonl(output_path, rows)

    def run_critique(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        iteration_dir = "iteration_0"       
        gen_dir = os.path.join(base_dir, OUTPUT_PATH, self.critic.model.model_name_or_path, iteration_dir, "initial_generations")
        
        if CURRENT_ITERATION>0:
            prev_iteration = CURRENT_ITERATION - 1
            iteration_dir = f"iteration_{prev_iteration}"
            gen_dir = os.path.join(base_dir, OUTPUT_PATH, self.critic.model.model_name_or_path, iteration_dir, "revisions")

        print(f"Current iteration:{CURRENT_ITERATION}, critique using file: {gen_dir}")

        gen_files = sorted([f for f in os.listdir(gen_dir) if f.endswith(".jsonl")])
        if not gen_files:
            raise RuntimeError("No generation files found in outputs/gen/")
        gen_path = os.path.join(gen_dir, gen_files[-1])

        gens = self.read_jsonl(gen_path)

        crit_rows = []
        for row in gens:
            pid = row["id"]
            prompt = row["prompt"]

            if CURRENT_ITERATION>0:
                candidate = row["revised"]
            else:
                candidate = row["candidate"]

            critique_text = self.critic.critique_response(
                user_prompt=prompt, 
                response=candidate, 
                sampling_params = self.sampling_params)

            crit_rows.append({
                "id": pid,
                "prompt":prompt,
                "critique": critique_text,
                "prev_response":candidate,
                "meta": {
                    "gen_file": os.path.basename(gen_path),
                    "ts": datetime.now().isoformat()
                }
            })

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"output_{timestamp}.jsonl"
        current_iteration_dir = f"iteration_{CURRENT_ITERATION}" 
        output_path = os.path.join(base_dir, OUTPUT_PATH, self.critic.model.model_name_or_path, current_iteration_dir, "critiques", output_file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.write_jsonl(output_path, crit_rows)

    def run_revision(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        iteration_dir = f"iteration_{CURRENT_ITERATION}"
        crit_dir = os.path.join(base_dir, OUTPUT_PATH, self.critic.model.model_name_or_path, iteration_dir, "critiques")

        print(f"Current iteration:{CURRENT_ITERATION}, revision using file: {crit_dir}")

        crit_files = sorted([f for f in os.listdir(crit_dir) if f.endswith(".jsonl")])
        crit_path = os.path.join(crit_dir, crit_files[-1])
        crits = self.read_jsonl(crit_path)

        rev_rows = []
        for row in crits:
            pid = row["id"]
            prompt = row["prompt"]
            candidate = row["prev_response"]
            critique = row["critique"]

            revised_text = self.critic.revise_response(
                user_prompt=prompt, 
                original_response = candidate,
                critique= critique, 
                sampling_params = self.sampling_params)

            rev_rows.append({
                "id": pid,
                "prompt": prompt,
                "revised": revised_text,
                "meta": {
                    "crit_file": os.path.basename(crit_path),
                    "iteration": 1,
                    "ts": datetime.now().isoformat()
                }
            })

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"output_{timestamp}.jsonl"
        current_iteration_dir = f"iteration_{CURRENT_ITERATION}"
        output_path = os.path.join(base_dir, OUTPUT_PATH, self.critic.model.model_name_or_path, current_iteration_dir, "revisions", output_file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.write_jsonl(output_path, rev_rows)

    def save_constitutional_output(self, user_prompt, cai_result, output_file):
        import json, os

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        record = {
            "user_prompt": user_prompt,
            "cai_result": cai_result,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_jsonl(self, path, rows):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def read_jsonl(self, path):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

