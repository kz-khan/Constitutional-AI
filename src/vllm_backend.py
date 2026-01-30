
import torch


class VLLMBackend:
    def __init__(
        self,
        model_name_or_path,
        tokenizer,  # keep it if you use it elsewhere, but DON'T pass into LLM
        cache_dir,
        tensor_parallel_size,
        gpu_memory_utilization,
        dtype="half",
        max_model_len=8192,
        enforce_eager=False,
        quantization=None,
        trust_remote_code=False,
    ):
        from vllm import LLM
        
        self.tokenizer = tokenizer
        self.model = LLM(
            model=model_name_or_path,
            tokenizer=model_name_or_path,     # <-- FIX: must be string/path
            download_dir=cache_dir,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            quantization=quantization,
            trust_remote_code=trust_remote_code,
        )

    def generate(self, messages, sampling_params):
        # Build chat prompt text
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # --- Convert your custom SamplingParams -> vLLM SamplingParams ---
        from vllm import SamplingParams as VLLMSamplingParams

        sp = VLLMSamplingParams(
            max_tokens=getattr(sampling_params, "max_tokens", 200),
            temperature=getattr(sampling_params, "temperature", 0.7),
            top_p=getattr(sampling_params, "top_p", 1.0),
            top_k=getattr(sampling_params, "top_k", -1),
            repetition_penalty=getattr(sampling_params, "repetition_penalty", 1.0),
        )

        outputs = self.model.generate([prompt], sp, use_tqdm=False)
        return [outputs[0].outputs[0].text]
