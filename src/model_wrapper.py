import os
import torch
from transformers import AutoTokenizer
import platform
from hf_backend import HFBackend
from vllm_backend import VLLMBackend

def is_windows() -> bool:
    return platform.system().lower().startswith("win")

class ModelWrapper:
    def __init__(
        self,
        model_name_or_path: str,
        cache_dir: str = "./cache",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        device: str | None = None,  

        # vLLM controls
        dtype: str | None = None,          # "half" or "bfloat16"
        max_model_len: int | None = None,  # e.g. 8192
        enforce_eager: bool = False,       # True for A40 stability
        quantization: str | None = None,   # "awq" / "gptq" if using quantized models
        trust_remote_code: bool = False,
    ):
        self.model_name_or_path = model_name_or_path

        if device is None:
            device = "cuda"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            use_fast=True,
            trust_remote_code=trust_remote_code,
        )

        # if is_windows():
        self.backend = HFBackend(
            model_name_or_path,
            self.tokenizer,
            self.device,
            cache_dir,
        )
        # else:
        #     if dtype is None:
        #         dtype = "half"
        #     if max_model_len is None:
        #         max_model_len = 8192

        #     # self.backend = VLLMBackend(
        #     #     model_name_or_path=model_name_or_path,
        #     #     tokenizer=self.tokenizer,  # OK as long as VLLMBackend doesn't pass it into LLM(...)
        #     #     cache_dir=cache_dir,
        #     #     tensor_parallel_size=tensor_parallel_size,
        #     #     gpu_memory_utilization=gpu_memory_utilization,
        #     #     dtype=dtype,
        #     #     max_model_len=max_model_len,
        #     #     enforce_eager=enforce_eager,
        #     #     quantization=quantization,
        #     #     trust_remote_code=trust_remote_code,
        #     # )

    def generate(self, messages, sampling_params):
        return self.backend.generate(messages, sampling_params)
