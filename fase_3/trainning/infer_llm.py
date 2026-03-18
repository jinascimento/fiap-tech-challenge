import os

import torch
from config.settings import settings
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# ADAPTER_DIR = "fase_3/output_llm/adapter_final"

ADAPTER_DIR = settings.BASE_DIR / "output_llm" / "adapter_final"

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

INSTRUCTION_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        settings.MODEL_NAME,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN") or None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        settings.MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN") or None,
    )
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)

    model.eval()
    return model, tokenizer


def generate_answer(model, tokenizer, instruction: str) -> str:
    prompt = INSTRUCTION_TEMPLATE.format(instruction=instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    parts = decoded.split("### Response:\n", 1)
    return (parts[1] if len(parts) == 2 else decoded).strip()
