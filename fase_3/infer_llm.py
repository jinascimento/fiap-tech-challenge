import os
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_DIR = "fase_3/output_llm/adapter_final"

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

INSTRUCTION_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN") or None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roda inferência com o modelo TinyLlama fine-tunado (LoRA)."
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        required=False,
        help="Pergunta/instrução para o modelo.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, tokenizer = load_model()

    if args.question:
        pergunta = args.question
    else:
        pergunta = (
            "Com base no contexto médico, responda em termos simples: "
            "What is high blood pressure (hypertension)?"
        )

    resposta = generate_answer(model, tokenizer, pergunta)
    print(resposta)