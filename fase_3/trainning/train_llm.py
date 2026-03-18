"""
Fine-tuning de LLM com QLoRA (4-bit) + LoRA (PEFT).

Utiliza o dataset no formato instruction/output, tokenização no padrão
Instruction/Response, e salva apenas os adaptadores LoRA.
Compatível com GPU (CUDA/MPS). Requer HF_TOKEN para modelos gated (como os LLaMA da Meta).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback

# Importação do config da fase_3 (permite rodar de dentro de fase_3)
try:
    from config.settings import settings
except ImportError:
    import sys
    BASE = Path(__file__).resolve().parent.parent
    if str(BASE) not in sys.path:
        sys.path.insert(0, str(BASE))
    from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("train_llm")

# Detecção de dispositivo: 4-bit (QLoRA) só com CUDA (bitsandbytes acelera só em GPU NVIDIA)
# Em MPS (Apple) ou CPU, 4-bit cai em CPU e fica muito lento; melhor carregar em bf16/fp16 e usar GPU (MPS)
def _get_device_info() -> tuple[str, bool]:
    """Retorna (device_name, use_4bit). use_4bit=True apenas com CUDA."""
    if torch.cuda.is_available():
        return "cuda", True
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps", False
    return "cpu", False

INSTRUCTION_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"
RESPONSE_START_MARKER = "\n\n### Response:\n"


def _get_response_start_char_index(text: str) -> int:
    """Retorna o índice do primeiro caractere da resposta (após '### Response:\\n')."""
    idx = text.find(RESPONSE_START_MARKER)
    if idx == -1:
        return len(text)
    return idx + len(RESPONSE_START_MARKER)


def _build_full_text(instruction: str, output: str) -> str:
    return INSTRUCTION_TEMPLATE.format(instruction=instruction) + output


def get_bnb_config() -> BitsAndBytesConfig:
    """Configuração de quantização 4-bit para QLoRA."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config() -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
    )


def load_model_and_tokenizer(
    model_name: str,
    bnb_config: BitsAndBytesConfig | None,
    use_flash_attention: bool = False,
    device_name: str | None = None,
    use_4bit: bool | None = None,
) -> tuple[Any, Any]:
    """
    Carrega modelo e tokenizer. Com CUDA: 4-bit (QLoRA) + LoRA. Com MPS/CPU: modelo em bf16 + LoRA (sem quant).
    Sempre usa o melhor dispositivo disponível (cuda > mps > cpu).
    """
    if device_name is None or use_4bit is None:
        device_name, use_4bit = _get_device_info()
    logger.info("Dispositivo: %s | QLoRA 4-bit: %s", device_name, use_4bit)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    use_accelerator = device_name in ("cuda", "mps")
    device_map = "auto" if use_accelerator else None

    if use_4bit and bnb_config is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN"),
            attn_implementation="flash_attention_2" if use_flash_attention else "eager",
        )
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
    else:
        dtype = torch.bfloat16 if torch.cuda.is_available() or (getattr(torch.backends, "mps", None) is not None) else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN"),
            attn_implementation="eager",
        )
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    # Aplicar LoRA
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def tokenize_and_mask_labels(
    examples: dict[str, list],
    tokenizer: Any,
    max_length: int,
    response_start_marker: str = RESPONSE_START_MARKER,
) -> dict[str, Any]:
    """
    Tokeniza instruction/output e gera labels com -100 na parte da instrução,
    para que o loss seja calculado apenas na resposta (Causal LM).
    """
    instructions = examples["instruction"]
    outputs = examples["output"]

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for instruction, output in zip(instructions, outputs):
        full_text = _build_full_text(instruction, output)
        response_start_char = _get_response_start_char_index(full_text)

        enc = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
            return_offsets_mapping=True,
        )

        input_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]

        labels = []
        for (start, end), token_id in zip(offsets, input_ids):
            if end <= response_start_char:
                labels.append(-100)
            else:
                labels.append(token_id)

        input_ids_list.append(input_ids)
        attention_mask_list.append(enc["attention_mask"])
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }


class DataCollatorCausalLMWithLabels:
    """
    Cola batches garantindo que input_ids, attention_mask e labels tenham
    o mesmo comprimento (padding em labels com -100 para não contribuir no loss).
    O DataCollatorForLanguageModeling padrão não preenche 'labels', gerando
    ValueError ao converter listas de tamanhos diferentes em tensor.
    """

    def __init__(
        self,
        tokenizer: Any,
        pad_to_multiple_of: int | None = 8,
    ):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of is not None:
            max_len = (
                (max_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )
        batch: dict[str, list[Any]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch["input_ids"].append(
                f["input_ids"] + [pad_token_id] * pad_len
            )
            batch["attention_mask"].append(
                f["attention_mask"] + [0] * pad_len
            )
            batch["labels"].append(
                f["labels"] + [-100] * pad_len
            )
        return {
            k: torch.tensor(v, dtype=torch.long)
            for k, v in batch.items()
        }


def get_dataset(
    data_path: str | Path,
    tokenizer: Any,
    max_length: int,
    max_samples: int | None = None,
) -> Any:
    """
    Carrega dataset JSONL (instruction/output) e aplica tokenização com labels.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")

    dataset = load_dataset("json", data_files=str(path), split="train")
    if max_samples is not None and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
        logger.info(f"Usando subconjunto de {max_samples} amostras para treino.")

    tokenized = dataset.map(
        lambda batch: tokenize_and_mask_labels(batch, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizando",
    )
    return tokenized


class SavePeftCallback(TrainerCallback):
    """Callback que garante salvamento apenas dos pesos PEFT (LoRA)."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)

    def on_train_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is not None and isinstance(model, PeftModel):
            out = self.output_dir / "adapter_final"
            out.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out)
            logger.info(f"Adaptadores LoRA salvos em: {out}")


def run_training(
    *,
    model_name: str | None = None,
    train_file: str | Path | None = None,
    output_dir: str | Path = "output_llm",
    max_length: int = 512,
    max_samples: int | None = None,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 16,
    num_train_epochs: int = 3,
    learning_rate: float = 2e-5,
    save_strategy: str = "epoch",
    use_flash_attention: bool = False,
) -> Path:
    """
    Executa o pipeline de fine-tuning: carrega modelo, dataset, treina e salva LoRA.

    Returns:
        Caminho do diretório onde os adaptadores foram salvos.
    """
    model_name = model_name or settings.MODEL_NAME
    train_file = Path(train_file or settings.TRAIN_FILE)
    output_dir = Path(output_dir)

    device_name, use_4bit = _get_device_info()
    logger.info(
        "Carregando modelo e tokenizer (%s)...",
        "QLoRA + LoRA (4-bit)" if use_4bit else "LoRA em precisão reduzida (GPU)",
    )
    bnb_config = get_bnb_config() if use_4bit else None
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        bnb_config,
        use_flash_attention=use_flash_attention,
        device_name=device_name,
        use_4bit=use_4bit,
    )

    logger.info("Carregando e tokenizando dataset...")
    dataset = get_dataset(train_file, tokenizer, max_length, max_samples)

    # MPS (Apple Silicon) não suporta pin_memory; desativar evita o warning
    mps_available = False
    if getattr(torch.backends, "mps", None) is not None:
        mps_available = torch.backends.mps.is_available()
    dataloader_pin_memory = torch.cuda.is_available() and not mps_available

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy=save_strategy,
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=dataloader_pin_memory,
    )

    # Collator customizado: preenche também 'labels' com -100 (o padrão só preenche input_ids/attention_mask)
    data_collator = DataCollatorCausalLMWithLabels(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=[SavePeftCallback(output_dir)],
    )

    trainer.train()
    # Salvar adaptadores no final (além do callback por epoch)
    adapter_path = output_dir / "adapter_final"
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_path)
    logger.info(f"Fine-tuning concluído. Adaptadores LoRA em: {adapter_path}")
    return adapter_path


def main() -> None:
    """Ponto de entrada para execução direta do script."""
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tuning de LLM com QLoRA + LoRA")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Nome do modelo HuggingFace (default: {settings.MODEL_NAME})",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help=f"Caminho do JSONL de treino (default: {settings.TRAIN_FILE})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_llm",
        help="Diretório de saída para checkpoints e adaptador final",
    )
    parser.add_argument("--max-length", type=int, default=512, help="Comprimento máximo em tokens")
    parser.add_argument("--max-samples", type=int, default=None, help="Limitar número de amostras (debug)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size por dispositivo")
    parser.add_argument("--grad-accum", type=int, default=16, help="Passos de acumulação de gradiente")
    parser.add_argument("--epochs", type=int, default=3, help="Número de épocas")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--flash-attn", action="store_true", help="Usar Flash Attention 2 (requer instalação)")
    args = parser.parse_args()

    run_training(
        model_name=args.model,
        train_file=args.data,
        output_dir=args.output_dir,
        max_length=args.max_length,
        max_samples=args.max_samples,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        use_flash_attention=args.flash_attn,
    )


if __name__ == "__main__":
    main()
