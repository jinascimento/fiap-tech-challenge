# FIAP Tech Challenge - Fase 3

Este repositório contém o projeto desenvolvido para a `Fase 3` do curso de Pós-Graduação em IA. O foco desta etapa é um assistente virtual médico.

## Fine-tuning de LLM (QLoRA + LoRA)

O treinamento usa o dataset pré-processado em `data/dataset_medico_treinamento.jsonl` (formato `instruction` / `output`), com quantização 4-bit (QLoRA), LoRA em `q_proj` e `v_proj`, e salva apenas os adaptadores ao final.

### Dependências

Instale as dependências do projeto (inclui `transformers`, `peft`, `bitsandbytes`, `accelerate`, `torch`, `datasets`):

```bash
# Na raiz do repositório
uv sync
# ou: pip install -e .
```

### Token Hugging Face

O modelo padrão configurado é o `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, que é **aberto** e geralmente não exige acesso gated. Ainda assim, é recomendável configurar um token de leitura para o Hugging Face (caso use outros modelos no futuro):

1. Crie um token em [Hugging Face → Settings → Access Tokens](https://huggingface.co/settings/tokens).
2. Configure no ambiente ou em `.env` na pasta `fase_3`:

```bash
export HF_TOKEN="seu_token_aqui"
# ou crie fase_3/.env com: HF_TOKEN=seu_token_aqui
```

O `main.py` já define `HF_TOKEN` a partir de `config.settings` (e de variáveis de ambiente).

### Execução

**Recomendado:** ativar o ambiente e rodar a partir da pasta `fase_3`:

```bash
cd fase_3
```

- **Apenas pipeline** (gerar/atualizar o JSONL de treino):

```bash
python main.py --pipeline
```

- **Apenas fine-tuning** (usa o JSONL já existente em `data/dataset_medico_treinamento.jsonl`):

```bash
python main.py --train
```

- **Pipeline + fine-tuning em sequência:**

```bash
python main.py --full
```

- **Treinamento direto pelo script** (mais opções):

```bash
python trainning/train_llm.py --data data/dataset_medico_treinamento.jsonl --output-dir output_llm --epochs 3
```

Parâmetros úteis do `train_llm.py`:

- `--model`: modelo HuggingFace (default: valor em `config.settings`, ex. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`).
- `--data`: caminho do JSONL.
- `--output-dir`: diretório de checkpoints e adaptador final.
- `--max-length`: comprimento máximo em tokens (default: 512).
- `--max-samples`: limita amostras (útil para debug).
- `--batch-size`, `--grad-accum`, `--epochs`, `--lr`: hiperparâmetros de treino.
- `--flash-attn`: usa Flash Attention 2 (requer `flash-attn` instalado e GPU compatível).

### Inferência com o modelo fine-tunado

Após o fine-tuning, os adaptadores LoRA são salvos (por padrão) em algo como `fase_3/output_llm/adapter_final/` ou no diretório informado em `--output-dir`.

Para rodar uma inferência simples via script de exemplo:

```bash
cd fase_3

# Ajuste o caminho abaixo para o diretório de adaptadores que você treinou
sed -i '' 's|ADAPTER_DIR = "output_llm_r16_lr5e-6/adapter_final"|ADAPTER_DIR = "output_llm/adapter_final"|' infer_llm.py

python infer_llm.py

ou com uv:

uv run python infer_llm.py
```

O script `infer_llm.py`:

- Carrega o modelo base definido em `BASE_MODEL` (por padrão `TinyLlama/TinyLlama-1.1B-Chat-v1.0`).
- Carrega os adaptadores LoRA do diretório `ADAPTER_DIR`.
- Usa o template de prompt `### Instruction:\n{instruction}\n\n### Response:\n` para gerar a resposta.

Você pode editar diretamente, em `infer_llm.py`, os valores de:

- `BASE_MODEL`: caso queira trocar o modelo base da Hugging Face.
- `ADAPTER_DIR`: para apontar para outro treino (por exemplo `output_llm_r16_lr5e-6/adapter_final`).
- A variável `pergunta` no bloco `if __name__ == "__main__":` para testar diferentes instruções.

### GPU

O script usa `device_map="auto"` e `fp16` quando há CUDA. Para rodar em GPU:

```bash
# Verificar CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Saída

- Checkpoints por época em `fase_3/output_llm/` (ou no `--output-dir` informado).
- Adaptadores LoRA finais em `fase_3/output_llm/adapter_final/` (apenas pesos LoRA, prontos para carregar com `PeftModel.from_pretrained(base_model, "fase_3/output_llm/adapter_final")`).
