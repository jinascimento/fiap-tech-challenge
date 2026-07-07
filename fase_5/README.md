# Modelagem de Ameacas STRIDE a partir de Diagramas de Arquitetura

MVP para o Hackathon FIAP Fase 5: dado um diagrama de arquitetura de software
(imagem), detectar os componentes e fluxos de comunicacao, reconstruir o
grafo da arquitetura e gerar automaticamente um Relatorio de Modelagem de
Ameacas baseado em STRIDE, com vulnerabilidades e contramedidas por
componente/fluxo.

Contexto completo do desafio: `IADT - Fase 5 - Hackaton.pdf`.
Analise exploratoria dos dois datasets (Hugging Face, autor Guilherme Santos):
`eda/RELATORIO_EDA.md`.

## Pipeline

```
imagem --[YOLOv8]--> componentes (32 classes)  ---\
                                                     >--> grafo dirigido --> motor de regras STRIDE --> relatorio.md
imagem --[YOLOv8-pose]--> setas (tail/tip)     ---/
```

1. **Deteccao de componentes** (`stride_vision/training/train_components.py`) -
   YOLOv8 sobre `dataset/stride-architecture-components-v1` (32 classes:
   atores, edge, compute, dados, seguranca, observabilidade, servicos
   externos e trust boundaries).
2. **Deteccao de fluxos** (`stride_vision/training/train_flows.py`) -
   YOLOv8-pose sobre `dataset/stride-architecture-flows-v1` (1 classe
   `flow_arrow` + 2 keypoints `tail`/`tip` que dao a direcao do fluxo).
3. **Reconstrucao do grafo** (`stride_vision/graph/reconstruct.py`) - associa
   cada keypoint `tail`/`tip` ao componente mais proximo, monta um grafo
   dirigido (`networkx.DiGraph`) e calcula, para cada no, quais
   `boundary_*` o contem e, para cada aresta, quais boundaries ela cruza.
4. **Motor de regras STRIDE** (`stride_vision/knowledge_base/`) - aplica
   `stride_rules.yaml` (mapeamento classe de componente -> ameacas
   STRIDE + contramedida, curado manualmente, ja que nenhum dos dois
   datasets cobre essa parte) sobre o grafo.
5. **Relatorio** (`stride_vision/report/generate.py`) - Markdown com
   componentes, fluxos, cruzamentos de boundary e ameacas por categoria
   STRIDE.

As etapas 3-5 ja funcionam hoje, sem depender de modelo treinado: aceitam
diretamente os arquivos de label `.txt` (formato YOLO / YOLO-pose) como
entrada, seja o ground-truth do dataset ou a saida de `model.predict(...)`.

Os modelos ja treinados (`models/components_yolov8n.pt`,
`models/flows_yolov8n_pose.pt`, ~12MB no total) **estao versionados no git**
-- quem clonar o repo so precisa instalar as dependencias e rodar a
inferencia, sem precisar baixar o dataset (8GB) nem retreinar.

## Setup rapido (so rodar a inferencia, sem treinar)

```bash
uv sync --python 3.14
# atencao: se seu Python default (pyenv etc.) nao tiver blake2b/blake2s no
# hashlib, o import do networkx quebra -- por isso fixamos --python 3.14
# gerenciado pelo proprio uv, que nao depende do pyenv do sistema.

.venv/bin/python -m stride_vision.infer \
  --image caminho/para/diagrama.jpg \
  --image-name "Nome da arquitetura" \
  --output report.md \
  --viz-output annotated.png   # opcional: imagem com as deteccoes desenhadas
```

`--components-weights`/`--flows-weights` sao opcionais e ja apontam por
padrao para `models/`. So precisa informa-los se estiver usando pesos de um
treino proprio (ex.: `runs/detect/runs/components/yolov8n_components/weights/best.pt`).

## Setup completo (para retreinar os modelos)

```bash
# dados: os dois datasets usam Git LFS -- ver eda/RELATORIO_EDA.md secao 3
brew install git-lfs && git lfs install
(cd dataset/stride-architecture-components-v1 && git lfs pull)
(cd dataset/stride-architecture-flows-v1 && git lfs pull)

uv sync --python 3.14
```

Gerar um relatorio a partir de labels ground-truth (sem modelo, so pra testar
o grafo/regras/relatorio):

```bash
.venv/bin/python -m stride_vision.pipeline \
  --components-label dataset/stride-architecture-components-v1/train/labels/<arquivo>.txt \
  --flows-label dataset/stride-architecture-flows-v1/train/labels/<arquivo>.txt \
  --image-name <nome> --output report.md
```

Treinar os modelos (rodou localmente em M4 via MPS, ~40min components + ~30min
flows com os parametros abaixo; ajuste `--imgsz`/`--epochs` conforme disco/tempo
disponivel, ou use `notebooks/train_colab.ipynb` para treinar com GPU na nuvem):

```bash
.venv/bin/python -m stride_vision.training.train_components --epochs 60 --imgsz 640 --batch 16 --device mps --patience 15
.venv/bin/python -m stride_vision.training.train_flows --epochs 60 --imgsz 640 --batch 16 --device mps --patience 15
```

Depois de treinar, para atualizar os pesos versionados em `models/`:

```bash
cp runs/detect/runs/components/yolov8n_components/weights/best.pt models/components_yolov8n.pt
cp runs/pose/runs/flows/yolov8n_flows/weights/best.pt models/flows_yolov8n_pose.pt
```

Corrigir o vazamento de split identificado na EDA (re-split por diagrama-base,
dry-run por padrao, `--apply` para copiar):

```bash
uv run --python 3.14 python3 -m stride_vision.data.resplit
```

## Resultados do treino (baseline, yolov8n, imgsz=640, 60 epocas + early stopping)

| Modelo | mAP50 | mAP50-95 | Observacoes |
|---|---|---|---|
| components (deteccao, 32 classes) | 0.822 | 0.644 | Classes com poucos exemplos ficaram fracas (`security_identity_provider` mAP50=0, `backup_service` 0.49) -- ver EDA. |
| flows (pose, tail/tip) | 0.547 (box) / 0.637 (pose) | 0.236 (box) / 0.519 (pose) | Early stopping na epoca 41 (melhor em 26); dataset menor (176 diagramas-base) e mais esparso. |

Testado nas duas arquiteturas de avaliacao do PDF do hackathon
(`reports/report_arquitetura1_aws.md`, `reports/report_arquitetura2_azure.md`):
os componentes identificados batem semanticamente bem com os diagramas reais
(ex.: Azure -- API Gateway, Entra ID, Logic Apps, Developer Portal; AWS --
Shield, CloudFront, WAF, os 3 ALBs, RDS, ElastiCache, KMS, CloudTrail, SES).

## Limitacoes conhecidas (ver EDA para detalhes)

- `actor_admin` nao tem nenhum exemplo anotado em todo o dataset; `integration_messaging`
  quase nenhum -- essas classes nao serao aprendidas pelo detector como esta.
- Split original tem ~4-5% de vazamento entre train/val/test (use `resplit.py`).
- Apenas 176/414 diagramas-base tem anotacao de fluxo -- o detector de setas
  treina/avalia num subconjunto menor que o de componentes.
- ~17-18% das setas anotadas em `flows` nao tem os dois keypoints visiveis,
  entao nao entram no grafo dirigido (ficam de fora da reconstrucao, contadas
  no relatorio como "ignoradas").
