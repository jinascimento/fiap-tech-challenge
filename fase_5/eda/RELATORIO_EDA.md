# EDA — Datasets STRIDE Architecture (Components + Flows)

Hackathon FIAP — Fase 5: *Modelagem de ameaças utilizando IA*

## 1. Contexto do desafio

A FIAP Software Security quer um MVP capaz de:

1. Interpretar automaticamente um diagrama de arquitetura de software (imagem), identificando os componentes (usuários, servidores, bases de dados, APIs etc.);
2. Gerar um **Relatório de Modelagem de Ameaças baseado em STRIDE**;
3. Construir/buscar um dataset de imagens de arquitetura e anotá-lo para treinar um modelo supervisionado de identificação de componentes;
4. Treinar o modelo;
5. Buscar, para cada componente identificado, as vulnerabilidades relacionadas e as contramedidas específicas para cada ameaça.

Avaliação: o código será testado contra arquiteturas de teste (diagramas fornecidos pela banca). Entregáveis: documentação do fluxo de desenvolvimento, vídeo de até 15 min, link do GitHub.

Os dois datasets baixados do Hugging Face (autor: Guilherme Santos, "Vision Architecture Analyzer") cobrem a parte de **percepção** do problema (passos 1–4): detectar os nós do diagrama (components) e as setas de fluxo entre eles (flows). A etapa 5 (vulnerabilidades/contramedidas por componente) **não está coberta por nenhum dos dois datasets** — é uma lacuna a preencher separadamente (ver seção 7).

## 2. Formato dos datasets

| | `stride-architecture-components-v1` | `stride-architecture-flows-v1` |
|---|---|---|
| Tarefa | Object detection (YOLO) | Pose estimation (YOLO-pose) |
| Classes | 32 (tipos de componente de arquitetura) | 1 (`flow_arrow`) |
| Anotação por objeto | `class x_center y_center w h` | bbox + 2 keypoints: `tail` (origem) e `tip` (destino), cada um com flag de visibilidade |
| Splits | train / val / test | train / val / test |

As 32 classes de `components` cobrem atores (`actor_user`, `actor_admin`), borda de rede (`edge_*`: CDN, WAF, gateway, DDoS protection...), computação (`compute_service`, `compute_worker`, `compute_load_balancer`), dados (`data_database`, `data_cache`, `data_storage`), segurança (`security_identity_provider`, `security_key_management`), observabilidade (`obs_monitoring`, `obs_audit`), serviços externos (`external_*`) e **trust boundaries** (`boundary_cloud`, `boundary_region`, `boundary_vpc_or_vnet`, `boundary_subnet_public/private`, `boundary_resource_group`, `boundary_autoscaling_group`). A presença explícita de classes de *boundary* é o que viabiliza identificar cruzamentos de fronteira de confiança — um dos sinais centrais do STRIDE.

O dataset `flows` reusa as mesmas imagens-base e identifica as setas de comunicação entre componentes, com keypoints ordenados (`tail`→`tip`) que dão a **direção** do fluxo — essencial para diferenciar, por exemplo, um fluxo de entrada externo (alvo de Spoofing/Tampering) de um fluxo interno.

A combinação pretendida pelos dois datasets é: detectar nós (components) + detectar arestas direcionadas (flows) sobre a mesma imagem → reconstruir um grafo da arquitetura → aplicar regras STRIDE sobre esse grafo (tipo de nó, cruzamento de boundary, direção do fluxo).

## 3. Achado crítico: conteúdo via Git LFS não estava baixado

Ambos os repositórios usam Git LFS para `.jpg/.jpeg/.png/.npy`. Como o `git-lfs` não estava instalado nesta máquina, o `git clone` original baixou **apenas os ponteiros LFS** (arquivos de ~130 bytes com `oid`/`size`), não o conteúdo real — qualquer tentativa de abrir essas imagens falhava silenciosamente (não eram imagens válidas). Os arquivos de anotação `.txt` (formato YOLO) **não usam LFS** e já estavam íntegros.

Correção aplicada nesta etapa: `brew install git-lfs`, `git lfs install`, `git lfs pull` nos dois repositórios. Volume real baixado: **~4 GB** (≈3.96 GB em `components`, ≈0.07 GB em `flows`). Confirmado pós-download: imagens abrem normalmente (ex.: JPEG 928×490, 3 canais) e os `.npy` carregam como arrays `uint8 (H, W, 3)`.

**Ação necessária para qualquer pessoa que clonar este projeto**: instalar `git-lfs` e rodar `git lfs pull` dentro de cada pasta de dataset antes de treinar — caso contrário, o pipeline vai "treinar" sobre ponteiros de texto.

## 4. Dataset `components` — achados

| Split | Arquivos de label | Diagramas-base únicos | Total de boxes | Classes presentes |
|---|---|---|---|---|
| train | 1490 | 292 | 9230 | 30/32 |
| val | 420 | 84 | 2400 | 25/32 |
| test | 230 | 46 | 1210 | 26/32 |

(414 diagramas-base únicos no total, em ~5 variações de augmentation cada: `_BW`, `_sharp`, `_contrast`, `_gamma_hi`, `_gamma_lo`, `_jpeg50`, `_blur1`, `_noise6`, `_degrade80`.)

**Inspeção visual** (`eda/outputs/sample_components_*.png`): as caixas e rótulos batem corretamente com os elementos do diagrama (ex.: `security_identity_provider` no ícone do Azure AD, `edge_gateway` no API Gateway, `integration_orchestrator` no Logic Apps, `boundary_resource_group` envolvendo o resource group) — qualidade de anotação boa nas amostras conferidas.

**Resolução das imagens** (train, n=1490): largura 518–1920px (média 1043px), altura 279–1221px (média 670px) — diagramas em resolução heterogênea, mas todos relativamente pequenos/médios (nenhum em 4K), compatível com input padrão de YOLO (640–1280px) sem perda séria de detalhe.

### 4.1 Desbalanceamento severo de classes (gráfico: `components_class_distribution.png`)

`compute_service` domina com 1530 boxes (train), enquanto várias classes têm dezenas ou **zero** exemplos:

- **`actor_admin`: 0 boxes em train, val E test** — classe presente no `data.yaml` mas sem nenhum exemplo anotado em todo o dataset. Inutilizável como está.
- **`integration_messaging`: 0 boxes em train e val** (não aparece como ausente no test, ou seja, tem exemplo só no test) — impossível de aprender, qualquer avaliação nessa classe será enganosa.
- Classes ausentes em **val**: `edge_portal`, `external_entry_point`, `integration_messaging`, `compute_worker`, `external_saas_service`, `communication_service`.
- Classes ausentes em **test**: `edge_portal`, `integration_messaging`, `compute_worker`, `security_key_management`, `obs_audit`.

Implicação prática: métricas por classe (mAP por classe) não vão existir/serão não confiáveis para essas classes nos respectivos splits. Vale considerar agrupar classes raras semanticamente próximas (ex.: poderia avaliar fundir `actor_admin` em `actor_user` ou tratar como "rare class" com data augmentation/oversampling direcionado) antes do treino.

### 4.2 Vazamento leve entre splits

Usando o nome base do diagrama (removendo prefixo aleatório e sufixo de augmentation), alguns diagramas-base aparecem em mais de um split:

- train ∩ val: 4 diagramas
- train ∩ test: 2 diagramas
- val ∩ test: 2 diagramas

É um vazamento pequeno (≈5% do val, ≈4% do test), mas real — o split foi feito por *arquivo* de imagem (incluindo variações de augmentation), não por *diagrama-base*. Recomenda-se, antes do treino final, re-splitar por diagrama-base para eliminar esse vazamento e ter uma validação honesta.

## 5. Dataset `flows` — achados

| Split | Arquivos de label | Diagramas-base únicos | Total de setas | Ambos keypoints visíveis | 1 keypoint ausente | Ambos ausentes |
|---|---|---|---|---|---|---|
| train | 590 | 116 | 3620 | 2840 (78%) | 170 (5%) | 610 (17%) |
| val | 180 | 36 | 1170 | 1010 (86%) | 130 (11%) | 30 (3%) |
| test | 130 | 26 | 650 | 480 (74%) | 50 (8%) | 120 (18%) |

Vazamento entre splits: praticamente nulo (0 entre train/val, 2 entre val/test).

**Implicação**: ~17–18% das setas em train/test não têm nenhum keypoint rotulado (apenas a caixa delimitadora da seta, sem origem/destino) — para essas, não dá para inferir direção do fluxo, só "existe comunicação entre algo perto de A e algo perto de B". Isso enfraquece a etapa de reconstrução de grafo *direcionado*, que é o que diferencia, no STRIDE, um fluxo de entrada (Spoofing/Tampering na borda) de um fluxo interno.

**Inspeção visual** (`eda/outputs/sample_flows_*.png`): a amostra conferida mostra um diagrama AWS com ~4-5 setas visíveis, mas só 2 estão anotadas no label (ambas sem keypoints de direção) — sinal de que a cobertura de anotação de `flows` é mais esparsa/incompleta que a de `components`, reforçando a necessidade de validar visualmente mais amostras antes de treinar.

## 6. Compatibilidade entre os dois datasets

Esta era a pergunta central: "em tese eles funcionam juntos". Resultado, comparando diagramas-base únicos:

- `components`: 414 diagramas-base únicos
- `flows`: 176 diagramas-base únicos
- **Interseção (tem componentes E fluxos anotados): 176** — ou seja, **100% dos diagramas de `flows` têm contraparte em `components`**, mas só **42% dos diagramas de `components` têm contraparte em `flows`** (238 diagramas só têm anotação de componentes, sem fluxos).
- Quando um diagrama existe nos dois datasets, o split (train/val/test) bate em 174/176 casos — só 2 divergências (`aws_solution_20260202_18` e sua variante `_bw`, que aparecem em splits diferentes entre os dois datasets).

**Conclusão prática**: os dois datasets *funcionam juntos*, mas a reconstrução completa do grafo (nós + arestas direcionadas) só é totalmente viável no subconjunto de **176 diagramas-base** (≈740 imagens contando augmentations) que têm as duas anotações. Os outros 238 diagramas de `components` ainda são úteis — só não para treinar/avaliar a etapa de detecção de fluxo — e podem ser usados normalmente para treinar o detector de componentes (modelo 1).

Como os splits dos dois datasets já vêm majoritariamente alinhados, dá para treinar os dois modelos (detector de componentes e detector de fluxo) cada um com seu split nativo, sem precisar re-splitar para juntar — só vale tratar os 2 diagramas com split divergente.

## 7. `.npy` é redundante em relação à imagem — mas não bit-idêntico

Cada imagem tem dois arquivos com o mesmo nome-base: `.jpg/.png` e `.npy` (array `(H, W, 3) uint8`). Comparando 5 pares amostrados, **não são bit-idênticos** (diferença média de 4.6 a 19.5 em escala 0-255; entre 8% e 36% dos pixels com diferença >10) — consistente com o `.npy` guardando o array antes da recompressão JPEG e o `.jpg` sendo uma recompressão com leve perda, não um arquivo totalmente diferente (mesmas dimensões, mesmo conteúdo visual, mesmo label). Para fins de treino, **usar apenas `.jpg/.png` é suficiente** — manter os dois formatos infla o dataset em ~2x no disco sem ganho (a perda JPEG nesse nível é irrelevante para detecção de ícones/retângulos em diagramas). Recomenda-se ignorar os `.npy` no pipeline de treino.

## 8. Lacuna: base de conhecimento de vulnerabilidades/contramedidas STRIDE

Nenhum dos dois datasets cobre o passo 5 do desafio (vulnerabilidades e contramedidas por componente). Essa base de conhecimento precisa ser construída/curada separadamente, por exemplo mapeando:

- cada classe de componente → categorias STRIDE aplicáveis (ex.: `data_database` → Tampering, Information Disclosure; `security_identity_provider` → Spoofing, Repudiation);
- cruzamento de `boundary_*` por um `flow_arrow` → reforça Spoofing/Tampering/Information Disclosure (cruzamento de fronteira de confiança é um gatilho clássico do STRIDE);
- fontes possíveis para popular essa base: catálogo STRIDE clássico (Microsoft), OWASP (Top 10 / Cheat Sheets), MITRE ATT&CK (Cloud Matrix), CSA Cloud Controls Matrix.

Esse trabalho é independente do treino dos modelos de visão e pode ser feito em paralelo.

## 9. Recomendações técnicas para as próximas etapas (não implementado nesta etapa)

1. **Higienizar o split** de `components` (re-splitar por diagrama-base, não por arquivo) para eliminar o vazamento train/val/test identificado na seção 4.2.
2. **Tratar classes raras/ausentes** (`actor_admin`, `integration_messaging` e as demais com poucos exemplos) — oversampling, fusão de classes semanticamente próximas, ou aceitar como limitação conhecida do MVP e documentar.
3. **Treinar dois modelos YOLO (Ultralytics)**: um de detecção (`components`, 32 classes) e um de pose (`flows`, 1 classe + 2 keypoints), usando só `.jpg/.png` (ignorando `.npy`).
4. **Reconstrução de grafo**: para cada imagem, cruzar geometricamente as detecções dos dois modelos (associar `tail`/`tip` de cada `flow_arrow` ao componente cuja bbox está mais próxima/contém o ponto) e montar um grafo dirigido de componentes + identificar quais `boundary_*` cada nó/aresta está dentro de ou cruza.
5. **Motor de regras STRIDE**: mapear tipo de nó + cruzamento de boundary + direção do fluxo → categorias STRIDE aplicáveis, usando a base de conhecimento da seção 8 para puxar vulnerabilidades/contramedidas específicas.
6. **Geração do relatório final** por diagrama, listando componentes identificados, fluxos, ameaças STRIDE aplicáveis e contramedidas recomendadas.

## 10. Como reproduzir esta EDA

```bash
# pré-requisito: git-lfs instalado e dados materializados
brew install git-lfs && git lfs install
(cd dataset/stride-architecture-components-v1 && git lfs pull)
(cd dataset/stride-architecture-flows-v1 && git lfs pull)

# rodar a EDA (sem instalar dependências no projeto — ambiente efêmero via uv)
cd eda
uv run --no-project --with numpy,pillow,matplotlib python3 analyze_datasets.py
```

Saídas em `eda/outputs/`: gráficos (`*.png`), amostras anotadas (`sample_components_*.png`, `sample_flows_*.png`) e `stats.json` com todos os números citados neste relatório.
