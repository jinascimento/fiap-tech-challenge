# Documentação do Projeto — Modelagem de Ameaças STRIDE via IA

Hackathon FIAP — Fase 5. Visão geral acessível do que foi construído. Para detalhes tecnicos (comandos,
estrutura de codigo, metricas completas), ver `README.md`. Para a analise
exploratoria dos dados, ver `eda/RELATORIO_EDA.md`.

---

## 1. Qual é o problema que estamos resolvendo

A FIAP Software Security quer um sistema que **olhe para uma imagem de
diagrama de arquitetura de software** (aqueles desenhos com caixinhas de
"banco de dados", "API", "usuário", setas conectando tudo) e
**automaticamente** aponte quais ameaças de segurança aquela arquitetura tem,
usando uma metodologia chamada **STRIDE**.

**STRIDE** é um checklist de 6 tipos de ameaça (a sigla vem das iniciais em
inglês):

| Letra | Nome | O que significa |
|---|---|---|
| S | Spoofing | Alguém se passando por outra pessoa/sistema (roubo de identidade) |
| T | Tampering | Alguém alterando dados que não deveria |
| R | Repudiation | Alguém nega ter feito uma ação, e não há prova disso |
| I | Information Disclosure | Dado sensível vazando para quem não deveria ver |
| D | Denial of Service | O sistema fica indisponível (sobrecarga) |
| E | Elevation of Privilege | Alguém ganha mais permissão do que deveria ter |

Normalmente um especialista em segurança olha o diagrama manualmente e pensa
"esse banco de dados aqui é vulnerável a Tampering, essa API é vulnerável a
Spoofing", etc. O desafio é fazer uma IA fazer essa primeira análise sozinha.

---

## 2. Os dados usados

Foram usados dois "datasets" (conjuntos de dados de treino) prontos,
publicados por outra pessoa (Guilherme Santos, Hugging Face), com **milhares
de imagens de diagramas reais de AWS/Azure já marcadas**:

- **`components`**: 4190 imagens onde cada ícone do diagrama (usuário, banco
  de dados, API gateway, etc — 32 tipos diferentes) já vem com um retângulo
  desenhado em volta e uma etiqueta dizendo o que é aquilo.
- **`flows`**: as mesmas imagens, mas marcando as **setas** que conectam os
  ícones, com dois pontos em cada seta indicando "de onde sai" e "pra onde
  vai" (a direção do fluxo de dados).

É um "gabarito": mostramos à IA milhares de exemplos de "esse desenho aqui é
um banco de dados", "essa seta vai do usuário pro servidor", e ela aprende a
reconhecer isso sozinha em desenhos novos, nunca vistos antes.

Uma EDA (análise exploratória) detalhada desses dados foi feita antes de
treinar qualquer coisa — ver `eda/RELATORIO_EDA.md` para os números completos
(quantidade de imagens, classes desbalanceadas, vazamento entre splits etc.).

---

## 3. A solução: um pipeline em 5 etapas

```
IMAGEM DO DIAGRAMA
      |
      v
[1] Detector de componentes  ->  "aqui tem um banco de dados, ali um usuario, ali um WAF..."
      |
      v
[2] Detector de setas         ->  "essa seta vai do usuario pro WAF, essa outra do WAF pro banco..."
      |
      v
[3] Montagem do grafo          ->  junta tudo num "mapa" de quem fala com quem, e dentro
                                    de qual "zona de confianca" (ex: dentro da VPC, na rede publica)
      |
      v
[4] Motor de regras STRIDE      ->  para cada tipo de componente e cada seta que cruza uma
                                    zona de confianca, aponta quais das 6 letras do STRIDE
                                    se aplicam, com uma sugestao de correcao
      |
      v
[5] Relatorio final              ->  um documento explicando tudo isso para a pessoa ler
```

### Etapas 1 e 2 — os "detectores" (Machine Learning de verdade)

Para ensinar o computador a reconhecer os ícones, foi usado o **YOLO** (You
Only Look Once), o algoritmo mais usado hoje para "detecção de objetos em
imagem": você mostra uma imagem e ele devolve uma lista de
"retângulo + etiqueta" (tipo "achei um cachorro aqui, um carro ali"). Aqui,
em vez de cachorro e carro, os "objetos" são ícones de arquitetura (banco de
dados, API gateway, etc).

Foram treinados **dois modelos separados**:

- Um para reconhecer os 32 tipos de componente (etapa 1).
- Um para reconhecer as setas e sua direção (etapa 2) — usando uma variante
  chamada "pose", a mesma tecnologia usada para detectar articulações do
  corpo humano em vídeo; aqui os "dois pontos articulados" são a origem e o
  destino da seta.

"Treinar" significa: o computador olha milhares de exemplos do gabarito,
erra bastante no início, e ajusta seus parâmetros internos até acertar cada
vez mais. Isso é medido em **épocas** (uma "época" = o modelo já viu todas as
imagens de treino uma vez).

### Etapa 3 — o grafo

Depois que os dois modelos rodam numa imagem nova, temos uma lista solta de
"componentes encontrados" e "setas encontradas". Esta etapa junta tudo:
pega cada seta, olha onde ela começa e termina, e decide "essa seta liga o
componente A ao componente B". Também confere se A e B estão dentro das
mesmas "caixas de fronteira" do diagrama (ex.: dentro da mesma rede privada,
ou se uma está na rede pública e a outra na privada) — isso importa muito
para o STRIDE, porque cruzar uma fronteira de rede é sempre um ponto de
risco maior.

### Etapa 4 — a base de conhecimento STRIDE

Aqui não há IA — é uma tabela escrita manualmente (arquivo `.yaml`) dizendo,
por exemplo: *"se o componente é do tipo `data_database`, ele é vulnerável a
Tampering (alguém alterar dados) e a Information Disclosure (vazamento), e a
contramedida é usar queries parametrizadas e criptografia"*. Isso foi feito
para os 32 tipos de componente, com base em boas práticas conhecidas de
segurança em nuvem. **Esse pedaço não vinha em nenhum dos datasets** — era
uma lacuna que precisou ser preenchida, porque os dados só ensinam a "ver" o
diagrama, não a "saber" quais ameaças cada peça carrega.

### Etapa 5 — o relatório

Um arquivo de texto (Markdown) juntando tudo: lista de componentes achados,
lista de conexões, e a lista de ameaças STRIDE aplicáveis com a sugestão de
correção, organizada por categoria (Spoofing, Tampering, etc).

---

## 4. Resultados

Depois de treinar, o sistema completo foi testado nas **duas imagens de
exemplo que o próprio PDF do desafio forneceu** (uma arquitetura AWS, uma
Azure) — imagens que o modelo **nunca tinha visto antes**. Foi gerada uma
versão "anotada" dessas imagens (com os retângulos desenhados por cima) para
conferência visual, e o resultado foi bom: o modelo reconheceu corretamente
a maioria dos componentes reais (Shield, CloudFront, WAF, bancos RDS, etc. na
AWS; API Gateway, Entra ID, Logic Apps na Azure), com confiança alta (a
maioria acima de 90%).

Em números técnicos (métricas de referência, não é preciso decorar):

| Modelo | mAP50 | mAP50-95 |
|---|---|---|
| Componentes (32 classes) | 0.822 | 0.644 |
| Fluxos (setas, pose) | 0.547 (caixa) / 0.637 (pose) | 0.236 (caixa) / 0.519 (pose) |

O modelo de fluxos rendeu menos porque teve menos exemplos de treino
disponíveis (176 diagramas com anotação de fluxo, contra 414 com anotação de
componente).

---


## 6. Onde encontrar cada coisa no projeto

```
fase_5/
  eda/RELATORIO_EDA.md                # analise exploratoria dos dados
  README.md                           # documentacao tecnica (comandos, setup)
  DOCUMENTACAO_PROJETO.md             # este documento
  dataset/                            # os dois datasets baixados do Hugging Face
  stride_vision/
    schema.py                         # estruturas de dados (Component, Flow, BBox)
    training/                         # scripts de treino dos dois modelos YOLO
    graph/reconstruct.py              # etapa 3: montagem do grafo
    knowledge_base/stride_rules.yaml  # etapa 4: base de conhecimento STRIDE
    knowledge_base/engine.py          # etapa 4: motor de regras
    report/generate.py                # etapa 5: relatorio em markdown
    report/visualize.py               # desenha as deteccoes sobre a imagem
    infer.py                          # roda tudo numa imagem real com os modelos treinados
    pipeline.py                       # roda tudo a partir de labels ground-truth (sem modelo)
    data/resplit.py                   # corrige vazamento entre splits de treino/val/teste
  runs/                                # pesos treinados e logs (nao versionado no git)
  reports/                             # relatorios e imagens anotadas gerados
```
