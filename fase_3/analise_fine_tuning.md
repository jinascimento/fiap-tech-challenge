# Analise do Fine-Tuning na Fase 3

## Visao geral

O projeto da `fase_3` implementa um processo de fine-tuning supervisionado para um modelo de linguagem com foco em assistencia medica. A estrategia adotada combina:

- modelo base `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- ajuste eficiente por parametros com `LoRA`
- quantizacao `4-bit` via `QLoRA` quando ha GPU CUDA disponivel
- salvamento apenas dos adaptadores finais, reduzindo custo de armazenamento

Na pratica, o projeto nao treina todos os pesos do modelo base. Ele adiciona camadas LoRA em pontos especificos da arquitetura e atualiza somente esses pesos adaptadores.

## Como os dados sao preparados

O pipeline de pre-processamento gera um arquivo `JSONL` em `data/dataset_medico_treinamento.jsonl`, com cada registro no formato:

```json
{"instruction": "...", "output": "..."}
```

As fontes de dados utilizadas sao:

- `PubMedQA`, usando pergunta, contexto e resposta longa
- `MedQuAD`, usando pares de pergunta e resposta clinica
- `hospital_protocols.json`, com exemplos sinteticos baseados em protocolos hospitalares

Antes da gravacao do dataset, o projeto aplica uma limpeza simples:

- normalizacao de espacos
- mascaramento de possiveis nomes de medicos e pacientes
- mascaramento de identificadores numericos longos

Essa etapa melhora a padronizacao do treino e reduz risco de exposicao de informacao sensivel.

## Estrutura do prompt de treinamento

Cada exemplo e convertido para o seguinte template:

```text
### Instruction:
{instruction}

### Response:
{output}
```

Esse formato e importante porque o modelo aprende a associar uma instrucao medica a uma resposta esperada dentro de um padrao consistente de chat.

## Como o loss e calculado

Um ponto positivo da implementacao e que o loss nao e calculado sobre toda a sequencia. O codigo mascara a parte da instrucao e deixa ativa apenas a parte da resposta.

Isso significa que:

- a instrucao entra como contexto
- a resposta e o alvo real de aprendizado
- o treinamento fica mais alinhado com o objetivo de gerar respostas medicas, e nao apenas reconstruir o prompt inteiro

Esse detalhe melhora a qualidade do ajuste supervisionado para tarefas de pergunta e resposta.

## Configuracao tecnica do fine-tuning

### Modelo base

- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

### Configuracao LoRA

- rank `r = 16`
- `lora_alpha = 16`
- `lora_dropout = 0.1`
- `target_modules = ["q_proj", "v_proj"]`
- `bias = "none"`

Essa configuracao mostra uma escolha conservadora e coerente com fine-tuning eficiente: adapta apenas projecoes de atencao importantes, mantendo baixo consumo de memoria.

### Configuracao QLoRA

Quando o treino roda em CUDA, o modelo e carregado com:

- quantizacao `4-bit`
- tipo `nf4`
- `double quantization`
- `compute dtype = float16`

Se o ambiente estiver em `MPS` ou `CPU`, o projeto desativa a quantizacao 4-bit e faz apenas LoRA em precisao reduzida. Isso torna o script mais portavel, embora o treinamento fora de CUDA seja menos eficiente.

### Hiperparametros padrao

- `max_length = 512`
- `batch_size = 2`
- `gradient_accumulation_steps = 16`
- `num_train_epochs = 3`
- `learning_rate = 2e-5`
- scheduler `cosine`
- `save_strategy = "epoch"`

Na pratica, a combinacao de batch pequeno com acumulacao de gradiente permite treinar mesmo em hardware limitado.

## Saidas geradas

O treinamento salva:

- checkpoints no diretorio de saida durante as epocas
- adaptadores finais em `output_llm/adapter_final`

O artefato mais importante do projeto e o adaptador LoRA final, composto por arquivos como:

- `adapter_config.json`
- `adapter_model.safetensors`

Isso confirma que o resultado versionado do treino e um adaptador PEFT, e nao uma copia completa do modelo base.

## Pontos fortes da abordagem

- reduz custo computacional com LoRA e QLoRA
- aproveita um modelo base pequeno e viavel para experimentacao
- usa dados medicos externos e dados sinteticos do dominio do projeto
- treina somente sobre a resposta, o que melhora o alinhamento da tarefa
- salva um artefato final leve, simples de reutilizar em inferencia

## Limitacoes observadas

- nao ha divisao explicita entre treino e validacao
- nao ha metricas quantitativas salvas no pipeline
- o dataset final nao fica versionado no repositorio, entao a reprodutibilidade depende da geracao local
- ha mistura de exemplos em ingles e portugues, o que pode afetar consistencia linguistica
- o modelo base e pequeno para um dominio sensivel como o medico, o que exige cautela no uso

## Conclusao

O fine-tuning da `fase_3` foi bem estruturado para um cenario academico e de recursos limitados. A implementacao mostra boas decisoes tecnicas, especialmente no uso de `QLoRA + LoRA`, no mascaramento do loss para aprender apenas a resposta e na composicao de um dataset medico hibrido.

Ao mesmo tempo, o projeto ainda depende de uma etapa mais robusta de avaliacao para demonstrar, com evidencias, o ganho real de qualidade, seguranca e aderencia clinica obtido pelo ajuste fino.
