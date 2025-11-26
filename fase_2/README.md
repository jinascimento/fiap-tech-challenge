# FIAP Tech Challenge - Fase 2

Este repositório contém o projeto desenvolvido para a `Fase 2` do curso de Pós-Graduação em IA. O foco desta etapa é a evolução de modelos preditivos e a aplicação prática de conceitos de `algoritmos genéticos`.

## Escopo do Projeto

Para este trecho, optamos pelo `Desafio 1`, que visa o aperfeiçoamento dos modelos criados no `Tech Challenge - Fase 1`. A evolução consiste na aplicação de técnicas de otimização e na criação de interfaces que permitam a experimentação dos parâmetros.

## Tecnologias e Ferramentas

- Linguagem: [Python 3.12+](https://www.python.org/)
- Gerenciador de Pacotes: [uv](https://docs.astral.sh/uv/)
- Interface Web: [Streamlit](https://streamlit.io/)
- Interface Gráfica (Simulação): [Pygame](https://www.pygame.org/)

## Configuração e Instalação

Utilizamos o  `uv` para garantir a reprodutibilidade do ambiente e gerenciamento simplificado de dependências.

1. Clonar o repositório:

```bash
git clone git@github.com:jinascimento/fiap-tech-challenge.git
cd fase_2
```

2. Instalar o `uv` (caso não possua), é possível seguir as instruções na [documentação oficial](https://docs.astral.sh/uv/getting-started/installation/).

## Execução das Aplicações

O projeto está dividido em duas frentes principais:

### 1. Painel analítico

#### Modelo

Lorem Ipsum

#### LLM

Além das predições obtidas através do modelo, usamos uma LLM (Google Gemini) para gerar insights técnicos e humanizados sobre os resultados.

Criamos três _perfis_ diferentes para instruirmos a LLM a assumir diferentes perspectivas ao produzir as mensagens.

#### Interface

Uma aplicação web, baseada no `Streamlit`, permite que profissionais possam inserir dados dos pacientes e obter tanto a predição do modelo quanto os insights da LLM, que gera uma resposta técnica, focada no médico com pontências tomadas de ação e próximos passos para o tratamento, quanto uma resposta humanizada, que o profissional pode utilizar como suporte para comunicar os resultados ao paciente.

#### Executando

```bash
uv run streamlit run app.py
```

O comando acima criará o ambiente virtual automaticamente, instalará as dependências e iniciará o servidor local.

### 2. Desafio de Camuflagem (Algoritmos Genéticos)

Como conteúdo extra e prático (baseado na **Aula 5**), implementamos uma simulação visual do **desafio da camuflagem**. A simulação demonstra como uma população evolui suas cores para se misturar ao ambiente através de seleção natural, crossover e mutação.

```bash
uv run python camouflage.py
```

#### Detalhes de implementação

A solução em `camouflage.py` utiliza a biblioteca `Pygame` para renderizar a evolução em tempo real. Os principais pilares implementados foram:

- indivíduos: representados por cores (RGB) em uma matriz de genes.
- fitness (Aptidão): calculado com base na proximidade cromática entre o indivíduo e a cor de fundo do ambiente.
- seleção: método de torneio para definir os progenitores da próxima geração.
- operadores genéticos: Aplicação de taxas de mutação e crossover para garantir a variabilidade genética e convergência.

## Integrantes

- Rodrigo de Suza Braga - RM368177
- Jhonatan Izaias do Nascimento - RM366840
- Fabiano Miranda Pereira - RM367756
- Yuslley - RM
