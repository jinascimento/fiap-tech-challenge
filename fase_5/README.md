# FIAP Tech Challenge - Fase 5

MVP em Streamlit para modelagem de ameacas com metodologia **STRIDE** a partir de diagramas de arquitetura de software.

Este projeto e a camada de **interface e humanizacao** do MVP. A analise STRIDE em si e feita por um modelo pre-treinado separado (acessado via CLI). Aqui:

1. O usuario faz upload de uma imagem com o diagrama de arquitetura.
2. O modelo STRIDE analisa a imagem e retorna um relatorio tecnico bruto.
3. O relatorio bruto e parseado para uma estrutura tipada.
4. Um agente [Pydantic AI](https://ai.pydantic.dev/) com **Gemini 2.5 Flash** traduz o conteudo tecnico em uma versao amigavel para nao-tecnicos.
5. O Streamlit renderiza a versao amigavel em destaque, com o relatorio tecnico completo preservado num expander.

## Requisitos

- Python 3.13
- [uv](https://docs.astral.sh/uv/) para gerenciamento de dependencias
- Uma API key do Google AI Studio ([gere aqui](https://aistudio.google.com/apikey))

## Setup

```bash
uv sync
cp .env.example .env
```

Edite `.env` e cole sua chave em `GEMINI_API_KEY`.

## Como rodar

```bash
uv run streamlit run app.py
```

A UI abrira em [http://localhost:8501](http://localhost:8501).

Enquanto o modelo STRIDE real nao esta integrado, o app roda em **modo MOCK**: qualquer imagem enviada retorna o relatorio de exemplo em `samples/relatorio_exemplo.txt`. Isso permite desenvolver a UI e o agente de humanizacao em paralelo.

## Estrutura

```
fase_5/
├── app.py                    # Entrypoint Streamlit
├── core/
│   ├── schemas.py            # Modelos Pydantic (tecnico + amigavel)
│   ├── stride_runner.py      # Wrapper do modelo STRIDE (MOCK hoje)
│   ├── stride_parser.py      # Parser: texto STRIDE -> RelatorioTecnico
│   └── friendly_agent.py     # Agente Pydantic AI + Gemini
├── samples/
│   └── relatorio_exemplo.txt # Saida STRIDE de exemplo (mock)
├── pyproject.toml
└── .env.example
```

## Como trocar o mock pelo modelo real

Basta editar a funcao `analisar_diagrama` em [core/stride_runner.py](core/stride_runner.py) para chamar o binario/script real via `subprocess.run(...)`. A interface publica (`analisar_diagrama(image_path: Path) -> str`) nao muda, entao nada mais precisa ser tocado.

## Fluxo

```
[imagem]
  -> stride_runner.analisar_diagrama  (MOCK ou CLI real)
  -> stride_parser.parse              (texto -> RelatorioTecnico)
  -> friendly_agent.humanizar         (Pydantic AI + Gemini -> RelatorioAmigavel)
  -> Streamlit                        (renderiza amigavel + bruto)
```
