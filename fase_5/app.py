"""Streamlit UI para o MVP de modelagem de ameacas STRIDE.

Fluxo:
    upload -> stride_runner (YOLO + STRIDE) -> stride_parser -> friendly_agent -> render
"""

from __future__ import annotations

import hashlib
import io
import tempfile
from collections import Counter
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core.friendly_agent import humanizar  # noqa: E402
from core.schemas import (  # noqa: E402
    AmeacaAmigavel,
    RelatorioAmigavel,
    RelatorioTecnico,
    Severidade,
)
from core.stride_parser import parse  # noqa: E402
from core.stride_runner import analisar_diagrama  # noqa: E402

st.set_page_config(
    page_title="STRIDE - Modelagem de Ameacas",
    page_icon=":shield:",
    layout="wide",
)


_SEVERIDADE_EMOJI: dict[Severidade, str] = {
    "critica": ":red_circle:",
    "alta": ":large_orange_diamond:",
    "media": ":large_yellow_circle:",
    "baixa": ":large_blue_circle:",
}

_SEVERIDADE_ORDEM: dict[Severidade, int] = {
    "critica": 0,
    "alta": 1,
    "media": 2,
    "baixa": 3,
}


@st.cache_data(show_spinner=False)
def _analisar_cached(image_bytes: bytes, _content_hash: str, image_name: str) -> str:
    """Roda o modelo STRIDE e cacheia por hash do conteudo da imagem."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = Path(tmp.name)
    try:
        return analisar_diagrama(tmp_path, image_name)
    finally:
        tmp_path.unlink(missing_ok=True)


@st.cache_data(show_spinner=False)
def _parse_cached(raw: str) -> RelatorioTecnico:
    return parse(raw)


@st.cache_data(show_spinner=False)
def _humanizar_cached(_rel_json: str) -> RelatorioAmigavel:
    """Cacheado por serializacao do relatorio tecnico para evitar re-chamadas."""
    rel = RelatorioTecnico.model_validate_json(_rel_json)
    return humanizar(rel)


def _render_ameaca(ameaca: AmeacaAmigavel) -> None:
    emoji = _SEVERIDADE_EMOJI.get(ameaca.severidade, ":white_circle:")
    with st.container(border=True):
        cabecalho, tag = st.columns([4, 1])
        with cabecalho:
            st.markdown(f"**{emoji} `{ameaca.componente}`**  \n_{ameaca.categoria}_")
        with tag:
            st.markdown(
                f"<div style='text-align:right;font-weight:600;text-transform:uppercase;'>"
                f"{ameaca.severidade}</div>",
                unsafe_allow_html=True,
            )
        st.markdown(f"**O que e:** {ameaca.explicacao_leigo}")
        st.markdown(f"**O que pode acontecer:** {ameaca.o_que_pode_acontecer}")
        st.markdown(f"**Como mitigar:** {ameaca.como_mitigar_em_portugues}")


def _relatorio_como_markdown(
    tec: RelatorioTecnico, amig: RelatorioAmigavel, raw: str
) -> str:
    linhas: list[str] = []
    linhas.append(f"# Relatorio de Ameacas - {tec.arquitetura}\n")
    linhas.append("## Resumo executivo\n")
    linhas.append(f"{amig.resumo_executivo}\n")
    linhas.append(
        f"- Componentes identificados: **{amig.total_componentes}**\n"
        f"- Ameacas identificadas: **{amig.total_ameacas}**\n"
    )

    linhas.append("\n## Principais riscos\n")
    for a in amig.principais_riscos:
        linhas.append(
            f"### {a.componente} - {a.categoria} ({a.severidade})\n"
            f"- **O que e:** {a.explicacao_leigo}\n"
            f"- **O que pode acontecer:** {a.o_que_pode_acontecer}\n"
            f"- **Como mitigar:** {a.como_mitigar_em_portugues}\n"
        )

    linhas.append("\n## Proximos passos\n")
    for i, passo in enumerate(amig.proximos_passos, start=1):
        linhas.append(f"{i}. {passo}\n")

    linhas.append("\n---\n\n## Relatorio tecnico completo (bruto)\n\n")
    linhas.append("```\n")
    linhas.append(raw)
    linhas.append("\n```\n")
    return "".join(linhas)


def _sidebar() -> None:
    with st.sidebar:
        st.header("Sobre")
        st.markdown(
            "MVP para analise automatica de arquiteturas usando a metodologia "
            "**STRIDE** (FIAP Tech Challenge - Fase 5)."
        )
        st.markdown(
            "1. Faca upload de um diagrama de arquitetura (PNG/JPG).\n"
            "2. O modelo STRIDE identifica componentes e ameacas.\n"
            "3. Um agente Gemini gera uma versao amigavel do relatorio."
        )


def _render_metricas(tec: RelatorioTecnico) -> None:
    contagem = Counter(a.categoria for a in tec.ameacas)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Componentes", tec.total_componentes)
    m2.metric("Ameacas totais", tec.total_ameacas)
    m3.metric("Categorias afetadas", len(contagem))
    top_cat = contagem.most_common(1)[0] if contagem else ("-", 0)
    m4.metric("Categoria mais frequente", top_cat[0], f"{top_cat[1]} ameacas")


def _render_amigavel(
    tec: RelatorioTecnico, amig: RelatorioAmigavel, raw: str
) -> None:
    st.subheader("Resumo executivo")
    st.info(amig.resumo_executivo)

    st.subheader("Principais riscos")
    riscos = sorted(amig.principais_riscos, key=lambda a: _SEVERIDADE_ORDEM[a.severidade])
    for ameaca in riscos:
        _render_ameaca(ameaca)

    st.subheader("Proximos passos")
    for i, passo in enumerate(amig.proximos_passos, start=1):
        st.markdown(f"**{i}.** {passo}")

    st.divider()
    with st.expander("Ver relatorio tecnico completo (saida bruta do modelo STRIDE)"):
        st.code(raw, language="text")

    md = _relatorio_como_markdown(tec, amig, raw)
    st.download_button(
        label="Baixar relatorio completo (Markdown)",
        data=md.encode("utf-8"),
        file_name=f"relatorio_stride_{tec.arquitetura.replace(' ', '_')}.md",
        mime="text/markdown",
    )


def main() -> None:
    st.title(":shield: STRIDE - Modelagem de Ameacas com IA")
    st.caption(
        "Envie um diagrama de arquitetura e receba uma analise de ameacas "
        "traduzida para linguagem acessivel."
    )

    _sidebar()

    upload = st.file_uploader(
        "Selecione uma imagem do diagrama de arquitetura",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
    )

    if upload is None:
        st.info("Aguardando upload de uma imagem para iniciar a analise.")
        return

    image_bytes = upload.getvalue()
    content_hash = hashlib.sha256(image_bytes).hexdigest()

    col_img, col_relatorio = st.columns([1, 2], gap="large")

    with col_img:
        st.subheader("Diagrama enviado")
        st.image(io.BytesIO(image_bytes), use_container_width=True)
        st.caption(f"Arquivo: `{upload.name}` ({len(image_bytes) / 1024:.1f} KB)")
        analisar = st.button("Analisar arquitetura", type="primary", use_container_width=True)

    if not analisar:
        with col_relatorio:
            st.info("Clique em **Analisar arquitetura** para gerar o relatorio.")
        return

    with col_relatorio:
        try:
            with st.spinner("Analisando arquitetura com o modelo STRIDE..."):
                image_name = Path(upload.name).stem
                raw = _analisar_cached(image_bytes, content_hash, image_name)
                tec = _parse_cached(raw)
        except Exception as exc:
            st.error(f"Falha na analise STRIDE: {exc}")
            return

        _render_metricas(tec)
        st.divider()

        try:
            with st.spinner("Traduzindo relatorio para linguagem amigavel (Gemini)..."):
                amig = _humanizar_cached(tec.model_dump_json())
        except RuntimeError as exc:
            st.error(str(exc))
            st.info(
                "Enquanto isso, voce ainda pode consultar o relatorio tecnico "
                "abaixo."
            )
            with st.expander("Ver relatorio tecnico completo (saida bruta)", expanded=True):
                st.code(raw, language="text")
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"Falha ao humanizar o relatorio: {exc}")
            with st.expander("Ver relatorio tecnico completo (saida bruta)", expanded=True):
                st.code(raw, language="text")
            return

        _render_amigavel(tec, amig, raw)


if __name__ == "__main__":
    main()
