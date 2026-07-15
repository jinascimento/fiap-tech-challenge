"""Wrapper para o modelo STRIDE pre-treinado.

Modo atual: MOCK. Ignora a imagem enviada e devolve o relatorio de exemplo
armazenado em ``samples/relatorio_exemplo.txt``.

Quando o modelo real estiver disponivel, basta substituir o corpo de
``analisar_diagrama`` por uma chamada ``subprocess.run(...)`` (ou uma
importacao Python direta). A assinatura publica deve permanecer estavel.
"""

from __future__ import annotations

from pathlib import Path

_SAMPLE_PATH = Path(__file__).resolve().parent.parent / "samples" / "relatorio_exemplo.txt"


class StrideRunnerError(RuntimeError):
    """Erro elevado quando o modelo STRIDE falha ao analisar o diagrama."""


def analisar_diagrama(image_path: Path) -> str:
    """Analisa um diagrama de arquitetura e retorna o relatorio STRIDE bruto.

    Args:
        image_path: Caminho para a imagem do diagrama (PNG/JPG).

    Returns:
        Texto do relatorio STRIDE, no formato emitido pelo modelo original.

    Raises:
        FileNotFoundError: Se a imagem nao existir.
        StrideRunnerError: Se o modelo real falhar (nao aplicavel no mock).
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Imagem nao encontrada: {image_path}")

    return _run_mock()


def _run_mock() -> str:
    """Le o relatorio de exemplo do disco.

    Substituir por chamada real quando o modelo estiver integrado. Exemplo:

        result = subprocess.run(
            ["stride-analyze", "--image", str(image_path), "--format", "text"],
            capture_output=True,
            text=True,
            timeout=180,
            check=True,
        )
        return result.stdout
    """
    if not _SAMPLE_PATH.exists():
        raise StrideRunnerError(
            f"Arquivo de exemplo do mock nao encontrado em {_SAMPLE_PATH}. "
            "Verifique se samples/relatorio_exemplo.txt existe."
        )
    return _SAMPLE_PATH.read_text(encoding="utf-8")
