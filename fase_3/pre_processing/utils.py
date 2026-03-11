import re


def anonymize_text(text: str) -> str:
    """Remove PII para garantir a segurança dos dados."""
    if not text:
        return ""

    # Mascara nomes de médicos e pacientes
    text = re.sub(
        r"(Dr\.|Dra\.|Paciente:?|Médico:?)\s+[A-Z][a-z]+", r"\1 [ANONIMIZADO]", text
    )

    # Mascara identificadores numéricos fictícios
    text = re.sub(r"\b\d{5,}\b", "[ID ANONIMIZADO]", text)

    return text


def clean_and_format(text: str) -> str:
    """Limpeza de strings e normalização."""
    text = re.sub(r"\s+", " ", text)
    return anonymize_text(text.strip())
