import json
import re
import datetime
from google.cloud import storage, firestore
from google import genai
from google.genai import types

from config.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)

# Inicialização dos clientes
storage_client = storage.Client(project=settings.PROJECT_ID)
db = firestore.Client(project=settings.PROJECT_ID)
client = genai.Client(
    vertexai=True,
    project=settings.PROJECT_ID,
    location=settings.GCP_LOCATION,
)

def upload_to_gcs(file_bytes, file_name):
    """Realiza o upload do arquivo para o GCS e retorna a URI gs://"""
    gcs_uri = f"gs://{settings.BUCKET_NAME}/audios/{file_name}"
    logger.info("Iniciando upload para GCS: %s", gcs_uri)

    try:
        bucket = storage_client.bucket(settings.BUCKET_NAME)
        blob = bucket.blob(f"audios/{file_name}")
        blob.upload_from_string(file_bytes)
        logger.info("Upload para GCS concluido: %s", gcs_uri)
        return gcs_uri
    except Exception:
        logger.exception("Falha no upload para GCS: %s", gcs_uri)
        raise

def analyze_audio_with_gemini(gcs_uri):
    """Envia o áudio para o Gemini 2.5 Flash-Lite para análise especializada"""
    logger.info("Iniciando analise de audio com Gemini: %s", gcs_uri)

    prompt = """
    Você é um assistente médico especializado em saúde mental materna e ginecológica.
    Analise este áudio de consulta pós-parto e identifique:
    1. Sinais de depressão pós-parto ou ansiedade extrema (tom de voz, velocidade da fala, conteúdo).
    2. Nível de risco (Baixo, Médio, Alto).
    3. Resumo estruturado dos sintomas relatados.
    4. Sugestão de encaminhamento.

    Responda obrigatoriamente em formato JSON com as chaves: 
    'status' (ex: Atenção, Normal), 
    'risco' (ex: Médio, Baixo), 
    'analise_detalhada', 
    'sintomas_detectados' (lista),
    'recomendacao'.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[
                types.Part.from_uri(file_uri=gcs_uri, mime_type="audio/mpeg"),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )

        # Tenta usar o parsed automático do SDK
        if hasattr(response, 'parsed') and response.parsed:
            logger.info("Analise de audio concluida via resposta parseada do SDK")
            return response.parsed

        # Fallback manual: extrai o texto do primeiro candidato
        if response.candidates and response.candidates[0].content.parts:
            raw_text = response.candidates[0].content.parts[0].text
            # Limpa possíveis blocos de código markdown (```json ... ```)
            clean_text = re.sub(r"```json\s?|\s?```", "", raw_text).strip()
            parsed_result = json.loads(clean_text)
            logger.info("Analise de audio concluida via fallback de parse manual")
            return parsed_result

        logger.warning("Resposta vazia ou malformada do modelo. Raw: %s", response)
        
        return {
            "status": "Erro",
            "risco": "N/A",
            "analise_detalhada": "O modelo não gerou uma resposta válida.",
            "sintomas_detectados": [],
            "recomendacao": "Tente novamente."
        }
        
    except Exception:
        logger.exception("Excecao na chamada ao Gemini")
        raise


def save_result_to_firestore(paciente_name, analysis_result):
    """Salva o resultado da análise no Firestore"""
    logger.info("Salvando resultado no Firestore para paciente: %s", paciente_name)

    try:
        doc_ref = db.collection("consultas_audio").document()

        data = {
            "paciente": paciente_name,
            "data": datetime.datetime.now().isoformat(),
            "status": analysis_result.get("status", "Analisado"),
            "risco": analysis_result.get("risco", "Baixo"),
            "detalhes": analysis_result,
            "tipo": "Audio - Pós-Parto"
        }

        doc_ref.set(data)
        logger.info("Resultado salvo no Firestore com sucesso para paciente: %s", paciente_name)
        return data
    except Exception:
        logger.exception("Falha ao salvar resultado no Firestore para paciente: %s", paciente_name)
        raise
