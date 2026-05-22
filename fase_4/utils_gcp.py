import json
import re
import datetime
from google.cloud import storage, firestore, videointelligence
from google import genai
from google.genai import types

from config.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)

# Inicialização dos clientes
storage_client = storage.Client(project=settings.PROJECT_ID)
db = firestore.Client(project=settings.PROJECT_ID)
video_intelligence_client = videointelligence.VideoIntelligenceServiceClient()
client = genai.Client(
    vertexai=True,
    project=settings.PROJECT_ID,
    location=settings.GCP_LOCATION,
)

def upload_to_gcs(file_bytes, file_name, folder="audios"):
    """Realiza o upload do arquivo para o GCS e retorna a URI gs://"""
    gcs_uri = f"gs://{settings.BUCKET_NAME}/{folder}/{file_name}"
    logger.info("Iniciando upload para GCS: %s", gcs_uri)

    try:
        bucket = storage_client.bucket(settings.BUCKET_NAME)
        blob = bucket.blob(f"{folder}/{file_name}")
        blob.upload_from_string(file_bytes)
        logger.info("Upload para GCS concluido: %s", gcs_uri)
        return gcs_uri
    except Exception:
        logger.exception("Falha no upload para GCS: %s", gcs_uri)
        raise

def analyze_video_with_intelligence_api(gcs_uri):
    """Utiliza a Video Intelligence API para detectar labels e conteúdos no vídeo"""
    logger.info("Iniciando analise com Video Intelligence API: %s", gcs_uri)
    
    features = [
        videointelligence.Feature.LABEL_DETECTION,
        videointelligence.Feature.EXPLICIT_CONTENT_DETECTION,
    ]

    try:
        operation = video_intelligence_client.annotate_video(
            request={
                "features": features,
                "input_uri": gcs_uri,
            }
        )
        
        logger.info("Aguardando finalização da Video Intelligence API...")
        result = operation.result(timeout=180)
        
        # Extrair labels principais
        labels = []
        if result.annotation_results:
            labels = [
                annotation.entity.description 
                for annotation in result.annotation_results[0].segment_label_annotations
            ][:10]
        
        logger.info("Video Intelligence API concluída. Labels detectados: %s", labels)
        return {
            "labels": labels,
            "status_video_ai": "Sucesso"
        }
    except Exception:
        logger.exception("Falha na Video Intelligence API")
        return {"labels": [], "status_video_ai": "Erro"}

def analyze_video_with_gemini(gcs_uri):
    """Envia o vídeo para o Gemini 2.5 Flash para análise de saúde e segurança feminina"""
    logger.info("Iniciando analise de video com Gemini: %s", gcs_uri)

    prompt = """
    Você é um sistema de IA especializado em proteção e saúde da mulher. 
    Analise este vídeo cuidadosamente para identificar sinais precoces de risco.
    
    FOCO DA ANÁLISE:
    1. Saúde Mental: Sinais de depressão pós-parto, ansiedade severa ou exaustão extrema (expressão facial, apatia, choro, agitação).
    2. Segurança Física: Sinais de violência doméstica, hematomas visíveis, comportamento defensivo ou medo em relação ao ambiente.
    3. Bem-estar Geral: Condições do ambiente e interação (se houver outras pessoas).

    INSTRUÇÕES:
    - Identifique microexpressões e linguagem corporal.
    - Se houver áudio, considere também o conteúdo e tom de voz.
    - Seja empático, mas clínico e objetivo na descrição.

    Responda obrigatoriamente em formato JSON com as chaves: 
    'status' (ex: Crítico, Atenção, Normal), 
    'risco' (ex: Alto, Médio, Baixo), 
    'analise_detalhada' (texto explicativo), 
    'sinais_detectados' (lista de strings),
    'recomendacao' (próximo passo sugerido).
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_uri(file_uri=gcs_uri, mime_type="video/mp4"),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )

        if hasattr(response, 'parsed') and response.parsed:
            return response.parsed

        if response.candidates and response.candidates[0].content.parts:
            raw_text = response.candidates[0].content.parts[0].text
            clean_text = re.sub(r"```json\s?|\s?```", "", raw_text).strip()
            return json.loads(clean_text)

        return {
            "status": "Erro",
            "risco": "N/A",
            "analise_detalhada": "O modelo não gerou uma resposta válida para o vídeo.",
            "sinais_detectados": [],
            "recomendacao": "Tente novamente ou use outro formato de arquivo."
        }
        
    except Exception:
        logger.exception("Excecao na chamada ao Gemini para vídeo")
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


def save_result_to_firestore(paciente_name, analysis_result, tipo_analise="Áudio"):
    """Salva o resultado da análise no Firestore"""
    logger.info("Salvando resultado no Firestore para paciente: %s (%s)", paciente_name, tipo_analise)

    try:
        doc_ref = db.collection("consultas_analises").document()

        data = {
            "paciente": paciente_name,
            "data": datetime.datetime.now().isoformat(),
            "status": analysis_result.get("status", "Analisado"),
            "risco": analysis_result.get("risco", "Baixo"),
            "detalhes": analysis_result,
            "tipo": tipo_analise
        }

        doc_ref.set(data)
        logger.info("Resultado salvo no Firestore com sucesso para paciente: %s", paciente_name)
        return data
    except Exception:
        logger.exception("Falha ao salvar resultado no Firestore para paciente: %s", paciente_name)
        raise
