from pathlib import Path
from google.cloud import texttospeech

from config.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


def validate_google_credentials() -> Path:
    credentials = settings.GOOGLE_CREDENTIALS_FILE.expanduser()

    if not credentials.is_file():
        error_message = (
            "Arquivo de credenciais do Google Cloud não encontrado. "
            "Por favor, crie um projeto no Google Cloud, ative a API Text-to-Speech, "
            "e baixe o arquivo JSON de credenciais (com o nome google-credentials.json) "
            "para este diretório."
        )
        logger.error("Credenciais do Google Cloud nao encontradas em: %s", credentials)
        raise FileNotFoundError(error_message)
    
    return credentials.expanduser()


def generate_audio_ssml(ssml_text, filename, speaking_rate=1.0, pitch=0.0):
    logger.info("Gerando audio: %s", filename)
    client = texttospeech.TextToSpeechClient()

    # Usando SSML em vez de texto simples para maior controle
    input_ssml = texttospeech.SynthesisInput(ssml=f"<speak>{ssml_text}</speak>")

    # Usando vozes (femininas) Neural2 que são as mais realistas disponíveis
    voice = texttospeech.VoiceSelectionParams(
        language_code=settings.TTS_LANGUAGE_CODE,
        name=settings.TTS_VOICE_NAME,
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=speaking_rate,
        pitch=pitch
    )

    response = client.synthesize_speech(
        input=input_ssml, voice=voice, audio_config=audio_config
    )

    with open(filename, "wb") as out:
        out.write(response.audio_content)
        logger.info("Audio gerado: %s", filename)


scenarios = [
    {
        "name": "pos_parto_01.mp3",
        "ssml": """
            Doutora... <break time="800ms"/> eu não sei o que está acontecendo comigo. 
            <emphasis level="reduced">Eu me sinto um lixo.</emphasis> 
            Meu bebê chora e eu só tenho vontade de sair correndo... <break time="500ms"/> 
            Não sinto alegria nenhuma, <prosody rate="slow">só um vazio enorme e um cansaço que não passa nunca.</prosody>
        """,
        "rate": 0.9,
        "pitch": -1.5
    },
    {
        "name": "violencia_01.mp3",
        "ssml": """
            Não... <break time="1s"/> está tudo bem em casa. 
            Eu só... <break time="600ms"/> eu caí da escada, foi isso. 
            O meu marido é muito bom pra mim, <break time="400ms"/> ele só... ele tem um temperamento forte as vezes... 
            <emphasis level="reduced">mas eu que sou muito desastrada.</emphasis>
        """,
        "rate": 0.95,
        "pitch": -0.5
    },
    {
        "name": "ansiedade_01.mp3",
        "ssml": """
            Doutora, eu não consigo parar de pensar... <break time="200ms"/> meu coração está sempre acelerado. 
            Eu li na internet que qualquer dorzinha pode ser perigoso para o bebê e agora eu não durmo mais. 
            <prosody rate="fast">E se eu não for uma boa mãe? E se algo acontecer no parto? Eu estou com muito medo!</prosody>
        """,
        "rate": 1.15, # Fala acelerada
        "pitch": 1.5   # Tom mais agudo/tenso
    },
    {
        "name": "saudavel_01.mp3",
        "ssml": """
            Olá doutora, tudo bem? <break time="300ms"/> Eu vim para o meu acompanhamento de rotina. 
            Estou me sentindo ótima, o bebê está chutando bastante e eu estou muito animada com os preparativos para o quarto. 
            Não tenho tido dores e estou conseguindo descansar bem. Está tudo perfeito!
        """,
        "rate": 1.0, # Fala estável
        "pitch": 0.0  # Tom equilibrado
    }
]


if __name__ == "__main__":
    credentials_path = validate_google_credentials()
    logger.info("Usando credenciais em: %s", credentials_path)

    path = settings.TEST_AUDIOS_DIR
    
    for s in scenarios:
        generate_audio_ssml(
            s["ssml"],
            str(path / s["name"]),
            s["rate"],
            s["pitch"],
        )

    logger.info("Geracao de audios concluida. Total: %d", len(scenarios))
