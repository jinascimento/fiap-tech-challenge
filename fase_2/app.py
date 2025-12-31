import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import streamlit as st
from google import genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY: str = st.secrets["GOOGLE_API_KEY"]

client = genai.Client(api_key=GOOGLE_API_KEY)

try:
    available_prompts = [f.stem for f in Path("prompts").glob("*.md")]
except Exception:
    available_prompts = ["padrao"]  # Fallback


@dataclass
class Patient:
    gender: str
    age: int
    hypertension: bool
    heart_disease: bool
    smoking_history: str
    bmi: float
    hba1c: float
    glucose: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


st.set_page_config(
    page_title="Tech Challenge - Diagnóstico Diabetes - Fase 2",
    page_icon="🏥",
    layout="wide",
)


def load_prompt_template(filename: str, **kwargs) -> str:
    """Carrega um modelo e preenche as lacunas no mesmo"""
    message = "Ocorreu um erro ao carregar o modelo."

    try:
        prompt_path = Path("prompts") / f"{filename}.md"

        with open(prompt_path, "r", encoding="utf-8") as f:
            template = f.read()

        return template.format(**kwargs)
    except FileNotFoundError:
        message = f"O arquivo de prompt '{filename}.md' não foi encontrado."
    except KeyError as e:
        message = f"A variável {e} esperada no modelo não foi fornecida."

    logger.error(message)
    return message


def mock_model_predict(patient: Patient) -> tuple[bool, float]:
    """Simula a inferência do modelo."""
    interval = random.randint(0, 1)

    time.sleep(interval)

    score = 0

    if patient.glucose > 140:
        score += 40
    if patient.hba1c > 6.5:
        score += 40
    if patient.bmi > 30:
        score += 10
    if patient.age > 50:
        score += 5
    if patient.heart_disease:
        score += 5

    probability = min(score + random.randint(0, 5), 99) / 100.0
    prediction = True if probability > 0.5 else False

    return prediction, probability


def llm_generation(patient: Patient, prediction, proba) -> tuple[str, str]:
    """
    Integração real com o Gemini para interpretação de diagnósticos médicos.
    """
    template_vars = {
        "data": patient.to_json(),
        "classification": "ALTO" if prediction == 1 else "BAIXO",
        "glucose": patient.glucose,
        "hba1c": patient.hba1c,
        "bmi": patient.bmi,
        "probability": f"{proba:.2%}",
    }

    prompt = load_prompt_template(selected_prompt, **template_vars)

    to_professional: str = "LLM indisponível"
    to_patient: str = "Não foi possível gerar uma explicação detalhada"

    try:
        start_time = time.perf_counter()

        response = client.models.generate_content(
            model="gemini-3-flash-preview", contents=prompt
        )

        duration = time.perf_counter() - start_time

        if not response.parts:
            logger.warning("Resposta vazia. Possível bloqueio de segurança")

            return to_professional, to_patient

        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count
            total_tokens = response.usage_metadata.total_token_count

            # Cálculo de vazão (Tokens de saída por segundo é a métrica mais relevante para UX)
            tokens_per_sec = output_tokens / duration if duration > 0 else 0

            logger.info(
                f"Latência: {duration:.2f}s | "
                f"Tokens: {input_tokens} in / {output_tokens} out | "
                f"Velocidade: {tokens_per_sec:.1f} tokens/s | "
                f"Total Tokens: {total_tokens} (input e output)"
            )

        full_text = response.text or ""

        if "[DIVIDER]" in full_text:
            tech_part, patient_part = full_text.split("[DIVIDER]", 1)
            to_professional = tech_part.strip()
            to_patient = patient_part.strip()
        else:
            # Caso a resposta não contenha o divisor, a explicação técnica
            # ainda pode ser útil para o médico
            to_patient = full_text.strip()

            logger.warning("O LLM não utilizou o divisor corretamente")

    except Exception as e:
        logger.error(f"Falha ao integrar LLM: {str(e)}", exc_info=True)

    return to_professional, to_patient


st.title("🏥 Sistema Inteligente de Triagem - Diabetes")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configuração da IA")
    selected_prompt = st.selectbox(
        "Estilo de Análise (Prompt)",
        options=available_prompts,
        index=0 if "padrao" in available_prompts else 0,
    )

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Dados do Paciente")

    with st.form("patient_form"):
        gender = st.selectbox("Gênero", ["Feminino", "Masculino"])
        age = st.number_input("Idade", min_value=1, max_value=120, value=45)

        c1, c2 = st.columns(2)
        with c1:
            hypertension = st.checkbox("Hipertensão")
        with c2:
            heart_disease = st.checkbox("Doença Cardíaca")

        smoking_history = st.selectbox(
            "Histórico de Fumo",
            ["Nunca fumou", "Ex-fumante", "Fumante atual", "Sem informação"],
        )

        bmi = st.number_input("IMC (Índice de Massa Corporal)", 10.0, 60.0, 27.5)
        hba1c = st.number_input("HbA1c (Hemoglobina Glicada)", 3.0, 15.0, 5.8)
        glucose = st.number_input("Glicose no Sangue", 50, 300, 120)

        st.markdown("---")

        submit_btn = st.form_submit_button(
            "🔍 Processar Diagnóstico",
            use_container_width=True,
        )


if submit_btn:
    patient: Patient = Patient(
        gender=gender,
        age=int(age),
        hypertension=hypertension,
        heart_disease=heart_disease,
        smoking_history=smoking_history,
        bmi=float(bmi),
        hba1c=float(hba1c),
        glucose=int(glucose),
    )

    with col2:
        status_container = st.empty()

        # Simulação do ML
        status_container.info("⚙️ Processando dados com o modelo de classificação...")
        prediction, proba = mock_model_predict(patient)

        # Simulação da LLM
        status_container.info("🧠 Processando dados com a IA Generativa...")
        tech_text, patient_text = llm_generation(patient, prediction, proba)

        # Limpa as mensagens de carregamento
        status_container.empty()

        st.subheader("Resultados da Análise")

        # Métricas de topo
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Classificação",
            "Positivo" if prediction == 1 else "Negativo",
            delta="Atenção" if prediction == 1 else "Normal",
            delta_color="inverse",
        )
        m2.metric("Confiança do Modelo", f"{proba:.1%}")
        m3.metric("Risco Calculado", "Alto" if proba > 0.5 else "Baixo")

        st.divider()

        # Abas para separar a visão técnica da visão do paciente
        tab1, tab2 = st.tabs(
            ["👨‍⚕️ Visão Médica (Técnica)", "🗣️ Visão Paciente (Humanizada)"]
        )

        with tab1:
            st.info(
                "Esta seção contém dados estatísticos e terminologia técnica para apoio à decisão clínica."
            )
            st.markdown(tech_text)

            with st.expander("Ver dados brutos (JSON)"):
                st.json(asdict(patient))

        with tab2:
            st.success(
                "Esta seção contém um roteiro de fala sugerido para comunicação empática."
            )
            st.markdown(f"> {patient_text}")

            st.button("📋 Copiar Texto para Prontuário", help="Simulação de cópia")

else:
    # Estado inicial
    with col2:
        st.info(
            "👈 Preencha os dados clínicos ao lado e clique em 'Processar Diagnóstico' para ver a simulação."
        )

        st.markdown("""
        ### O que este sistema fará (Simulação):
        1.  **Processamento de ML:** Receberá os dados estruturados e passará pelo modelo otimizado (XGBoost/RandomForest).
        2.  **Geração de Texto:** O resultado numérico será enviado para uma LLM (Gemini).
        3.  **Saída Dupla:**
            * Relatório técnico para o profissional de saúde.
            * Explicação acessível para o paciente.
        """)
