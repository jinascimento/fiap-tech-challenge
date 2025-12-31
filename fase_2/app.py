import json
import logging
import random
import time
from dataclasses import asdict, dataclass

import streamlit as st
from google import genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY: str = st.secrets["GOOGLE_API_KEY"]

client = genai.Client(api_key=GOOGLE_API_KEY)


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
    data = asdict(patient)

    prompt = f"""
    Atue como um Especialista em Endocrinologia e Assistente de IA de alta precisão.
    Sua tarefa é interpretar os resultados de um modelo de Machine Learning otimizado por Algoritmos Genéticos para triagem de Diabetes.

    DADOS CLÍNICOS DO PACIENTE:
    {json.dumps(data, indent=2)}

    RESULTADO DA PREDIÇÃO:
    - Classificação: {"DIABETES DETECTADO" if prediction == 1 else "NEGATIVO PARA DIABETES"}
    - Confiança do Modelo: {proba:.2%}

    SUA RESPOSTA DEVE CONTER DUAS PARTES RIGIDAMENTE SEPARADAS PELO MARCADOR [DIVIDER]:

    PARTE 1: RELATÓRIO TÉCNICO (PARA O MÉDICO)
    - Analise os biomarcadores (Glicose: {data["glucose"]} mg/dL, HbA1c: {data["hba1c"]}%).
    - Correlacione com IMC ({data["bmi"]}) e histórico de saúde.
    - Forneça insights acionáveis e sugira exames complementares se necessário.
    - Use terminologia médica adequada.

    PARTE 2: COMUNICAÇÃO HUMANIZADA (PARA O PACIENTE)
    - Explique o resultado de forma simples, empática e sem jargões técnicos.
    - Foque em acolhimento e nos próximos passos práticos.
    - Mantenha um tom profissional porém encorajador.

    IMPORTANTE: Use o marcador [DIVIDER] exatamente entre as duas partes.
    """

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

        full_text = response.text

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
