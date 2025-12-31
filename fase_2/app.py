import json
import logging
import random
import time

import streamlit as st
from google import genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY: str = st.secrets["GOOGLE_API_KEY"]

client = genai.Client(api_key=GOOGLE_API_KEY)

st.set_page_config(
    page_title="Tech Challenge - Diagnóstico Diabetes - Fase 2",
    page_icon="🏥",
    layout="wide",
)


def mock_model_predict(data) -> tuple[bool, float]:
    """Simula a inferência do modelo."""
    interval = random.randint(1, 5)

    time.sleep(interval)

    score = 0

    if data["glucose"] > 140:
        score += 40
    if data["hba1c"] > 6.5:
        score += 40
    if data["bmi"] > 30:
        score += 10
    if data["age"] > 50:
        score += 5
    if data["heart_disease"]:
        score += 5

    probability = min(score + random.randint(0, 5), 99) / 100.0
    prediction = 1 if probability > 0.5 else 0

    return prediction, probability


def llm_generation(data, prediction, proba) -> tuple[str, str]:
    """
    Integração real com o Gemini para interpretação de diagnósticos médicos.
    """
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

    tech: str = "LLM indisponível"
    patient: str = "Não foi possível gerar uma explicação detalhada"

    try:
        start_time = time.perf_counter()

        response = client.models.generate_content(
            model="gemini-3-flash-preview", contents=prompt
        )

        duration = time.perf_counter() - start_time

        if not response.parts:
            logger.warning("Resposta vazia. Possível bloqueio de segurança")

            return tech, patient

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
            tech = tech_part.strip()
            patient = patient_part.strip()
        else:
            # Caso a resposta não contenha o divisor, a explicação técnica
            # ainda pode ser útil para o médico
            tech = full_text.strip()

            logger.warning("O LLM não utilizou o divisor corretamente")

    except Exception as e:
        logger.error(f"Falha ao integrar LLM: {str(e)}", exc_info=True)

    return tech, patient


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
    patient_data = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking": smoking_history,
        "bmi": bmi,
        "hba1c": hba1c,
        "glucose": glucose,
    }

    with col2:
        status_container = st.empty()

        # Simulação do ML
        status_container.info("⚙️ Processando dados com o modelo de classificação...")
        prediction, proba = mock_model_predict(patient_data)

        # Simulação da LLM
        status_container.info("🧠 Processando dados com a IA Generativa...")
        tech_text, patient_text = llm_generation(patient_data, prediction, proba)

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
                st.json(patient_data)

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
