import streamlit as st
import time
import random


st.set_page_config(
    page_title="Tech Challenge - Diagnóstico Diabetes - Fase 2",
    page_icon="🏥",
    layout="wide"
)


def mock_model_predict(data) -> tuple[bool, float]:
    """Simula a inferência do modelo."""
    interval = random.randint(1, 5)

    time.sleep(interval)
    
    score = 0

    if data['glucose'] > 140: score += 40
    if data['hba1c'] > 6.5: score += 40
    if data['bmi'] > 30: score += 10
    if data['age'] > 50: score += 5
    if data['heart_disease']: score += 5
    
    probability = min(score + random.randint(0, 5), 99) / 100.0
    prediction = 1 if probability > 0.5 else 0
    
    return prediction, probability


def mock_llm_generation(data, prediction, proba) -> tuple[str, str]:
    """Simula a geração de texto da LLM."""
    interval = random.randint(1, 5)

    time.sleep(interval)
    
    risco_texto = "ALTO" if prediction == 1 else "BAIXO"
    
    # Resposta técnica
    technical_response = f"""
    **Análise Técnica do Modelo Preditivo (v1.0 - Algoritmo Genético Otimizado)**
    
    * **Classificação do Modelo:** Diabetes {risco_texto} RISCO (Classe {prediction})
    * **Probabilidade Estimada:** {proba:.2%}
    * **Fatores de Influência Principais:**
        * A Glicose ({data['glucose']} mg/dL) e o HbA1c ({data['hba1c']}%) foram os determinantes principais para a decisão do modelo.
        * O IMC de {data['bmi']} contribuiu moderadamente para o vetor de decisão.
    * **Sugestão de Conduta:** Correlacionar com histórico familiar e considerar solicitação de curva glicêmica se ainda não realizada.
    """

    # Resposta dirigida ao paciente
    if prediction == 1:
        patient_response = f"""
        Olá. Com base nos exames que você forneceu, os resultados indicam que **precisamos ter uma atenção especial**.
        
        Seus níveis de açúcar no sangue e outros indicadores sugerem uma compatibilidade com **Diabetes**. Não se assuste, isso significa apenas que precisamos iniciar um acompanhamento mais próximo para cuidar da sua saúde.
        
        O próximo passo é conversarmos sobre alguns ajustes na alimentação e talvez o uso de alguma medicação para manter tudo sob controle. Vamos cuidar disso juntos.
        """
    else:
        patient_response = f"""
        Olá. Tenho ótimas notícias. Com base na análise dos seus dados atuais, **não há indicativos de diabetes neste momento**.
        
        Seus níveis de glicose e hemoglobina estão dentro de faixas que consideramos seguras. Continue mantendo seus hábitos saudáveis, cuidando da alimentação e praticando exercícios, pois isso está ajudando muito a manter seu IMC em {data['bmi']}.
        
        Vamos apenas manter nossos check-ups de rotina para garantir que tudo continue bem.
        """
        
    return technical_response, patient_response


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
            ["Nunca fumou", "Ex-fumante", "Fumante atual", "Sem informação"]
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
        "glucose": glucose
    }
    
    with col2:
        status_container = st.empty()
        
        # Simulação do ML
        status_container.info("⚙️ Processando dados com o modelo de classificação...")
        prediction, proba = mock_model_predict(patient_data)
        
        # Simulação da LLM
        status_container.info("🧠 Processando dados com a IA Generativa...")
        tech_text, patient_text = mock_llm_generation(patient_data, prediction, proba)
        
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
        tab1, tab2 = st.tabs([
            "👨‍⚕️ Visão Médica (Técnica)",
            "🗣️ Visão Paciente (Humanizada)"
        ])
        
        with tab1:
            st.info("Esta seção contém dados estatísticos e terminologia técnica para apoio à decisão clínica.")
            st.markdown(tech_text)
            
            with st.expander("Ver dados brutos (JSON)"):
                st.json(patient_data)

        with tab2:
            st.success("Esta seção contém um roteiro de fala sugerido para comunicação empática.")
            st.markdown(f"> {patient_text}")
            
            st.button("📋 Copiar Texto para Prontuário", help="Simulação de cópia")

else:
    # Estado inicial
    with col2:
        st.info("👈 Preencha os dados clínicos ao lado e clique em 'Processar Diagnóstico' para ver a simulação.")

        st.markdown("""
        ### O que este sistema fará (Simulação):
        1.  **Processamento de ML:** Receberá os dados estruturados e passará pelo modelo otimizado (XGBoost/RandomForest).
        2.  **Geração de Texto:** O resultado numérico será enviado para uma LLM (Gemini).
        3.  **Saída Dupla:**
            * Relatório técnico para o profissional de saúde.
            * Explicação acessível para o paciente.
        """)