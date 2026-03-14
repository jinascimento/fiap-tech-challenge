import time

import streamlit as st
from agents.graph import app

st.set_page_config(
    page_title="Assistente Médico - Tech Challenge",
    page_icon="🏥",
    layout="wide",
)

st.markdown(
    "<style> .stAlert { margin-top: 10px; } </style>",
    unsafe_allow_html=True,
)

st.title("🏥 Assistente Médico Hospitalar")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "source" in message:
            st.caption(f"📌 **Fontes:** {message['source']}")
        if message.get("risk") == "ALTO":
            st.error("⚠️ Risco Elevado Detectado")

# Chat
if prompt := st.chat_input("Como posso auxiliar na conduta clínica hoje?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Processando fluxo clínico...", expanded=True) as status:
            st.write("🔍 Classificando intenção...")

            inputs = {"input": prompt, "logs": []}
            final_state = app.invoke(inputs)

            st.write("🆔 Identificando paciente...")

            # time.sleep(1)  # Simulação de latência para UX

            st.write(
                f"📊 Consultando banco de dados (ID: {final_state.get('patient_id')})..."
            )

            # time.sleep(1)

            st.write("🧠 Analisando protocolos e gerando conduta...")

            status.update(label="Análise Concluída!", state="complete", expanded=False)

        # Resposta Final
        full_response = final_state.get("response", "Erro ao processar resposta.")

        st.markdown(full_response)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "source": ", ".join(final_state.get("sources", ["Interna"])),
                "risk": final_state.get("risk_level", "BAIXO"),
                "full_logs": final_state.get("logs", []),
            }
        )


with st.sidebar:
    st.header("🕵️ Auditoria do Agente")

    if st.session_state.messages:
        last_message = st.session_state.messages[-1]
        if "full_logs" in last_message:
            st.subheader("Rastro de Decisão")
            for log in last_message["full_logs"]:
                st.write(f"• {log}")

    st.divider()

    if st.button("Limpar Histórico"):
        st.session_state.messages = []
        st.rerun()

    st.subheader("⚠️ Limites de Atuação")
    st.warning("Validação humana obrigatória.")
