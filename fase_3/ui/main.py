import time

import streamlit as st

st.set_page_config(
    page_title="Assistente Médico Virtual - Tech Challenge",
    page_icon="🏥",
)

st.title("🏥 Assistente Médico Hospitalar")
st.markdown("""
Esta interface permite a interação com a LLM personalizada treinada com protocolos internos.
O sistema utiliza **LangChain** para contextualização e **LangGraph** para fluxos de decisão.
""")

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "source" in message:
            st.caption(f"📌 Fonte: {message['source']}")


if prompt := st.chat_input("Como posso auxiliar na conduta clínica hoje?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        assistant_response = (
            "Baseado nos protocolos internos do hospital e nos dados do paciente..."
        )
        source_info = "Protocolo de Manejo Clínico v2.1 - Setor de Cardiologia"

        # Efeito de digitação
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)
        st.caption(f"📌 Fonte: {source_info}")

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response, "source": source_info}
    )

with st.sidebar:
    st.header("Auditoria e Logs")
    st.info("Pipeline: LangChain + Fine-tuned LLM")

    if st.button("Limpar Histórico"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("⚠️ Limites de Atuação")
    st.warning(
        "Este assistente é uma ferramenta de apoio. Nunca prescreva sem validação humana."
    )
