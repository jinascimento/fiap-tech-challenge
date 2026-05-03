import streamlit as st
import pandas as pd
from utils_gcp import upload_to_gcs, analyze_audio_with_gemini, save_result_to_firestore, db

# Configuração da Página
st.set_page_config(page_title="Monitoramento Saúde Mulher", layout="wide")

st.title("🏥 Sistema de Monitoramento Multimodal")
st.markdown("---")

# Função para carregar dados reais do Firestore
def load_data_from_firestore():
    docs = db.collection("consultas_audio").order_by("data", direction="DESCENDING").limit(20).stream()
    data = []
    for doc in docs:
        d = doc.to_dict()
        d["id"] = doc.id
        data.append(d)
    return data

# Abas do sistema
tab_upload, tab_monitor = st.tabs(["📤 Upload de Consulta", "📊 Dashboard de Riscos"])

with tab_upload:
    st.header("Upload de Áudio da Consulta")
    paciente = st.text_input("Nome da Paciente", "Paciente Anônima")
    uploaded_file = st.file_uploader("Escolha o arquivo de áudio (.wav, .mp3)", type=["wav", "mp3"])

    if uploaded_file:
        st.audio(uploaded_file, format='audio/wav')

        if st.button("Processar via IA"):
            with st.spinner("Enviando para o bucket e analisando via Gemini Multimodal..."):
                try:
                    # 1. Upload
                    gcs_uri = upload_to_gcs(uploaded_file.getvalue(), uploaded_file.name)

                    # 2. Análise via Gemini
                    analysis = analyze_audio_with_gemini(gcs_uri)

                    # 3. Salvar no Firestore
                    record = save_result_to_firestore(paciente, analysis)

                    st.success("Análise concluída e salva no banco de dados!")
                    st.json(analysis) # Mostra o resultado bruto para validação
                except Exception as e:
                    st.error(f"Erro no processamento: {str(e)}")

with tab_monitor:
    st.header("Monitoramento de Consultas (Dados Reais do Firestore)")

    results = load_data_from_firestore()
    if results:
        df = pd.DataFrame(results)
        # Mostrar apenas colunas principais no dataframe
        cols_to_show = ["data", "paciente", "status", "risco", "tipo"]
        event = st.dataframe(df[cols_to_show], on_select="rerun", selection_mode="single-row")

        if event.selection.rows:
            idx = event.selection.rows[0]
            selected_row = df.iloc[idx]

            st.subheader(f"Detalhes da Consulta: {selected_row['paciente']}")

            detalhes = selected_row.get("detalhes", {})
            st.write(f"**Análise Detalhada:** {detalhes.get('analise_detalhada', 'N/A')}")
            st.write("**Sintomas Detectados:**")
            for sintoma in detalhes.get('sintomas_detectados', []):
                st.write(f"- {sintoma}")
            st.info(f"**Recomendação Médica:** {detalhes.get('recomendacao', 'N/A')}")
    else:
        st.info("Nenhuma consulta processada ainda.")

# Footer simples
st.sidebar.markdown("---")
st.sidebar.info("Projeto Tech Challenge 4 - Pós Tech")