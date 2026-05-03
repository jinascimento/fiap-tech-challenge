import streamlit as st
import pandas as pd
from utils_gcp import upload_to_gcs, analyze_audio_with_gemini, analyze_video_with_gemini, analyze_video_with_intelligence_api, save_result_to_firestore, db

# Configuração da Página
st.set_page_config(page_title="Monitoramento Saúde Mulher", layout="wide")

st.title("🏥 Sistema de Monitoramento Multimodal")
st.markdown("---")

# Função para carregar dados reais do Firestore
def load_data_from_firestore():
    # Usando a nova coleção consolidada
    docs = db.collection("consultas_analises").order_by("data", direction="DESCENDING").limit(20).stream()
    data = []
    for doc in docs:
        d = doc.to_dict()
        d["id"] = doc.id
        data.append(d)
    return data

# Abas do sistema
tab_upload, tab_monitor = st.tabs(["📤 Upload de Consulta", "📊 Dashboard de Riscos"])

with tab_upload:
    st.header("Análise Multimodal (Áudio e Vídeo)")
    paciente = st.text_input("Nome da Paciente", "Paciente Anônima")
    
    tipo_arquivo = st.radio("Selecione o tipo de mídia:", ["Áudio", "Vídeo"], horizontal=True)
    
    if tipo_arquivo == "Áudio":
        uploaded_file = st.file_uploader("Escolha o arquivo de áudio (.wav, .mp3)", type=["wav", "mp3"])
        if uploaded_file:
            st.audio(uploaded_file, format='audio/wav')
    else:
        uploaded_file = st.file_uploader("Escolha o arquivo de vídeo (.mp4, .mov, .avi)", type=["mp4", "mov", "avi"])
        if uploaded_file:
            st.video(uploaded_file)

    if uploaded_file:
        if st.button(f"Analisar {tipo_arquivo} via Gemini AI"):
            with st.spinner(f"Processando {tipo_arquivo} via Vertex AI..."):
                try:
                    folder = "videos" if tipo_arquivo == "Vídeo" else "audios"
                    # 1. Upload
                    gcs_uri = upload_to_gcs(uploaded_file.getvalue(), uploaded_file.name, folder=folder)

                    # 2. Análise Multimodal
                    if tipo_arquivo == "Vídeo":
                        # Chamada para Video Intelligence API (Video AI)
                        video_ai_result = analyze_video_with_intelligence_api(gcs_uri)
                        
                        # Chamada para Gemini (Vertex AI)
                        analysis = analyze_video_with_gemini(gcs_uri)
                        
                        # Combinar resultados
                        analysis["video_intelligence"] = video_ai_result
                    else:
                        analysis = analyze_audio_with_gemini(gcs_uri)

                    # 3. Salvar no Firestore
                    record = save_result_to_firestore(paciente, analysis, tipo_analise=tipo_arquivo)

                    st.success(f"Análise de {tipo_arquivo} concluída e salva!")
                    
                    if tipo_arquivo == "Vídeo":
                        st.subheader("Resultados Combinados")
                        with st.expander("Detecção de Objetos/Cenas (Video Intelligence API)"):
                            labels = analysis.get("video_intelligence", {}).get("labels", [])
                            st.write(f"**Labels Detectados:** {', '.join(labels) if labels else 'Nenhum label detectado'}")
                    
                    # Layout para mostrar resultados do Gemini
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Status", analysis.get("status", "N/A"))
                    with col2:
                        st.metric("Risco", analysis.get("risco", "N/A"))
                    
                    st.write("**Análise Detalhada:**")
                    st.write(analysis.get("analise_detalhada", ""))
                    
                    st.write("**Sinais/Sintomas Detectados:**")
                    for item in analysis.get("sintomas_detectados", analysis.get("sinais_detectados", [])):
                        st.write(f"- {item}")
                        
                    st.info(f"**Recomendação:** {analysis.get('recomendacao', '')}")
                    
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
