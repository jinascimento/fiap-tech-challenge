# Projeto de Classificação de Pacientes com Diabetes

## 📋 Descrição

Este projeto implementa uma solução de Machine Learning para classificar pacientes que possuem diabetes com base em dados médicos. O modelo utiliza algoritmos de classificação para analisar características clínicas e prever a presença da doença.

## 🎯 Objetivos

- Realizar análise exploratória dos dados (EDA)
- Treinar e comparar diferentes algoritmos de machine learning
- Avaliar o desempenho dos modelos usando métricas apropriadas
- Selecionar o modelo com melhor desempenho para classificação de diabetes

## 📊 Dataset

O dataset utilizado contém informações de **100.000 pacientes** com as seguintes características:

### Features (Variáveis de Entrada):
- **gender**: Gênero do paciente
- **age**: Idade
- **hypertension**: Hipertensão (0 = não, 1 = sim)
- **heart_disease**: Doença cardíaca (0 = não, 1 = sim)
- **smoking_history**: Histórico de tabagismo
- **bmi**: Índice de massa corporal
- **HbA1c_level**: Nível de hemoglobina glicada
- **blood_glucose_level**: Nível de glicose no sangue

Deve ser baixado através do link [Diabetes prediction dataset
](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset/data), disponível no `Kaggle`

### Target (Variável Alvo):
- **diabetes**: Presença de diabetes (0 = não, 1 = sim)

## 🔬 Análise Exploratória

O projeto inclui uma análise estatística abrangente que identifica:

- **Associação entre tabagismo e diabetes**: Ex-fumantes apresentam maior prevalência (17%)
- **Correlação idade-diabetes**: Pacientes diabéticos são em média 20 anos mais velhos
- **Impacto do IMC**: Pacientes com diabetes têm IMC médio significativamente maior
- **Relação com comorbidades**: Hipertensão e doenças cardíacas aumentam o risco

## 🤖 Modelos Testados

### 1. Regressão Logística
- **Acurácia**: 96.0%
- **Precision**: 0.84
- **Recall**: 0.49
- **F1-Score**: 0.61

### 2. Regressão Logística (Balanceada)
- **Acurácia**: 86.4%
- **Precision**: 0.29
- **Recall**: 0.87
- **F1-Score**: 0.44

### 3. Random Forest Classifier
- **Acurácia**: 96.2%
- **Precision**: 0.75
- **Recall**: 0.59
- **F1-Score**: 0.66

### 4. K-Nearest Neighbors (KNN)
- **Acurácia**: 96.2%
- **Precision**: 0.84
- **Recall**: 0.48
- **F1-Score**: 0.61

### 5. MLP Classifier (Rede Neural) ⭐
- **Acurácia**: 97.2%
- **Precision**: 0.98
- **Recall**: 0.56
- **F1-Score**: 0.72

## 🏆 Modelo Selecionado

O **MLP Classifier** foi escolhido como o melhor modelo devido a:

- **Maior F1-Score (0.72)**: Melhor equilíbrio entre precisão e recall
- **Alta Precisão (98%)**: Minimiza falsos positivos
- **Confiabilidade**: Apenas 10 falsos positivos em 17.811 predições

## 🛠️ Tecnologias Utilizadas

- **Python 3.x**
- **Pandas**: Manipulação de dados
- **NumPy**: Computação numérica
- **Scikit-learn**: Machine Learning
- **Matplotlib/Seaborn**: Visualização
- **SciPy**: Testes estatísticos
- **SHAP**: Interpretabilidade do modelo

## 📦 Instalação e Execução

### Pré-requisitos
- Python 3.8+
- Jupyter Notebook

### Instalação das Dependências
```bash
pip install -r requirements.txt
```

### Execução
```bash
jupyter notebook classificacao_pacientes_X_diabetes.ipynb
```

## 🐳 Docker

Para executar o projeto usando Docker:

```bash
# Construir a imagem
docker build -t diabetes-classification .

# Executar o container
docker run -p 8888:8888 diabetes-classification
```

## 📈 Resultados Principais

- **Dataset balanceado**: 10% dos pacientes possuem diabetes
- **Preprocessamento robusto**: Remoção de outliers e duplicatas
- **Feature Engineering**: Criação de faixas etárias e codificação categórica
- **Modelo otimizado**: MLP Classifier com 97.2% de acurácia

## 🔍 Interpretabilidade

O projeto utiliza SHAP (SHapley Additive exPlanations) para explicar as predições do modelo, permitindo entender quais características são mais importantes para cada classificação.

## 📊 Métricas de Avaliação

- **Acurácia**: Proporção de predições corretas
- **Precision**: Proporção de verdadeiros positivos entre todos os positivos
- **Recall**: Proporção de verdadeiros positivos identificados
- **F1-Score**: Média harmônica entre precision e recall

## 🎯 Aplicações Práticas

Este modelo pode ser utilizado para:

- **Triagem médica**: Identificação precoce de pacientes em risco
- **Apoio à decisão clínica**: Ferramenta auxiliar para profissionais de saúde
- **Prevenção**: Identificação de fatores de risco modificáveis
- **Pesquisa médica**: Análise de padrões em grandes volumes de dados

## 📝 Considerações Éticas

- **Não substitui diagnóstico médico**: Ferramenta de apoio à decisão
- **Privacidade dos dados**: Anonimização e proteção de informações sensíveis
- **Viés algorítmico**: Monitoramento contínuo de performance entre diferentes grupos
- **Transparência**: Explicabilidade das decisões do modelo

## 👥 Autores

- **Projeto**: Classificação de Pacientes com Diabetes
- **Versão**: 1.0
- **Data**: 2024

## 📄 Licença

Este projeto é destinado para fins educacionais e de pesquisa.

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor, abra uma issue ou pull request para sugerir melhorias.

## 📞 Contato

Para dúvidas ou sugestões sobre o projeto, entre em contato através dos canais apropriados.
