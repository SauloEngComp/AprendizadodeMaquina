import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr

def train_model():
    """
    Carrega os dados, treina o modelo e GERA OS 3 GRÁFICOS de performance.
    Retorna o modelo, scaler, colunas e as figuras dos gráficos.
    """
    try:
        df = pd.read_excel("insurance_organizado.xlsx", skiprows=[1])
    except FileNotFoundError:
        print("Erro: O arquivo 'insurance_organizado.xlsx' não foi encontrado.")
        return None, None, None, None, None, None

    df = df.rename(columns={"IMC": "imc", "crianças": "criancas", "região": "regiao"})

    media_enc = df["encargos"].median()
    df["alto_custo"] = (df["encargos"] > media_enc).astype(int)
    X = df.drop(columns=["encargos", "alto_custo"])
    y = df["alto_custo"]
    
    X_encoded = pd.get_dummies(X, columns=['sexo', 'fumante', 'regiao'], drop_first=True)
    encoded_columns = X_encoded.columns
    
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    # --- GERAÇÃO DOS GRÁFICOS DE PERFORMANCE ---
    
    # 1. Matriz de Confusão
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Previsto Baixo', 'Previsto Alto'], 
                yticklabels=['Verdadeiro Baixo', 'Verdadeiro Alto'])
    plt.xlabel('Previsão do Modelo')
    plt.ylabel('Valor Real')
    plt.title('Matriz de Confusão', fontsize=16)
    plt.close()

    # 2. Importância das Variáveis (Feature Importance)
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': encoded_columns, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    
    fig_importance = plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
    plt.title('Importância de Cada Variável', fontsize=16)
    plt.xlabel('Nível de Importância')
    plt.ylabel('Variável')
    plt.tight_layout()
    plt.close()

    # 3. Curva ROC
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig_roc = plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC (Receiver Operating Characteristic)', fontsize=16)
    plt.legend(loc="lower right")
    plt.close()
    
    print("✅ Modelo treinado e gráficos de performance gerados com sucesso!")
    return model, scaler, encoded_columns, fig_cm, fig_importance, fig_roc

# Treina o modelo e obtém os artefatos
model, scaler, encoded_columns, conf_matrix_fig, feat_imp_fig, roc_fig = train_model()

# --- FUNÇÃO DE PREVISÃO (Sem alterações na lógica interna) ---
def predict_insurance_cost(idade, sexo, imc, criancas, fumante, regiao):
    input_data = pd.DataFrame({
        "idade": [idade], "sexo": [sexo], "imc": [imc],
        "criancas": [criancas], "fumante": [fumante], "regiao": [regiao]
    })
    input_encoded = pd.get_dummies(input_data, columns=['sexo', 'fumante', 'regiao'], drop_first=True)
    input_reindexed = input_encoded.reindex(columns=encoded_columns, fill_value=0)
    input_scaled = scaler.transform(input_reindexed)
    prediction_proba = model.predict_proba(input_scaled)[0]
    class_labels = ["Custo Baixo", "Custo Alto"]
    result = {label: prob for label, prob in zip(class_labels, prediction_proba)}
    return result

# --- CONSTRUÇÃO DA NOVA INTERFACE COM ABAS E 3 GRÁFICOS ---
if model is not None:
    with gr.Blocks(theme=gr.themes.Soft(), title="Análise de Custo de Seguro") as iface:
        gr.Markdown("# 🩺 Análise Preditiva de Custo de Seguro Médico", elem_id="main-title")
        
        with gr.Tabs():
            # Aba 1: Classificador Interativo
            with gr.TabItem("Classificador Interativo"):
                gr.Markdown("### Insira os dados do paciente para obter uma previsão de custo.")
                with gr.Row():
                    with gr.Column(scale=1):
                        idade = gr.Slider(minimum=18, maximum=100, value=30, step=1, label="Idade")
                        sexo = gr.Radio(choices=["male", "female"], value="male", label="Sexo")
                        imc = gr.Slider(minimum=10.0, maximum=60.0, value=25.0, step=0.1, label="IMC")
                        criancas = gr.Slider(minimum=0, maximum=10, value=0, step=1, label="Número de Crianças")
                    with gr.Column(scale=1):
                        fumante = gr.Radio(choices=["Sim", "não"], value="não", label="É Fumante?")
                        regiao = gr.Dropdown(choices=["southeast", "southwest", "northeast", "northwest"], value="southeast", label="Região")
                        submit_btn = gr.Button("Analisar Custo", variant="primary", elem_id="submit-button")
                
                gr.Markdown("---")
                gr.Markdown("### Resultado da Previsão")
                output_label = gr.Label(label="Probabilidade de Custo")
            
            # Aba 2: Performance do Modelo
            with gr.TabItem("Performance do Modelo"):
                gr.Markdown("## Análise Visual da Performance do Modelo")
                
                gr.Markdown("### Matriz de Confusão")
                gr.Markdown("Mostra o total de acertos e erros do modelo.")
                gr.Plot(value=conf_matrix_fig)
                
                gr.Markdown("### Importância das Variáveis")
                gr.Markdown("Este gráfico classifica as variáveis da mais para a menos influente na previsão do modelo. Variáveis no topo são as que o modelo considera mais importantes.")
                gr.Plot(value=feat_imp_fig)

                gr.Markdown("### Curva ROC")
                gr.Markdown("A Curva ROC avalia a capacidade do classificador de distinguir entre as classes. Quanto mais a linha laranja se aproxima do canto superior esquerdo, melhor é o modelo. A área sob a curva (AUC) resume essa performance em um único número (máximo 1.0).")
                gr.Plot(value=roc_fig)

        # Conectar o botão à função de previsão
        submit_btn.click(
            fn=predict_insurance_cost,
            inputs=[idade, sexo, imc, criancas, fumante, regiao],
            outputs=output_label
        )

    if __name__ == "__main__":
        iface.launch()
