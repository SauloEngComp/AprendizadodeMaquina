import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import gradio as gr

# --- 1. LÓGICA ORIGINAL PARA TREINAMENTO DO MODELO ---
# (Esta parte roda uma vez para preparar o modelo)

def generate_dataset(n=1000):
    """Gera um dataset sintético mais robusto para o modelo."""
    np.random.seed(42)
    df = pd.DataFrame({
        "idade": np.random.randint(18, 65, size=n),
        "sexo": np.random.choice(["male", "female"], size=n),
        "imc": np.round(np.random.normal(28, 6, size=n), 2),
        "criancas": np.random.poisson(1, size=n),
        "fumante": np.random.choice(["yes", "no"], size=n, p=[0.25, 0.75]),
        "regiao": np.random.choice(["southeast", "southwest", "northeast", "northwest"], size=n)
    })
    
    # Fórmula para criar uma relação mais clara entre as features e o custo
    base = 2000 + (df["idade"]**2) + ((df["imc"] - 25)**3 * 10) + (df["criancas"] * 500)
    # Fumantes têm um custo exponencialmente maior
    custo_fumante = np.exp(0.1 * df["idade"]) * (df["fumante"] == "yes").astype(int) * 200
    df["encargos"] = np.round(base + custo_fumante + np.random.normal(0, 1000, size=n), 2)
    # Garante que os encargos não sejam negativos
    df['encargos'] = df['encargos'].clip(lower=0)
    return df

def train_model():
    """
    Carrega os dados, pré-processa e treina o modelo.
    Retorna os artefatos necessários para a previsão: o modelo, o scaler e as colunas.
    """
    # Carregar e preparar os dados
    df = generate_dataset()
    media_enc = df["encargos"].median() # Usar mediana pode ser mais robusto a outliers
    df["alto_custo"] = (df["encargos"] > media_enc).astype(int)

    # Pré-processamento
    X = df.drop(columns=["encargos", "alto_custo"])
    y = df["alto_custo"]
    
    # One-Hot Encoding
    X_encoded = pd.get_dummies(X, columns=['sexo', 'fumante', 'regiao'], drop_first=True)
    
    # Guardar as colunas codificadas para usar na previsão
    encoded_columns = X_encoded.columns
    
    # Normalização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    # Divisão treino/teste
    X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # Treinamento do Modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    print("✅ Modelo treinado com sucesso!")
    return model, scaler, encoded_columns

# Treina o modelo e obtém os artefatos necessários
model, scaler, encoded_columns = train_model()


# --- 2. FUNÇÃO DE PREVISÃO PARA A INTERFACE ---

def predict_insurance_cost(idade, sexo, imc, criancas, fumante, regiao):
    """
    Recebe os dados do usuário, processa-os e retorna a previsão do modelo.
    """
    # Criar um DataFrame com os dados de entrada
    input_data = pd.DataFrame({
        "idade": [idade],
        "sexo": [sexo],
        "imc": [imc],
        "criancas": [criancas],
        "fumante": [fumante],
        "regiao": [regiao]
    })
    
    # Aplicar One-Hot Encoding EXATAMENTE como no treino
    input_encoded = pd.get_dummies(input_data, columns=['sexo', 'fumante', 'regiao'], drop_first=True)
    
    # Alinhar as colunas com as do modelo treinado
    # Isso garante que todas as colunas esperadas pelo modelo existam, preenchendo com 0 as que faltarem.
    input_reindexed = input_encoded.reindex(columns=encoded_columns, fill_value=0)
    
    # Aplicar a normalização com o MESMO scaler do treino
    input_scaled = scaler.transform(input_reindexed)
    
    # Fazer a previsão de probabilidade
    prediction_proba = model.predict_proba(input_scaled)[0]
    
    # Formatar o resultado para o Gradio
    # O componente gr.Label interpreta dicionários para mostrar confianças
    class_labels = ["Custo Baixo", "Custo Alto"]
    result = {label: prob for label, prob in zip(class_labels, prediction_proba)}
    
    return result

# --- 3. CONSTRUÇÃO DA INTERFACE COM GRADIO ---

with gr.Blocks(theme=gr.themes.Soft(), title="Classificador de Custo de Seguro") as iface:
    gr.Markdown(
        """
        # 🩺 Classificador de Custo de Seguro Médico
        Insira os dados abaixo para prever se o custo do seguro será classificado como **Alto** ou **Baixo**.
        O modelo foi treinado com um RandomForestClassifier.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Dados Pessoais")
            idade = gr.Slider(minimum=18, maximum=100, value=30, step=1, label="Idade")
            sexo = gr.Radio(choices=["male", "female"], value="male", label="Sexo")
            imc = gr.Slider(minimum=10.0, maximum=60.0, value=25.0, step=0.1, label="Índice de Massa Corporal (IMC)")
            criancas = gr.Slider(minimum=0, maximum=10, value=0, step=1, label="Número de Crianças")
            
        with gr.Column(scale=1):
            gr.Markdown("### Hábitos e Localização")
            fumante = gr.Radio(choices=["yes", "no"], value="no", label="É Fumante?")
            regiao = gr.Dropdown(
                choices=["southeast", "southwest", "northeast", "northwest"],
                value="southeast",
                label="Região"
            )
            
            # Botão de previsão
            submit_btn = gr.Button("Analisar Custo", variant="primary")
            
    with gr.Row():
         # Saída
        output_label = gr.Label(label="Resultado da Classificação")
        
    # Conectar o botão à função de previsão
    submit_btn.click(
        fn=predict_insurance_cost,
        inputs=[idade, sexo, imc, criancas, fumante, regiao],
        outputs=output_label
    )
    
    gr.Markdown(
        """
        ---
        *Exemplo de uso: Tente alterar o campo "É Fumante?" de "no" para "yes" e veja o impacto significativo na previsão de custo alto.*
        """
    )

# Lançar a interface
if __name__ == "__main__":
    iface.launch()