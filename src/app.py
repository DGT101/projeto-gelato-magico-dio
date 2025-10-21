import os
from flask import Flask, request, jsonify
import joblib

# --- Configuração ---
app = Flask(__name__)

# --- Carregamento do Modelo ---
MODEL_PATH = "model/model.pkl"
model = None

# Tenta carregar o modelo do arquivo .pkl
if os.path.exists(MODEL_PATH):
    print(f"Carregando modelo de '{MODEL_PATH}'...")
    try:
        model = joblib.load(MODEL_PATH)
        print("Modelo carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar o modelo de {MODEL_PATH}: {e}")
else:
    print(f"Arquivo do modelo não encontrado em {MODEL_PATH}.")
    print("Por favor, execute o 'src/train.py' primeiro para criar o modelo.")

# --- Definição dos Endpoints ---

@app.route('/')
def home():
    return "API do Modelo Gelato Mágico está no ar! Use /predict"

@app.route('/predict', methods=['POST'])
def predict():
    # Se o modelo não foi carregado na inicialização, retorna erro
    if model is None:
        return jsonify({"error": "Modelo não está carregado. Verifique os logs do servidor."}), 500

    try:
        data = request.get_json()
        if 'temperatura' not in data:
            return jsonify({"error": "Dado 'temperatura' não encontrado no corpo da requisição"}), 400

        import pandas as pd
        temp_df = pd.DataFrame({'temperatura': [data['temperatura']]})
        
        prediction = model.predict(temp_df)
        vendas_previstas = round(prediction[0], 0) 

        return jsonify({
            "temperatura_informada": data['temperatura'],
            "vendas_previstas": vendas_previstas
        })

    except Exception as e:
        return jsonify({"error": f"Erro durante a predição: {str(e)}"}), 500

# --- Execução da Aplicação ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001)) 
    app.run(host='0.0.0.0', port=port)