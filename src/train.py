import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import joblib
import os

# 1. Iniciar um "Run" do MLflow
# O MLflow vai criar uma pasta 'mlruns' para salvar tudo
with mlflow.start_run():

    # --- Carregamento e Preparação dos Dados ---
    print("Carregando dados...")
    df = pd.read_csv('inputs/dados_sorveteria.csv')

    # Definir X (features) e y (target)
    X = df[['temperatura']]
    y = df['vendas']

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Treinamento do Modelo ---
    print("Treinando modelo...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # --- Avaliação do Modelo ---
    print("Avaliando modelo...")
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Modelo treinado!")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.2f}")

    # --- Registro no MLflow ---
    print("Registrando no MLflow...")

    # Logar métricas
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    # Logar o modelo
    # O 'model_name' é o que registra o modelo no "Model Registry"
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="GelatoMagicoModel" # Nome do modelo no Registry
    )
    
    # --- Salvar cópia local para a API ---
    print("Salvando cópia do modelo em 'model/model.pkl'...")
    os.makedirs('model', exist_ok=True) # Cria a pasta 'model' se não existir
    joblib.dump(model, 'model/model.pkl')

    print("\n--- Run MLflow finalizado ---")
