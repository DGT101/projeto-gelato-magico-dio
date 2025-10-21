🍦 Projeto Gelato Mágico: Previsão de Vendas com MLOps

Projeto prático de Machine Learning e MLOps desenvolvido para o bootcamp da DIO. O objetivo é criar um sistema que prevê a quantidade de sorvetes a serem vendidos pela sorveteria "Gelato Mágico" com base na temperatura do dia, otimizando a produção e evitando desperdícios.

Este repositório documenta todo o ciclo de vida do projeto, desde a geração dos dados e treinamento do modelo até o deploy de uma API de previsão em um ambiente de nuvem.

🚀 Tecnologias Utilizadas

Linguagem: Python 3

Análise de Dados: Pandas, Matplotlib, Seaborn

Machine Learning: Scikit-learn (Regressão Linear)

MLOps: MLflow para rastreamento de experimentos e gerenciamento de modelos.

API: Flask e Gunicorn para servir o modelo via REST API.

Cloud (Deploy): Render para a hospedagem e disponibilização da API na web.

📊 O Pipeline de MLOps

O projeto foi estruturado para ser um pipeline de MLOps simples, porém completo e reprodutível.

1. Análise e Geração de Dados

Os dados foram gerados sinteticamente (src/generate_data.py) para simular uma correlação forte e positiva entre a temperatura e as vendas. A análise exploratória confirma essa relação, que é a premissa fundamental para o nosso modelo de regressão.

*Exemplo: *
![alt text](image.png)

2. Treinamento e Rastreamento com MLflow

O script src/train.py é responsável por:

Carregar os dados.

Dividir os dados em conjuntos de treino e teste.

Treinar um modelo de Regressão Linear.

Registrar os parâmetros, métricas (como R² e RMSE) e o próprio modelo como um "artefato" em um experimento do MLflow.

Isso nos permite comparar diferentes treinamentos e gerenciar as versões do modelo de forma organizada.

*Exemplo: *
![alt text](image-1.png)

3. API de Previsão

Uma API REST foi desenvolvida com Flask (src/app.py) para servir o modelo treinado. Ela expõe um endpoint /predict que recebe a temperatura e retorna a previsão de vendas. Para o deploy, o modelo foi serializado com joblib, garantindo que a API seja independente.

O teste local com curl confirma que a API está respondendo corretamente.

☁️ API em Produção

A API foi colocada em produção na plataforma Render. Você pode fazer previsões em tempo real enviando uma requisição POST.

URL da API: https://gelato-magico-api.onrender.com

Como usar:

curl -X https://gelato-magico-api.onrender.com/predict \
     -H "Content-Type: application/json" \
     -d '{"temperatura": 32.5}'


Exemplo de Resposta:

{
  "temperatura_informada": 32.5,
  "vendas_previstas": 865.0
}


🔧 Como Executar Localmente

Clone o repositório:

git clone [https://github.com/DGT101/projeto-gelato-magico-dio.git](https://github.com/DGT101/projeto-gelato-magico-dio.git)
cd SEU-REPOSITORIO


Crie e ative um ambiente virtual:

python -m venv venv
source venv/bin/activate


Instale as dependências:

pip install -r requirements.txt


Treine o modelo (isso criará o model/model.pkl):

python src/train.py


Inicie a API Flask:

python src/app.py


Em outro terminal, teste o endpoint:

curl -X POST [http://127.0.0.1:5001/predict](http://127.0.0.1:5001/predict) -H "Content-Type: application/json" -d '{"temperatura": 25}'


🔮 Possibilidades e Próximos Passos

Features Adicionais: Enriquecer o modelo com novas variáveis, como "dia da semana", "feriado" ou "previsão de chuva".

Modelos Avançados: Experimentar algoritmos mais complexos (como Random Forest ou Gradient Boosting) e comparar sua performance usando o MLflow.

Testes Automatizados: Implementar testes de unidade para a API para garantir a robustez.
