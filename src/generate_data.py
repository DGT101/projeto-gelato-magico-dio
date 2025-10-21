import pandas as pd
import numpy as np

# Definir a semente aleatória para reprodutibilidade
np.random.seed(42)

num_dias = 100
temperaturas = np.random.uniform(15, 35, num_dias) # Temperaturas entre 15°C e 35°C

# Criar vendas com uma correlação forte + ruído
# Vendas = (Temperatura * Fator) + Base + Ruído
vendas = (temperaturas * 25) + 50 + np.random.normal(0, 20, num_dias)

# Garantir que não há vendas negativas
vendas = np.maximum(vendas, 0).astype(int)

df = pd.DataFrame({
    'temperatura': temperaturas,
    'vendas': vendas
})

# Salvar na pasta 'inputs'
df.to_csv('inputs/dados_sorveteria.csv', index=False)

print("Arquivo 'inputs/dados_sorveteria.csv' criado com sucesso!")
print(df.head())