import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# --- CONFIGURAÇÕES DE CAMINHO ---
CAMINHO_CSV = 'dados/Dataset_Libras/dataset_libras.csv'
CAMINHO_MODELO = 'modelo_libras.pkl'

print("[1/4] Carregando os dados do CSV...")
try:
    df = pd.read_csv(CAMINHO_CSV)
except FileNotFoundError:
    print(f"[ERRO] O arquivo não foi encontrado no caminho: {CAMINHO_CSV}")
    exit()

# Separar as Respostas (y) dos Dados/Coordenadas (X)
X = df.drop('letra', axis=1) 
y = df['letra']              

# Dividir os dados: 80% para ensinar a IA, 20% para a "prova" final
print("[2/4] Separando dados de treino e teste...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"[3/4] Treinando a Inteligência Artificial com {len(X_train)} amostras (isso pode levar alguns segundos)...")
modelo = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
modelo.fit(X_train, y_train)

# Aplicar a "prova" nos 20% que a IA não viu durante o treino
previsoes = modelo.predict(X_test)
precisao = accuracy_score(y_test, previsoes)

print(f" RESULTADO DA PROVA: Acurácia de {precisao * 100:.2f}% ")


print("[4/4] Salvando o modelo treinado...")
with open(CAMINHO_MODELO, 'wb') as f:
    pickle.dump(modelo, f)

print(f"[SUCESSO] O arquivo '{CAMINHO_MODELO}' foi gerado na raiz do seu projeto!")