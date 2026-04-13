import cv2
import mediapipe as mp
import pickle
import pandas as pd
import os

pasta_imagens = "montar_palavra"

if not os.path.exists(pasta_imagens):
    os.makedirs(pasta_imagens)
    print(f"📁 Pasta '{pasta_imagens}' criada! Coloque suas imagens numeradas (1.jpg, 2.jpg...) lá dentro e rode.")
    exit()

print("⏳ Carregando a IA e preparando para ler a palavra...")

try:
    with open("modelos/modelo_libras.pkl", "rb") as f:
        modelo = pickle.load(f)
except:
    print("❌ ERRO: Arquivo 'modelos/modelo_libras.pkl' não encontrado.")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

arquivos = [f for f in os.listdir(pasta_imagens) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not arquivos:
    print(f"⚠️ A pasta '{pasta_imagens}' está vazia.")
    exit()

def extrair_numero(nome_arquivo):
    return int(nome_arquivo.split('.')[0])

arquivos.sort(key=extrair_numero)

palavra_formada = ""
print("\n🔍 Analisando as imagens...")

for nome_arquivo in arquivos:
    caminho = os.path.join(pasta_imagens, nome_arquivo)
    img = cv2.imread(caminho)
    
    if img is None: continue
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultados = hands.process(img_rgb)
    
    if resultados.multi_hand_landmarks:
        mao_escolhida = min(resultados.multi_hand_landmarks, key=lambda mao: mao.landmark[0].y)
        
        linha_dados = []
        pontos_x, pontos_y, pontos_z = [], [], []
        
        # 1. Pega pulso
        pulso_x = mao_escolhida.landmark[0].x
        pulso_y = mao_escolhida.landmark[0].y
        pulso_z = mao_escolhida.landmark[0].z
        
        # 2. Subtrai
        for ponto in mao_escolhida.landmark:
            pontos_x.append(ponto.x - pulso_x)
            pontos_y.append(ponto.y - pulso_y)
            pontos_z.append(ponto.z - pulso_z)
            
        # 3. Escala
        max_valor = max(max(map(abs, pontos_x)), max(map(abs, pontos_y)), max(map(abs, pontos_z)))
        if max_valor == 0: max_valor = 1.0
            
        for i in range(21):
            linha_dados.extend([pontos_x[i] / max_valor, pontos_y[i] / max_valor, pontos_z[i] / max_valor])
            
        colunas = []
        for i in range(21):
            colunas.extend([f'x{i}', f'y{i}', f'z{i}'])
            
        df_dados = pd.DataFrame([linha_dados], columns=colunas)
        
        letra_prevista = modelo.predict(df_dados)[0]
        palavra_formada += letra_prevista
        print(f"Foto {nome_arquivo} -> {letra_prevista}")
    else:
        print(f"Foto {nome_arquivo} -> [Nenhuma mão detectada]")
        palavra_formada += "_" 

print("-" * 30)
print(f"✨ PALAVRA FINAL TRADUZIDA: ** {palavra_formada} **")
print("-" * 30)