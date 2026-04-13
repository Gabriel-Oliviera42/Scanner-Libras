import cv2
import mediapipe as mp
import os
import csv

# 1. Configurações iniciais
mp_hands = mp.solutions.hands
pasta_imagens = "dados/dataset_libras"       # Caminho atualizado
arquivo_csv = "dados/coordenadas_libras.csv" # Caminho atualizado

# 2. Criando o cabeçalho do nosso arquivo CSV
cabecalho = ['Letra']
for i in range(21):
    cabecalho.extend([f'x{i}', f'y{i}', f'z{i}'])

print("Iniciando a extração de coordenadas...")

# 3. Abrindo o arquivo CSV para escrever os dados
with open(arquivo_csv, mode='w', newline='', encoding='utf-8') as f:
    escritor = csv.writer(f)
    escritor.writerow(cabecalho)
    
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        
        # 4. Varrendo todas as imagens
        for nome_arquivo in os.listdir(pasta_imagens):
            if nome_arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                
                letra = nome_arquivo.replace("Letra ", "").replace("letra ", "")[0].upper()
                caminho_imagem = os.path.join(pasta_imagens, nome_arquivo)
                img = cv2.imread(caminho_imagem)
                
                if img is None:
                    continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                resultados = hands.process(img_rgb)
                
                # 5. Se achou alguma mão... (A MÁGICA FOI ATUALIZADA AQUI)
                if resultados.multi_hand_landmarks:
                    mao_escolhida = min(resultados.multi_hand_landmarks, key=lambda mao: mao.landmark[0].y)
                    
                    linha_dados = [letra]
                    pontos_x = []
                    pontos_y = []
                    pontos_z = []
                    
                    # Ponto 0 é o pulso. Vamos pegar a posição dele.
                    pulso_x = mao_escolhida.landmark[0].x
                    pulso_y = mao_escolhida.landmark[0].y
                    pulso_z = mao_escolhida.landmark[0].z
                    
                    # Passo 1: Translação (Centraliza no 0,0,0)
                    for ponto in mao_escolhida.landmark:
                        pontos_x.append(ponto.x - pulso_x)
                        pontos_y.append(ponto.y - pulso_y)
                        pontos_z.append(ponto.z - pulso_z)
                        
                    # Passo 2: Normalização de Escala (Ignora a distância)
                    max_valor = max(
                        max(map(abs, pontos_x)),
                        max(map(abs, pontos_y)),
                        max(map(abs, pontos_z))
                    )
                    
                    # Previne divisão por zero
                    if max_valor == 0: max_valor = 1.0
                        
                    # Divide e salva no CSV
                    for i in range(21):
                        linha_dados.extend([
                            pontos_x[i] / max_valor, 
                            pontos_y[i] / max_valor, 
                            pontos_z[i] / max_valor
                        ])
                    
                    escritor.writerow(linha_dados)
                    print(f"Coordenadas da letra '{letra}' ({nome_arquivo}) salvas com sucesso!")
                else:
                    print(f"Nenhuma mão detectada na imagem: {nome_arquivo}")

print(f"\nProcesso finalizado! Arquivo gerado: {arquivo_csv}")