import cv2
import mediapipe as mp
import csv
import os

# --- CONFIGURAÇÕES ---
ARQUIVO_CSV = 'dados/Dataset_Libras/dataset_libras.csv'
TOTAL_AMOSTRAS = 1000

# Letras permitidas (sem movimento)
LETRAS_PERMITIDAS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y']

# Inicializa MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Prepara o arquivo CSV
cabecalho = ['letra'] + [f'{eixo}{i}' for i in range(21) for eixo in ['x', 'y', 'z']]

if not os.path.exists(ARQUIVO_CSV):
    with open(ARQUIVO_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(cabecalho)
    print(f"[INFO] Arquivo {ARQUIVO_CSV} criado com cabeçalhos.")

# Variáveis de controle de gravação
gravando = False
letra_atual = ""
contador = 0

# Inicia a Câmera
cap = cv2.VideoCapture(0)

print("\n--- COMANDOS ---")
print("- Pressione a tecla da LETRA que deseja gravar (ex: 'A').")
print("- Pressione 'H' ou 'ESC' para sair do programa.")
print("- Lembre-se: Mova a mão LENTAMENTE durante a gravação para gerar variabilidade!\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1) # Espelha para ficar intuitivo
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = hands.process(frame_rgb)

    # --- LÓGICA DE GRAVAÇÃO ---
    if gravando and resultados.multi_hand_landmarks:
        mao = resultados.multi_hand_landmarks[0]
        
        # 1. Normalização (O "Zoom" e Centralização Matemáticos)
        pontos_x, pontos_y, pontos_z = [], [], []
        pulso_x, pulso_y, pulso_z = mao.landmark[0].x, mao.landmark[0].y, mao.landmark[0].z
        
        for ponto in mao.landmark:
            pontos_x.append(ponto.x - pulso_x)
            pontos_y.append(ponto.y - pulso_y)
            pontos_z.append(ponto.z - pulso_z)
            
        max_valor = max(max(map(abs, pontos_x)), max(map(abs, pontos_y)), max(map(abs, pontos_z)))
        if max_valor == 0: max_valor = 1.0
        
        linha_dados = [letra_atual]
        for i in range(21):
            linha_dados.extend([pontos_x[i] / max_valor, pontos_y[i] / max_valor, pontos_z[i] / max_valor])
            
        # 2. Salvar no CSV
        with open(ARQUIVO_CSV, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(linha_dados)
            
        contador += 1
        
        # 3. Finaliza se bateu a meta
        if contador >= TOTAL_AMOSTRAS:
            gravando = False
            print(f"[SUCESSO] {TOTAL_AMOSTRAS} amostras da letra '{letra_atual}' coletadas!")

    # --- DESENHOS NA TELA (UI) ---
    if resultados.multi_hand_landmarks:
        mp_drawing.draw_landmarks(frame, resultados.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

    # Textos de instrução
    if gravando:
        cv2.putText(frame, f"GRAVANDO: {letra_atual}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(frame, f"{contador} / {TOTAL_AMOSTRAS}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(frame, "MOVA A MAO LENTAMENTE", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "Aperte a LETRA para iniciar", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Amostras salvas no CSV. Pronto.", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('Coletor de Dados - PDI', frame)

    # --- CAPTURA DE TECLAS ---
    tecla = cv2.waitKey(1) & 0xFF
    
    # Sair do programa
    if tecla == ord('h') or tecla == 27: # 27 é o ESC
        break
        
    # Iniciar gravação se apertou uma letra válida e não está gravando no momento
    elif not gravando:
        char_tecla = chr(tecla).upper()
        if char_tecla in LETRAS_PERMITIDAS:
            letra_atual = char_tecla
            contador = 0
            gravando = True
            print(f"\n[INFO] Iniciando gravação da letra: {letra_atual}")

cap.release()
cv2.destroyAllWindows()