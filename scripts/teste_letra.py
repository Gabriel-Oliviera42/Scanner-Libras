import cv2
import mediapipe as mp

# 1. Iniciando o MediaPipe (versão clássica que consertamos)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 2. Carregando a sua imagem (garanta que o nome está correto)
caminho_imagem = "dataset_libras/Letra A - Libras.png"
img = cv2.imread(caminho_imagem)

if img is None:
    print(f"❌ Erro: Não consegui achar a imagem '{caminho_imagem}'. Veja se o nome e a extensão (.jpg, .png) estão certos.")
else:
    print("⏳ Processando a imagem...")
    
    # O OpenCV lê imagens em BGR (Azul, Verde, Vermelho), mas o MediaPipe exige RGB.
    # Então precisamos converter as cores antes de enviar para o motor.
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 3. Ligando o motor de detecção de mãos
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        
        # O motor analisa a foto e procura as mãos
        resultados = hands.process(img_rgb)

        # Se ele encontrar alguma mão (os 21 pontos)...
        if resultados.multi_hand_landmarks:
            print("Mão detectada com sucesso! Desenhando a teia verde...")
            
            for hand_landmarks in resultados.multi_hand_landmarks:
                # Desenha os pontos vermelhos e as linhas verdes em cima da foto original
                mp_drawing.draw_landmarks(
                    img, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )
            
            # 4. Mostra o resultado na tela
            print("Pressione qualquer tecla na janela da imagem para fechar.")
            cv2.imshow("Teste Libras - MediaPipe", img)
            
            # Espera você apertar uma tecla para fechar a janela e encerrar o código
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        else:
            print("❌ Nenhuma mão foi detectada. Tente uma imagem mais clara ou com menos poluição no fundo.")