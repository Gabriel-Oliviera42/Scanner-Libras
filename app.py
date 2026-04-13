import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
import pandas as pd
import os
import time
import random

st.set_page_config(layout="wide")
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

@st.cache_resource
def carregar_modelo():
    try:
        with open('modelo_libras.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        return None

def avaliar_e_tratar_imagem(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    brilho_medio = np.mean(gray)
    
    if brilho_medio > 100:
        metodo_usado = "Normal (luz boa)"
        img_tratada = img_bgr.copy()
    elif brilho_medio > 40:
        metodo_usado = "CLAHE (sombras)"
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        img_tratada = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)
    else:
        metodo_usado = "Gamma (muito escuro)"
        invGamma = 1.0 / 2.2
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        img_tratada = cv2.LUT(img_bgr, table)
        
    return img_tratada, metodo_usado, brilho_medio

def processar_ia(img_bgr, hands_motor, modelo):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resultados = hands_motor.process(img_rgb)
    letra_predita = "_"
    
    if resultados.multi_hand_landmarks:
        mao = resultados.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(img_bgr, mao, mp_hands.HAND_CONNECTIONS)
        
        if modelo is not None:
            pontos_x, pontos_y, pontos_z = [], [], []
            pulso_x, pulso_y, pulso_z = mao.landmark[0].x, mao.landmark[0].y, mao.landmark[0].z
            
            for ponto in mao.landmark:
                pontos_x.append(ponto.x - pulso_x)
                pontos_y.append(ponto.y - pulso_y)
                pontos_z.append(ponto.z - pulso_z)
                
            max_valor = max(max(map(abs, pontos_x)), max(map(abs, pontos_y)), max(map(abs, pontos_z)))
            if max_valor == 0: max_valor = 1.0
                
            linha_dados = []
            for i in range(21):
                linha_dados.extend([pontos_x[i]/max_valor, pontos_y[i]/max_valor, pontos_z[i]/max_valor])
                
            df_dados = pd.DataFrame([linha_dados], columns=[f'{c}{i}' for i in range(21) for c in ['x','y','z']])
            letra_predita = modelo.predict(df_dados)[0]
            
    return img_bgr, letra_predita

def obter_caminho_imagem(letra):
    caminhos_tentativa = [
        os.path.join("dados", "Dataset_Libras", "dataset_libras_images"),
        os.path.join("dado", "dataset_libras_images"),
        os.path.join("dados", "dataset_libras_images")
    ]
    
    for pasta_base in caminhos_tentativa:
        if os.path.exists(pasta_base):
            for ext in ['png', 'jpeg', 'jpg', 'PNG', 'JPEG']:
                caminho = os.path.join(pasta_base, f"Letra {letra} - Libras.{ext}")
                if os.path.exists(caminho):
                    return caminho
    return None

def executar_leitura_continua(cap, hands_motor, modelo, ligar_camera):
    col_video, col_info = st.columns([3, 1])
    with col_video:
        placeholder_img = st.empty()
    with col_info:
        st.subheader("Resultado")
        placeholder_txt = st.empty()
        st.write("Diagnóstico da imagem")
        txt_brilho = st.empty()
        txt_filtro = st.empty()

    try:
        while ligar_camera:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            img_tratada, metodo, brilho = avaliar_e_tratar_imagem(frame)
            img_final, letra = processar_ia(img_tratada, hands_motor, modelo)
            placeholder_img.image(cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            if letra != "_":
                placeholder_txt.success(f"Letra: {letra}")
            else:
                placeholder_txt.info("Aguardando sinal...")
            txt_brilho.metric("Luz média", f"{brilho:.0f}")
            txt_filtro.info(f"PDI: {metodo}")
    finally:
        cap.release()
        hands_motor.close()

def executar_formacao_palavras(cap, hands_motor, modelo, ligar_camera):
    st.sidebar.write("Ajustes de tempo")
    frames_para_travar = st.sidebar.slider("Frames para confirmar letra", 5, 30, 15)

    col_video, col_texto = st.columns([3, 1])
    with col_video:
        placeholder_img = st.empty()
    with col_texto:
        st.subheader("Quadro de texto")
        txt_palavra = st.empty()
        txt_status = st.empty()
        if st.button("Limpar texto"):
            st.session_state['palavra_formada'] = ""
        st.write("Diagnóstico da imagem")
        txt_filtro = st.empty()
    
    if 'palavra_formada' not in st.session_state:
        st.session_state['palavra_formada'] = ""

    letra_rastreada = "_"
    contador_frames = 0
    pausa_digitacao = 0
    frames_sem_mao = 0

    try:
        while ligar_camera:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            img_tratada, metodo, brilho = avaliar_e_tratar_imagem(frame)
            img_final, letra_atual = processar_ia(img_tratada, hands_motor, modelo)
            
            txt_filtro.info(f"PDI: {metodo} (Luz: {brilho:.0f})")
            
            if letra_atual != "_":
                frames_sem_mao = 0
                if pausa_digitacao > 0:
                    pausa_digitacao -= 1
                    txt_status.warning("Aguarde...")
                else:
                    if letra_atual == letra_rastreada:
                        contador_frames += 1
                        txt_status.info(f"Lendo '{letra_atual}': {contador_frames}/{frames_para_travar}")
                        if contador_frames >= frames_para_travar:
                            st.session_state['palavra_formada'] += letra_atual
                            pausa_digitacao = 30
                            contador_frames = 0
                    else:
                        letra_rastreada = letra_atual
                        contador_frames = 1
            else:
                contador_frames = 0
                pausa_digitacao = 0
                frames_sem_mao += 1
                if frames_sem_mao == 40:
                    if len(st.session_state['palavra_formada']) > 0 and st.session_state['palavra_formada'][-1] != " ":
                        st.session_state['palavra_formada'] += " "

            placeholder_img.image(cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB), use_container_width=True)
            texto_tela = st.session_state['palavra_formada'] if st.session_state['palavra_formada'] else "..."
            txt_palavra.markdown(f"<h2 style='color: #4CAF50;'>{texto_tela}</h2>", unsafe_allow_html=True)
    finally:
        cap.release()
        hands_motor.close()

def executar_desafio(cap, hands_motor, modelo, ligar_camera):
    dificuldade = st.sidebar.radio("Modo de Jogo:", ["Treino", "Desafio"])
    
    letras_jogo = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W']
    
    # variaveis que uso no jogo 
    if 'recorde' not in st.session_state: st.session_state['recorde'] = 0
    if 'pontos' not in st.session_state: st.session_state['pontos'] = 0
    if 'alvo' not in st.session_state: st.session_state['alvo'] = random.choice(letras_jogo)
    if 'tempo_limite' not in st.session_state: st.session_state['tempo_limite'] = 10.0

    # função para reiciar o jogo
    def reiniciar_status():
        st.session_state['pontos'] = 0
        st.session_state['tempo_limite'] = 10.0
        st.session_state['alvo'] = random.choice(letras_jogo)

    # zero os pontos se mudar de jogo
    if 'modo_atual' not in st.session_state or st.session_state['modo_atual'] != dificuldade:
        st.session_state['modo_atual'] = dificuldade
        reiniciar_status()

    # codigo do gemini usando o StreamLit para criar o Front
    col_video, col_info = st.columns([3, 1])
    
    with col_video: 
        placeholder_img = st.empty()
        
    with col_info:
        txt_alvo = st.empty()
        txt_status = st.empty()
        txt_pontos = st.empty()
        txt_recorde = st.empty()
        ph_ref = st.empty()

    contador_frames = 0
    inicio_rodada = time.time()

    if dificuldade == "Treino":
        img_path = obter_caminho_imagem(st.session_state['alvo'])
        if img_path: 
            ph_ref.image(img_path, caption="Referência", use_container_width=True)
    else:
        ph_ref.empty()

    try:
        while ligar_camera:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            
            txt_alvo.markdown(f"<h1 style='text-align:center; color:#2196F3; font-size: 80px;'>{st.session_state['alvo']}</h1>", unsafe_allow_html=True)
            txt_pontos.metric("Sua Pontuação", st.session_state['pontos'])
            
            if dificuldade == "Sobrevivência (Recorde)":
                txt_recorde.metric("🏆 Recorde", max(st.session_state['recorde'], st.session_state['pontos']))
            else:
                txt_recorde.empty()

            img_tratada, _, _ = avaliar_e_tratar_imagem(frame)
            img_final, letra_detectada = processar_ia(img_tratada, hands_motor, modelo)

            if dificuldade == "Sobrevivência (Recorde)":
                tempo_decorrido = time.time() - inicio_rodada
                tempo_restante = st.session_state['tempo_limite'] - tempo_decorrido
                
                cor_texto = (0, 0, 255) if tempo_restante < 3.0 else (0, 255, 255) 
                cv2.putText(img_final, f"Tempo: {max(0, tempo_restante):.1f}s", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, cor_texto, 3, cv2.LINE_AA)
                
                # REINÍCIO AUTOMÁTICO SE O TEMPO ZERAR
                if tempo_restante <= 0:
                    if st.session_state['pontos'] > st.session_state['recorde']:
                        st.session_state['recorde'] = st.session_state['pontos']
                    
                    st.session_state['pontos'] = 0
                    st.session_state['tempo_limite'] = 10.0
                    st.session_state['alvo'] = random.choice(letras_jogo)
                    inicio_rodada = time.time() # Reseta o relógio
                    contador_frames = 0
                    txt_status.empty()
                    continue # Pula para a próxima leitura da câmera sem travar

            if letra_detectada == st.session_state['alvo']:
                contador_frames += 1
                txt_status.success("Isso! Segure...")
                
                cv2.rectangle(img_final, (0,0), (int((contador_frames/10)*frame.shape[1]), 15), (0,255,0), -1)
                
                if contador_frames >= 10: 
                    st.session_state['pontos'] += 1
                    
                    if dificuldade == "Sobrevivência (Recorde)":
                        st.session_state['tempo_limite'] = max(2.0, st.session_state['tempo_limite'] * 0.9)
                    
                    nova_letra = random.choice(letras_jogo)
                    while nova_letra == st.session_state['alvo']:
                        nova_letra = random.choice(letras_jogo)
                        
                    st.session_state['alvo'] = nova_letra
                    contador_frames = 0
                    inicio_rodada = time.time()
                    txt_status.empty()
                    
                    if dificuldade == "Treino (Aprender)":
                        img_path = obter_caminho_imagem(st.session_state['alvo'])
                        if img_path: 
                            ph_ref.image(img_path, caption="Referência", use_container_width=True)
                        else:
                            ph_ref.empty()

            else:
                contador_frames = 0
                txt_status.info("Faça o sinal correto...")

            placeholder_img.image(cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB), use_container_width=True)
    finally:
        cap.release()
        hands_motor.close()

def modo_tradutor_libras():
    st.title("Tradução de Libras")
    
    sub_modo = st.radio(
        "Selecione o modo de tradução:", 
        ["Leitura contínua", "Formação de palavras", "Desafio"]
    )
    
    modelo = carregar_modelo()
    if modelo is None:
        st.error("Modelo não encontrado.")
        return

    ligar_camera = st.checkbox("Ligar câmera", key="cam_tradutor")

    if ligar_camera:
        cap = cv2.VideoCapture(0)
        hands_motor = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

        if sub_modo == "Leitura contínua":
            executar_leitura_continua(cap, hands_motor, modelo, ligar_camera)
        elif sub_modo == "Formação de palavras":
            executar_formacao_palavras(cap, hands_motor, modelo, ligar_camera)
        elif sub_modo == "Desafio":
            executar_desafio(cap, hands_motor, modelo, ligar_camera)

def aplicar_filtro_media(img_rgb):
    st.subheader("Filtro da Média")
    kernel_size = st.sidebar.slider("Tamanho do Kernel (Media)", 3, 31, 5, step=2)
    img_result = cv2.blur(img_rgb, (kernel_size, kernel_size))
    st.image(img_result, caption=f"Média ({kernel_size}x{kernel_size})", use_container_width=True)

def aplicar_filtro_gaussiano(img_rgb):
    st.subheader("Filtro Gaussiano")
    kernel_size = st.sidebar.slider("Tamanho do Kernel (Gaussiano)", 3, 31, 5, step=2)
    img_result = cv2.GaussianBlur(img_rgb, (kernel_size, kernel_size), 0)
    st.image(img_result, caption=f"Gaussiano ({kernel_size}x{kernel_size})", use_container_width=True)

def aplicar_filtro_mediana(img_rgb):
    st.subheader("Filtro da Mediana")
    kernel_size = st.sidebar.slider("Tamanho do Kernel (Mediana)", 3, 31, 5, step=2)
    img_result = cv2.medianBlur(img_rgb, kernel_size)
    st.image(img_result, caption=f"Mediana (k={kernel_size})", use_container_width=True)

def aplicar_filtro_sobel(img_gray):
    st.subheader("Filtros de Borda (Sobel)")
    ksize = st.sidebar.slider("Tamanho do Kernel Sobel", 1, 7, 3, step=2)
    tipo_sobel = st.sidebar.radio("Qual direção do Sobel?", ["Sobel X (Vertical)", "Sobel Y (Horizontal)", "Sobel Total (Magnitude)"])
    
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    
    sobelx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=ksize)
    
    if tipo_sobel == "Sobel X (Vertical)":
        img_result = cv2.convertScaleAbs(sobelx)
    elif tipo_sobel == "Sobel Y (Horizontal)":
        img_result = cv2.convertScaleAbs(sobely)
    else:
        sobel_mag = cv2.magnitude(sobelx, sobely)
        img_result = cv2.convertScaleAbs(sobel_mag)
        
    st.image(img_result, caption=tipo_sobel, use_container_width=True)

def aplicar_filtro_laplaciano(img_gray):
    st.subheader("Filtro de Borda (Laplaciano)")
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    laplacian = cv2.Laplacian(img_blur, cv2.CV_64F)
    img_result = cv2.convertScaleAbs(laplacian)
    st.image(img_result, caption="Laplaciano", use_container_width=True)

def aplicar_abertura(img_gray):
    st.subheader("Operação de Abertura (Remove ruídos externos)")
    limiar = st.sidebar.slider("Limiar de Binarização", 0, 255, 127)
    kernel_size = st.sidebar.slider("Tamanho do Elemento Estruturante", 3, 21, 5, step=2)
    
    _, img_thresh = cv2.threshold(img_gray, limiar, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    img_result = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_thresh, caption="Binarização", use_container_width=True)
    with col2:
        st.image(img_result, caption="Abertura", use_container_width=True)

def aplicar_fechamento(img_gray):
    st.subheader("Operação de Fechamento (Preenche buracos internos)")
    limiar = st.sidebar.slider("Limiar de Binarização", 0, 255, 127)
    kernel_size = st.sidebar.slider("Tamanho do Elemento Estruturante", 3, 21, 5, step=2)
    
    _, img_thresh = cv2.threshold(img_gray, limiar, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    img_result = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_thresh, caption="Binarização", use_container_width=True)
    with col2:
        st.image(img_result, caption="Fechamento", use_container_width=True)

def aplicar_equalizacao(img_gray):
    st.subheader("Equalização de Histograma")
    img_eq = cv2.equalizeHist(img_gray)
    st.image(img_eq, caption="Histograma Equalizado", use_container_width=True)

def aplicar_operacoes_entre_imagens(img_rgb):
    st.subheader("Operações entre Imagens")
    arquivo2 = st.file_uploader("Envie a SEGUNDA imagem", type=["jpg", "png", "jpeg"])
    
    if arquivo2 is not None:
        file_bytes2 = np.asarray(bytearray(arquivo2.read()), dtype=np.uint8)
        img2 = cv2.imdecode(file_bytes2, 1)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        img2_resized = cv2.resize(img2_rgb, (img_rgb.shape[1], img_rgb.shape[0]))
        
        operacao = st.sidebar.radio("Tipo de Operação", ["Soma", "Subtração", "Diferença Absoluta"])
        
        if operacao == "Soma":
            peso = st.sidebar.slider("Peso da Imagem 1 (Transparência)", 0.0, 1.0, 0.5)
            img_result = cv2.addWeighted(img_rgb, peso, img2_resized, 1.0 - peso, 0)
        elif operacao == "Subtração":
            img_result = cv2.subtract(img_rgb, img2_resized)
        elif operacao == "Diferença Absoluta":
            img_result = cv2.absdiff(img_rgb, img2_resized)
            
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img_rgb, caption="Imagem 1", use_container_width=True)
        with col2:
            st.image(img2_resized, caption="Imagem 2", use_container_width=True)
        with col3:
            st.image(img_result, caption=f"Resultado: {operacao}", use_container_width=True)

def modo_laboratorio_pdi():
    st.title("Laboratório PDI")
    st.write("Selecione a técnica no menu lateral e ajuste os parâmetros.")
    
    arquivo = st.file_uploader("Envie a imagem principal para análise", type=["jpg", "png", "jpeg"])
    
    if arquivo is not None:
        file_bytes = np.asarray(bytearray(arquivo.read()), dtype=np.uint8)
        img_original = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
        
        st.sidebar.subheader("Controles PDI")
        filtro_selecionado = st.sidebar.selectbox(
            "Escolha a operação:", 
            [
                "Filtro da Média",
                "Filtro Gaussiano", 
                "Filtro da Mediana",
                "Filtros de Borda (Sobel)", 
                "Filtros de Borda (Laplaciano)", 
                "Operação de Abertura",
                "Operação de Fechamento",
                "Equalização de Histograma", 
                "Operações entre Imagens" 
            ]
        )
        
        if filtro_selecionado == "Filtro da Média":
            aplicar_filtro_media(img_rgb)
        elif filtro_selecionado == "Filtro Gaussiano":
            aplicar_filtro_gaussiano(img_rgb)
        elif filtro_selecionado == "Filtro da Mediana":
            aplicar_filtro_mediana(img_rgb)
        elif filtro_selecionado == "Filtros de Borda (Sobel)":
            aplicar_filtro_sobel(img_gray)
        elif filtro_selecionado == "Filtros de Borda (Laplaciano)":
            aplicar_filtro_laplaciano(img_gray)
        elif filtro_selecionado == "Operação de Abertura":
            aplicar_abertura(img_gray)
        elif filtro_selecionado == "Operação de Fechamento":
            aplicar_fechamento(img_gray)
        elif filtro_selecionado == "Equalização de Histograma":
            aplicar_equalizacao(img_gray)
        elif filtro_selecionado == "Operações entre Imagens":
            aplicar_operacoes_entre_imagens(img_rgb)

def main():
    st.sidebar.title("Navegação")
    modo = st.sidebar.radio(
        "Selecione o módulo:", 
        [
            "Tradutor de Libras", 
            "Laboratório PDI"
        ]
    )
    
    if modo == "Tradutor de Libras":
        modo_tradutor_libras()
    elif modo == "Laboratório PDI":
        modo_laboratorio_pdi()

if __name__ == "__main__":
    main()