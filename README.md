# BioScanner Libras - Reconhecimento em Tempo Real

Este projeto é uma aplicação interativa que utiliza **Visão Computacional** e **Inteligência Artificial** para reconhecer o alfabeto da Língua Brasileira de Sinais (Libras) em tempo real através da webcam. 

O sistema foi construído para ser um jogo educacional, possuindo modos de "Treino" e "Sobrevivência (Recorde)", além de contar com processamento digital de imagens (PDI) para se adaptar a diferentes condições de iluminação.

## Tecnologias Utilizadas
* **Python** (Linguagem principal)
* **OpenCV** (Captura de vídeo e Processamento Digital de Imagens)
* **MediaPipe** (Detecção e extração das coordenadas das mãos)
* **Scikit-Learn / Pandas** (Treinamento do modelo de Machine Learning)
* **Streamlit** (Criação da interface web interativa)

## Como a IA funciona
O modelo foca na detecção de sinais estáticos (letras de A a W, excluindo as que exigem movimento como H, J, K, X, Y, Z). 
A IA não analisa a imagem inteira, mas sim as **coordenadas 3D (X, Y, Z) de 21 pontos da mão** extraídos pelo MediaPipe. Essas coordenadas são normalizadas com base no pulso do usuário, o que significa que o modelo funciona independentemente da distância da mão para a câmera.

Além disso, o algoritmo conta com um sistema inteligente de correção de brilho:
* **Luz ideal:** Lê a imagem normalmente.
* **Sombras:** Aplica filtro CLAHE para equalizar o contraste.
* **Ambiente escuro:** Aplica Correção Gamma para clarear os pixels escuros suavemente.

## Dataset (Kaggle)
As imagens e o arquivo CSV utilizados para treinar este modelo possuem 1.000 variações para cada letra e estão disponíveis publicamente.
**[(https://www.kaggle.com/datasets/gabiru420/brazilian-sign-language-alphabet-landmarks)]**
