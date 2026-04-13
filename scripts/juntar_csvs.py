import os
import csv

pasta_landmarks = "dados/landmarks"
arquivo_final = "dados/coordenadas_libras.csv"

print("Iniciando a união dos arquivos driblando o bloqueio do Windows...")

linhas_totais = 0

# Abre o arquivo final para escrita
with open(arquivo_final, mode='w', newline='', encoding='utf-8') as f_out:
    escritor = csv.writer(f_out)
    
    # Varre a pasta landmarks procurando os arquivos .csv
    for arquivo in os.listdir(pasta_landmarks):
        if arquivo.endswith(".csv"):
            letra = arquivo.replace(".csv", "") 
            caminho_arquivo = os.path.join(pasta_landmarks, arquivo)
            
            # Lê o arquivo atual e escreve no arquivo final
            with open(caminho_arquivo, mode='r', encoding='utf-8') as f_in:
                leitor = csv.reader(f_in)
                linhas_processadas = 0
                
                for linha in leitor:
                    # Insere a letra no comecinho da linha
                    linha.insert(0, letra)
                    escritor.writerow(linha)
                    linhas_processadas += 1
                    linhas_totais += 1
                    
            print(f"Arquivo {arquivo} processado ({linhas_processadas} linhas).")

print("\n" + "="*50)
print("Todos os arquivos foram unidos.")
print(f"O seu novo dataset gigante tem {linhas_totais} linhas prontas para treinamento!")
print("="*50)