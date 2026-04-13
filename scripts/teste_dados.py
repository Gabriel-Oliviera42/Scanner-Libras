import pandas as pd
colunas = ['Letra']
for i in range(21):
    colunas.extend([f'x{i}', f'y{i}', f'z{i}'])

df = pd.read_csv("dados/coordenadas_libras.csv", names=colunas)

print("Mínimos (A borda esquerda/topo da mão):")
print(df[['x0', 'x12', 'y0', 'y12']].min())

print("\nMáximos (A borda direita/baixo da mão):")
print(df[['x0', 'x12', 'y0', 'y12']].max())