from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Caminho do arquivo
path = 'C:/Users/SDE/Desktop/Codes/Python/Segmentação de Parceiro/rfm.xlsx'
df = pd.read_excel(path)

# Removendo a coluna de ID, se necessário
df = df.iloc[:, 1:]  # Ajuste conforme a estrutura do seu arquivo

# Escalando os dados
scaler = StandardScaler()
X_escalado = scaler.fit_transform(df)

# Avaliando a pontuação silhouette para diferentes números de clusters
resultados_silhouette = {}

for n_clusters in range(2, 7):  # Testando de 2 até 6 clusters
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, random_state=42)
    labels = kmeans.fit_predict(X_escalado)
    silhouette_avg = silhouette_score(X_escalado, labels)
    resultados_silhouette[n_clusters] = silhouette_avg

# Encontrando o número ideal de clusters
melhor_n_clusters = max(resultados_silhouette, key=resultados_silhouette.get)
melhor_silhouette = resultados_silhouette[melhor_n_clusters]
print(f"Melhor número de clusters: {melhor_n_clusters}")
print(f"Pontuação silhouette: {melhor_silhouette:.4f}")

# Ajustando o modelo final com o melhor número de clusters
kmeans_final = KMeans(n_clusters=4, init='k-means++', max_iter=100, random_state=42)
df['Cluster'] = kmeans_final.fit_predict(X_escalado)


# Visualizando os clusters nas variáveis originais (apenas duas variáveis para exemplo)
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=df, x=df.columns[0], y=df.columns[1], hue='Cluster', palette='Set2', s=100
)
plt.title('Clusters Visualizados nas Variáveis Originais')
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.legend(title='Cluster')
plt.show()

plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=df, x=df.columns[0], y=df.columns[2], hue='Cluster', palette='Set2', s=100
)
plt.title('Clusters Visualizados nas Variáveis Originais')
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[2])
plt.legend(title='Cluster')
plt.show()


plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=df, x=df.columns[1], y=df.columns[2], hue='Cluster', palette='Set2', s=100
)
plt.title('Clusters Visualizados nas Variáveis Originais')
plt.xlabel(df.columns[1])
plt.ylabel(df.columns[2])
plt.legend(title='Cluster')
plt.show()


# Visualizando a pontuação silhouette por número de clusters
plt.figure(figsize=(8, 5))
plt.plot(list(resultados_silhouette.keys()), list(resultados_silhouette.values()), marker='o')
plt.title('Pontuação Silhouette por Número de Clusters')
plt.xlabel('Número de Clusters')
plt.ylabel('Pontuação Silhouette')
plt.axvline(x=melhor_n_clusters, color='red', linestyle='--', label='Melhor Número de Clusters')
plt.legend()
plt.show()