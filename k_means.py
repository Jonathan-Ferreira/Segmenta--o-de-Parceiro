from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = 'C:/Users/SDE/Desktop/Codes/Python/Segmentação de Parceiro/rfm.xlsx'
df = pd.read_excel(path)

df = df.iloc[:, 1:]

print(df)


# label_encoder = LabelEncoder()
# df['status_type']= label_encoder.fit_transform(df['status_type'])
# df['status_published']= label_encoder.fit_transform(df['status_published'])


minmax_scaled = MinMaxScaler(feature_range=(0, 1))
X_minmax = minmax_scaled.fit_transform(df)


cs = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_minmax)
    cs.append(kmeans.inertia_)

# Plotagem do gráfico do método do cotovelo
plt.plot(range(1, 11), cs)
plt.title('O Método do Cotovelo')
plt.xlabel('Número de Clusters')
plt.ylabel('Inércia (CS)')

# Encontre o "cotovelo" através da derivada segunda (segunda diferença)
segunda_diferenca = np.diff(np.diff(cs))
melhor_n_clusters = segunda_diferenca.argmax() + 2  # +2 porque diff reduz um elemento

# Marcar o ponto do "cotovelo" no gráfico
plt.axvline(x=melhor_n_clusters, color='red', linestyle='--', label='Melhor Número de Clusters')

# Ajustando o K-Means com o melhor número de clusters (definido pelo método do cotovelo)
kmeans_minmax = KMeans(n_clusters=melhor_n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans_minmax.fit(X_minmax)

# Adicionando os rótulos dos clusters ao DataFrame
df['Cluster'] = kmeans_minmax.labels_

# Redução de dimensionalidade para visualização (2D) usando PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_minmax)

# Obtendo os nomes das colunas originais
feature_names = df.columns[:-1]  # Exclui a coluna de cluster, se houver

# Criando um DataFrame com os loadings
loadings = pd.DataFrame(
    pca.components_.T,
    columns=['Componente 1', 'Componente 2'],
    index=feature_names
)

print("Contribuições das variáveis para cada componente principal:")
print(loadings)

# Criando um DataFrame com os componentes principais
df_pca = pd.DataFrame(data=X_pca, columns=['Componente 1', 'Componente 2'])

# Renomeando as colunas para algo mais interpretável, se necessário
df_pca.columns = ['Cluster por RFM', 'Cluster por Frequência']

# Adicionando os rótulos dos clusters
df_pca['Cluster'] = kmeans_minmax.labels_

# Visualizando os clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df_pca, x='Cluster por RFM', y='Cluster por Frequência', hue='Cluster', palette='Set2', s=100)
plt.title('Clusters Visualizados em 2D')
plt.xlabel('Cluster por RFM')
plt.ylabel('Cluster por Frequência')
plt.legend(title='Cluster')
plt.show()










# plt.legend()
# plt.show()


# cs = []

# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
#     kmeans.fit(X_minmax)
#     cs.append(kmeans.inertia_)

# # Plotagem do gráfico do método do cotovelo
# plt.plot(range(1, 11), cs)
# plt.title('O Método do Cotovelo')
# plt.xlabel('Número de Clusters')
# plt.ylabel('Inércia (CS)')

# # Encontre o "cotovelo" através da derivada segunda (segunda diferença)
# segunda_diferenca = np.diff(np.diff(cs))
# melhor_n_clusters = segunda_diferenca.argmax() + 2  # +2 porque diff reduz um elemento

# # Marcar o ponto do "cotovelo" no gráfico
# plt.axvline(x=melhor_n_clusters, color='red', linestyle='--', label='Melhor Número de Clusters')

# plt.legend()
# plt.show()

