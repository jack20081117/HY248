import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import umap
plt.rcParams['font.family']=['Microsoft YaHei']

schoolIDs=pd.read_csv('data.csv')['schoolID']
dataframe=pd.read_csv('data.csv',index_col=0)

algorithm=input('Use T-SNE or PCA or UMAP? t/p/u')
if algorithm=='t':
    tsne=TSNE(n_components=2,learning_rate=300,init='pca',perplexity=5)
    nodePos=tsne.fit_transform(dataframe)
elif algorithm=='p':
    pca=PCA(n_components=2,whiten=True)
    nodePos=pca.fit_transform(dataframe)
else:
    down=umap.UMAP(n_components=2,n_neighbors=5)
    nodePos=down.fit_transform(dataframe)

xs,ys=[],[]

for i in range(49):
    xs.append(nodePos[i,0])
    ys.append(nodePos[i,1])

length=len(schoolIDs)
for i in range(length):
    plt.text(xs[i],ys[i],schoolIDs[i],fontsize=8)

plt.scatter(xs,ys,s=10,marker='D')

plt.title('T-SNE' if algorithm=='t' else 'UMAP' if algorithm=='u' else 'PCA')
plt.grid()
plt.show()

if __name__ == '__main__':
    pass