import pandas as pd
import numpy as np
import umap
from matplotlib import pyplot as plt

plt.rcParams['font.family']=['Microsoft YaHei']

schoolIDs=pd.read_csv('data.csv')['schoolID']
dataframe=pd.read_csv('data.csv',index_col=0)

sphere_mapper=umap.UMAP(output_metric='haversine',random_state=42,n_neighbors=5).fit(dataframe)

xs=np.sin(sphere_mapper.embedding_[:,0])*np.cos(sphere_mapper.embedding_[:,1])
ys=np.sin(sphere_mapper.embedding_[:,0])*np.sin(sphere_mapper.embedding_[:,1])
zs=np.cos(sphere_mapper.embedding_[:,0])

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(xs,ys,zs,cmap='Spectral')

for i in range(len(schoolIDs)):
    ax.text(xs[i],ys[i],zs[i],s=schoolIDs[i],fontsize=8)

plt.show()

if __name__=='__main__':
    pass
