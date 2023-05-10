import pandas as pd
import numpy as np
import logging;logging.basicConfig(level=logging.WARNING)
from sklearn.decomposition import NMF
from matplotlib import pyplot as plt

dataframeA=pd.read_csv('data.csv',index_col=0)
#logging.info(dataframeA)
dataframeB=np.array(dataframeA.fillna(1e-6))
#logging.info(dataframeB)
dataframeB/dataframeB.sum(axis=0)
#logging.info(dataframeB)
dataframeC=dataframeB.transpose()
#logging.info(dataframeC)

nTopics=int(input("How many topics?(2~10)"))
assert 2<=nTopics<=10,"Sorry,the number of topics cannot be out of range [2,10]."

nmf=NMF(n_components=nTopics,beta_loss='kullback-leibler',solver='mu')
nt=nmf.fit_transform(dataframeC)

schoolIDs=pd.read_csv('data.csv')['schoolID']
for i in range(len(schoolIDs)):
    schoolIDs[i]=str(schoolIDs[i])

def plotTopWords(model,featureNames,nTopWords,title):
    fig,axes=plt.subplots(2,5,figsize=(10,5),sharex='all')
    axes=axes.flatten()
    for topic_idx,topic in enumerate(model.components_):
        topFeaturesIndex=topic.argsort()[:-nTopWords-1:-1]
        logging.info(topFeaturesIndex)
        topFeatures=[featureNames[i] for i in topFeaturesIndex]
        weights=topic[topFeaturesIndex]

        ax=axes[topic_idx]
        ax.barh(topFeatures,weights,height=0.7)
        ax.set_title("Topic %s"%(topic_idx+1),fontdict={"fontsize":10})
        ax.invert_yaxis()
        ax.tick_params(axis="both",which="major",labelsize=10)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title,fontsize=10)

    plt.subplots_adjust(top=0.9,bottom=0.05,wspace=0.5,hspace=0.1)
    plt.show()

if __name__ == '__main__':
    plotTopWords(nmf,schoolIDs,8,'Topics of HY248')