# HY248
一个基于HY中学24届8班的人际关系的项目，采用PCA、t-SNE、UMAP、NMF等算法。

## Run|运行
将数据手动写入根目录下的data.csv中。

（0，0）写上schoolID，第一列写上学号，第一行写上不同维度（使用英文），再分别对每个人进行评估。

评估得到的结果可以是浮点数，但必须非负，否则nmf无法分解。

需要下载的库有：
```batch
pip install scikit-learn
pip install matplotlib
pip install numpy
pip install pandas
pip install umap-learn
```

接下来运行各个代码文件即可。

## 原理
详见bilibili：<a href="https://www.bilibili.com/read/cv23323182?spm_id_from=333.999.0.0">专栏：如何通过降维算法对248进行分析</a>