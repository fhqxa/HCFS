X=[11978 12.5 93.5 31908; 57500 67.6 238.0 15900];
%T=clusterdata(X,0.9) %一次聚类法
%%层次聚类法
%Step1  寻找变量之间的相似性
%用pdist函数计算相似矩阵，有多种方法可以计算距离，进行计算之前最好先将数据用zscore函数进行标准化。
X2=zscore(X);  %标准化数据
Y2=pdist(X2);  %计算距离(默认欧式距离)
%Step2   定义变量之间的连接
Z2=linkage(Y2);
%Step3  评价聚类信息
C2=cophenet(Z2,Y2);       %//0.94698
%Step4 创建聚类，并作出谱系图
T=cluster(Z2,6);
H=dendrogram(Z2)