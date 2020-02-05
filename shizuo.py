# -*- coding: utf-8 -*-
#以wine的KLDA降维为原型的python发展版本，基于pandas 
#20/2/4-5
import pandas as pd
import numpy as np
import scipy.linalg
#total = pd.read_table('./wine/wine.txt',header=None)
total = pd.read_table('./wine/wine.txt',header=None, delimiter=',')#以逗号分隔
#新增题目,这不算一行
total.columns = ['type','f1', 'f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','f13']
#样本总数
N = total.shape[0]
#每类样本数
VC = total['type'].value_counts()
N1 = VC[1]
N2 = VC[2]
N3 = VC[3]
#每类样本切片(dtypes为object)
cls1_data = total.iloc[0:N1,1:]#注意从0开始，且不包括N1
cls2_data = total.iloc[N1:N1+N2,1:]
cls3_data = total.iloc[N1+N2:,1:]
#总样本（按竖直方向上合并）
cls_data = pd.concat([cls1_data,cls2_data,cls3_data])
#数据标准化（zscore零均值规范化）
cls1_data = (cls1_data - cls1_data.mean())/cls1_data.std()
cls2_data = (cls2_data - cls2_data.mean())/cls2_data.std()
cls3_data = (cls3_data - cls3_data.mean())/cls3_data.std()
cls_data = (cls_data - cls_data.mean())/cls_data.std()
E_cls1 = cls1_data.mean()
E_cls2 = cls2_data.mean()
E_cls3 = cls3_data.mean()
cov_size = N
##### B
B1 = 1/N1 * np.ones((N1,N1))
B2 = 1/N2 * np.ones((N2,N2))
B3 = 1/N3 * np.ones((N3,N3))
#B = blkdiag(B1,B2,B3);
B = scipy.linalg.block_diag(B1,B2,B3)

##计算核矩阵
K = np.matmul(cls_data,cls_data.T)
pinvK = np.linalg.pinv(K)#伪逆
A = pinvK.dot(pinvK).dot(K).dot(B).dot(K)

##特征值分解
[D,V] = np.linalg.eig(A)#[特征值，特征向量]
V = V.real
D = D.real

#eigValue = D.reshape(N,1)
eigValue = D
##将特征向量按特征值的大到小顺序排序
Yt = np.sort(eigValue)#默认从小到大
Yt = Yt[::-1]#从大到小
index = np.argsort(-eigValue)
#排好序
eigVector = V[:,index]
eigValue = eigValue[index]

D = eigValue
rat1 = D/np.sum(D)
rat2 = np.cumsum(D)/np.sum(D)
# 调出特征值，贡献率，累计贡献率
D = D.reshape((N,1))
rat1 = rat1.reshape((N,1))
rat2 = rat2.reshape((N,1))
#水平合并
content = np.hstack((D,rat1,rat2))
result1 = pd.DataFrame(content,columns=('特征值','贡献率','累计贡献率'))

#主成分载荷
threshold = 0.85
count = sum(rat2<threshold)
#normalization
norm_eigVector = np.sqrt(np.sum(np.multiply(eigVector,eigVector),0))
eigVector = eigVector/np.tile(norm_eigVector,(eigVector.shape[1],1))

# dimension reduction
V = eigVector
data_klda = K.dot(V[:,:count[0]+1])#178x2

# 得到新数据
type3 = np.array(total['type']).reshape(N,1)
newcontent = np.hstack((type3,data_klda))#水平合并
new = pd.DataFrame(newcontent,columns=('type','f1','f2'))

#draw
new1 = data_klda[:N1,:]
new2 = data_klda[N1:N1+N2,:]
new3 = data_klda[N1+N2:,:]

import matplotlib.pyplot as plt
for i in range(N1):
    h1 = plt.scatter(new1[i,0],new1[i,1],c='red')#最开始想到是plt.plot,但是散点图有专门的函数
for i in range(N2):
    h2 = plt.scatter(new2[i,0],new2[i,1],c='blue')
for i in range(N3):
    h3 = plt.scatter(new3[i,0],new3[i,1],c='green')    
plt.legend(handles=[h1,h2,h3], labels=['class1','class2','class3'],loc="upper right")
plt.title('KLDA')
plt.show()
