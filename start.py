import pandas as pd
from sklearn import datasets,linear_model,metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
df=pd.read_csv('youtube_dislike_dataset.csv')


df=df[['view_count','likes','dislikes','comment_count']]

# removing outliers
qv1=df['view_count'].quantile(0.25)
qv3=df['view_count'].quantile(0.75)
IQRv=qv3-qv1
lov=qv1-1.5*IQRv
upv=qv3+1.5*IQRv
upv_ar=np.where(df['view_count']>=upv)[0]
lov_ar=np.where(df['view_count']<=lov)[0]

df.drop(index=upv_ar,inplace=True)
df.drop(index=lov_ar,inplace=True)


x=np.array(df['view_count'])
y=np.array(df['likes'])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=1)

n=np.size(x_train)

m_x=np.mean(x_train)
m_y=np.mean(y_train)

ss_xy=np.sum(y_train*x_train)-n*m_y*m_x
ss_xx=np.sum(x_train*x_train)-n*m_x*m_x

b_1 = ss_xy/ss_xx
b_0 = m_y - b_1*m_x

xgl=x
ygl=b_0+b_1*xgl
# plt.plot(xgl,ygl,color='green')
# plt.scatter(x_train,y_train,color='blue')
# plt.show()
# plt.plot(xgl,ygl,color='green')
# plt.scatter(x_test,y_test,color='blue')
# plt.show()

y_pred=b_0+b_1*x_test
m=np.sum(abs(y_test-y_pred))
mae=m/np.size(x)
print(mae)
plt.plot(xgl,ygl,color='green')
plt.scatter(x_train,y_train,color='red')
plt.plot(xgl,ygl,color='green')
plt.scatter(x_test,y_test,color='blue')
plt.show()