from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding
from keras.layers import Conv1D,GlobalAveragePooling1D,MaxPooling1D
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import pandas as pd

data_np=np.array(pd.read_csv('./uci_cad.csv',header='infer',encoding='utf-8'))

X=np.array([line[:-1] for line in data_np])
scale_func=preprocessing.MinMaxScaler(feature_range=(0, 20))
X_scale=scale_func.fit_transform(X)
y=np.array([line[-1] for line in data_np])
x_train,x_test,y_train,y_test=train_test_split(X_scale,y,test_size=0.25,random_state=0)

model=Sequential()
model.add(Conv1D(63,3,activation='relu',input_shape=(13,20)))
model.add(Conv1D(64,3,activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128,3,activation='relu'))
model.add(Conv1D(128,3,activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(5,activation='softmax'))

sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=16,epochs=10)

score=model.evaluate(x_test,y_test,batch_size=16)
