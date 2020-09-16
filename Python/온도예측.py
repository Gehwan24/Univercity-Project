import copy
import warnings
import numpy as np
import seaborn as sns
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import math
from google.colab import drive
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, SimpleRNN, Embedding, Dense, LSTM, Dropout, Flatten, Conv1D, Conv2D, Conv3D, GlobalMaxPooling1D, MaxPooling1D, BatchNormalization, Activation, Bidirectional, TimeDistributed
from keras.models import Sequential
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.models import load_model, Model
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

warnings.filterwarnings(action='ignore')

# 구글 드라이브 마운트
drive.mount('/content/gdrive', force_remount=True)

# custom mse
def mse_AIFrenz(y_true, y_pred):
    diff = abs(y_true - y_pred)
    less_then_one = np.where(diff < 1, 0, diff)
    # multi-column일 경우에도 계산 할 수 있도록 np.average를 한번 더 씌움
    try:
        score = np.average(np.average(less_then_one ** 2, axis = 0))
    except ValueError:
        score = mean_squared_error(y_true, y_pred)
    return score

def mse_keras(y_true, y_pred):
    score = tf.py_function(func=mse_AIFrenz, inp=[y_true, y_pred], Tout=tf.float32,  name='custom_mse')
    return score

def idModulo(data, mod):
    for i in range(0, len(data)):
        data['id'][i] =math.sin((math.pi*(data['id'][i] % mod) )/ mod)
        

def dataScaling(data, index, scaler):
    return pd.DataFrame(scaler.fit_transform(data[index]), columns=index)

def matrixToTensor(data):
    return np.array(data).reshape(data.shape[0], data.shape[1], 1)

#*******************************************************************************
# 변수들
#*******************************************************************************
train_data = pd.read_csv('/content/gdrive/My Drive/train.csv')    # 훈련 데이터
test_data = pd.read_csv('/content/gdrive/My Drive/test.csv')      # 예측할 데이터
AX_Scaler = MinMaxScaler(feature_range=(-1, 1))                     #     X     0 ~ 1 스케일러
AY_Scaler = MinMaxScaler(feature_range=(0, 1))
BX_Scaler = MinMaxScaler(feature_range=(0, 1))
BY18_Scaler = MinMaxScaler(feature_range=(0, 1))
CX_Scaler = MinMaxScaler(feature_range=(0, 1))

x_indexs = [
            ['X00','X07','X28','X31','X32'],      #기온
            ['X01','X06','X22','X27','X29'],      #현지기압
            ['X02','X03','X18','X24','X26'],      #풍속
            ['X04','X10','X21','X36','X39'],      #일일 누적강수량
            ['X05','X08','X09','X23','X33'],      #해면기압
            ['X11','X34'],                        #누적 일사량
            ['X12','X20','X30','X37','X38'],      #습도
            ['X13','X15','X17','X25','X35']       #풍향
]

x_index = ['X00','X01','X02','X03','X04','X05','X06','X07','X08','X09',
           'X10','X11','X12','X13','X14','X15','X16','X17','X18','X19',
           'X20','X21','X22','X23','X24','X25','X26','X27','X28','X29',
           'X30','X31','X32','X33','X34','X35','X36','X37','X38','X39']

y_index = ['Y00','Y01','Y02','Y03','Y04','Y05','Y06', 'Y07','Y08', 'Y09','Y10', 'Y11','Y12', 'Y13','Y14', 'Y15','Y16','Y17'] # Y인덱스

#*******************************************************************************
# Phase1: 데이터 전처리
#*******************************************************************************
# id 주기화

train_data["id"] = np.float32(train_data["id"])
test_data["id"]=np.float32(test_data["id"])

idModulo(train_data, 144)
idModulo(test_data, 144)

train_data[["id"]+x_index] = pd.DataFrame(AX_Scaler.fit_transform(train_data[["id"]+x_index]), columns=["id"]+x_index)
test_data[["id"]+x_index] = pd.DataFrame(AX_Scaler.fit_transform(test_data[["id"]+x_index]), columns=["id"]+x_index)

# 데이터 분류
AX = []
BX = []
CX = []
for i in range(0, len(x_indexs)):
    AX.append(pd.concat([train_data["id"], train_data[x_indexs[i]]], axis=1)[:4320])
    BX.append(pd.concat([train_data["id"], train_data[x_indexs[i]]], axis=1)[4320:])
    CX.append(pd.concat([test_data["id"], test_data[x_indexs[i]]], axis=1))

AY = train_data.loc[:,"Y00" : "Y17"][:4320]
BY18 = train_data["Y18"][4320:]

BX[3].drop(['X04','X36'], axis=1, inplace=True)

# 데이터 reshape
AX_Train = []
for i in range(0, len(x_indexs)):
    AX_Train.append(matrixToTensor(AX[i]))

BX_Train = []
for i in range(0, len(x_indexs)):
    BX_Train.append(matrixToTensor(BX[i]))

CX_Train = []
for i in range(0, len(x_indexs)):
    CX_Train.append(matrixToTensor(CX[i]))

AY_Train = np.array(AY)
BY18_Train = np.array(BY18)

#******************************************************************************
# Phase2: 모델 생성
#*******************************************************************************
inputs = []
layers = []
for i in range(0, len(x_indexs)):
    inputs.append(Input(shape=(None, 1)))
    layers.append(Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(inputs[i]))
    layers[i] = Dropout(0.4)(layers[i])
    layers[i] = Bidirectional(LSTM(64, return_sequences=False, kernel_regularizer=regularizers.l2(0.01)))(layers[i])

outputs = concatenate(layers)
outputs = Dense(len(y_index))(outputs)

model = Model(inputs, outputs)

model.compile(optimizer='adam', loss='mse', metrics=[mse_keras])
hist = model.fit(AX_Train, AY_Train, epochs=100, batch_size=144, validation_split=0.2, callbacks=[EarlyStopping()])

#*******************************************************************************
# Phase3: Y00 ~ Y17 예측값과 Y18의 상관관계
#*******************************************************************************
pre_ans = model.predict(BX_Train)
np_ans = np.array(pre_ans)                                           # 배열 변환
y18 =np.array( pd.DataFrame(BY18_Train))                       # Y18 데이터 배열 변환 
ans = np.concatenate((np_ans, y18),axis = 1)                         # 합치기
print("Y0 ~ Y18 corr")

y0_y18 =  pd.DataFrame(ans).corr()                       
plt.figure(figsize=(10,10))                                           #출력 크기조절
ax = sns.heatmap(y0_y18, cmap = "RdBu", annot = True,vmin=0, vmax=1)  #히트맵 정의
ax.set_ylim(len(y0_y18.columns),0)                                    #y축 잘림 방지
plt.show()                                                            #출력

for l in model.layers[:-1]:
    l.trainable = False

new_output = model.output
new_output = Dense(1)(new_output)

new_model = Model(model.input, new_output)
new_model.compile(optimizer='adam', loss='mse', metrics=[mse_keras])

hist = new_model.fit(BX_Train, BY18_Train, epochs=200, batch_size=144, validation_split=0.2, callbacks=[EarlyStopping()])

#*******************************************************************************
# Phase3: Y00 ~ Y17 예측값과 Y18의 상관관계
#*******************************************************************************
p_ans = new_model.predict(AX_Train)
n_ans = np.array(p_ans)                                           # 배열 변환
ans = np.concatenate((AY,n_ans),axis = 1)                         # 합치기
print("Y0 ~ Y18 corr")

y0_y18 =  pd.DataFrame(ans).corr()                       
plt.figure(figsize=(10,10))                                           #출력 크기조절
ax = sns.heatmap(y0_y18, cmap = "RdBu", annot = True,vmin=0, vmax=1)  #히트맵 정의
ax.set_ylim(len(y0_y18.columns),0)                                    #y축 잘림 방지
plt.show()  


print("3일치 Y18값")
plt.figure(figsize=(27, 4))
plt.plot(BY18_Train)
plt.show()

print("예상 Y18값")
plt.figure(figsize=(27, 4))
predict = new_model.predict(BX_Train)
plt.plot(predict)
plt.show()

plt.figure(figsize=(27, 4))
plt.plot(BY18_Train)
plt.plot(predict)
plt.show()
ans = new_model.predict(CX_Train)

print("80치 예측값")
plt.figure(figsize=(27, 4))
plt.plot(ans)
plt.show()

pd.DataFrame(ans).to_csv('/content/gdrive/My Drive/res11.csv')