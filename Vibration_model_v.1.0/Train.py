import pandas as pd 
import numpy as np

from module import utils

# Parameter : Tx -- Time_step, 5초 간격으로 Train 혹은 Prediction, n_feature -- Health Index 특징 개수
Tx = 5
n_feature = 6
batch_size = 32
epochs = 1000

# csv 에서 pandas dataframe으로 변환
#healthindex = pd.read_csv("healthindex.csv")

# dataframe에서 Training numpy array의 데이터로 변환 하여 모델 Training 준비
#X, Y = utils.dataframe_to_train_data(healthindex)

# CSV 파일에서 데이터를 읽어와 Training numpy array의 데이터로 변환 하여 모델 Training 준비
X, Y = utils.csv_to_train_data("healthindex.csv")

# 입력 데이터를 Feature 축을 기준으로 평균과 표준편차를 계산
avg = utils.get_avg(X)
std = utils.get_std(X)

# 입력 데이터 z score normalization
X = utils.z_score_normalization(X, avg, std)

# 이상 진동 감지 모델 load
model = utils.get_vibmodel((Tx,n_feature))

# Train
history = utils.train_model(model, X, Y, batch_size, epochs)

# history log text 생성
texts = utils.get_train_history(history)

# 훈련된 모델 .h5 형태로 저장
utils.save_model(model, avg, std, "my_model_20211015")

for text in texts:
    print (text)