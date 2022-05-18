import pandas as pd 
import numpy as np

from module import utils
from tensorflow.keras.models import Model, load_model


model, avg, std = utils.load_trained_model("my_model_20211015")

# csv 에서 pandas dataframe으로 변환
#healthindex = pd.read_csv("inference_healthindex.csv")

# dataframe에서 numpy array의 데이터로 변환 하여 test 준비
#X = utils.dataframe_to_test_data(healthindex)


# CSV 파일에서 데이터를 읽어와 numpy array의 데이터로 변환 하여 모델 Test 준비
X = utils.csv_to_test_data("inference_healthindex.csv")

# 입력 데이터 z score normalization
X = utils.z_score_normalization(X, avg, std)

# Inference
labels, confidences = utils.inference(model, X)

print (labels)
print (confidences)