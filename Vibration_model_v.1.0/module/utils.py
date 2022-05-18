import pandas as pd 
import numpy as np
import os
import csv

from tensorflow.keras.layers import Dense, Input, TimeDistributed, LSTM, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

def csv_to_train_data(filename, segment_sec = 5):
    """
    진동센서에서 받아진 HealthIndex 데이터를 5초 기준으로 하나의 훈련 데이터로 만드는 함수
    연속된 5초의 데이터가 같은 라벨(정상 혹은 이상진동)일 때, 훈련 데이터로 생성
    
    Argument:
    filename -- 저장된 csv 파일 이름
    
    Returns:
    X -- 5초의 Health index 데이터
    Y -- X와 매핑된 Label
    """
    
    rms                = []
    shape_indcator     = []
    variance           = []
    standard_deviation = []
    srav               = []
    amav               = []
    label              = []

    with open(filename, 'r') as file:
        csv_reader = csv.DictReader(file)

        for csv_data in csv_reader:
            for key, val in csv_data.items():
                if key == "RMS":
                    rms.append(float(val))
                elif key == "ShapeIndicator":
                    shape_indcator.append(float(val))
                elif key == "Variance":
                    variance.append(float(val))
                elif key == "StandardDeviation":
                    standard_deviation.append(float(val))
                elif key == "SRAV":
                    srav.append(float(val))
                elif key == "AMAV":
                    amav.append(float(val))
                elif key == "Label":
                    label.append(float(val))
    
    
    # Health Index 
    rms                = np.expand_dims(rms, axis = 1)
    shape_indcator     = np.expand_dims(shape_indcator, axis = 1)
    variance           = np.expand_dims(variance, axis = 1)
    standard_deviation = np.expand_dims(standard_deviation, axis = 1)
    srav               = np.expand_dims(srav, axis = 1)
    amav               = np.expand_dims(amav, axis = 1)
    
    # Label
    label              = np.expand_dims(label, axis = 1)
    
    # Health Index concat
    concat_data        = np.concatenate((rms, shape_indcator, 
                                        variance, standard_deviation, 
                                        srav, amav), axis =1)
    
    # segment_sec를 간격으로 하나의 데이터 세트로 생성
    X = np.array([ concat_data[i:i+5] for i in range(0, len(concat_data) - segment_sec, segment_sec) ])
    
    # segment_sec를 간격으로  하나의 Label을 생성. // 0 혹은 1이 아닌 경우 Positive와 Negative가 섞인 데이터기 때문에 삭제 필요
    Y = np.expand_dims( [ np.mean(label[i:i+5])  for i in range(0, len(concat_data) - segment_sec, segment_sec) ], axis = 1)
    
    # Label이 0 혹은 1이 아닌 경우 삭제
    del_list = []
    for idx in range(len(Y)):
        if Y[idx] != 0.0 and Y[idx] != 1.0:
            del_list.append(idx)

    for i in range(len(del_list)-1, -1, -1):
        Y = np.delete(Y, del_list[i], axis = 0)
        X = np.delete(X, del_list[i], axis = 0)
        
    # X와 Y의 위치를 유지하며 random으로 섞기
    idxs = np.arange(X.shape[0])
    np.random.shuffle(idxs)

    Y = Y[idxs]
    X = X[idxs]
    
    return X, Y

def csv_to_test_data(filename, segment_sec = 5):
    """
    진동센서에서 받아진 HealthIndex 데이터를 5초 기준으로 하나의 테스트 데이터로 만드는 함수
    
    Argument:
    filename -- 저장된 csv 파일 이름
    
    Returns:
    X -- 5초의 Health index 데이터
    """
    
    rms                = []
    shape_indcator     = []
    variance           = []
    standard_deviation = []
    srav               = []
    amav               = []

    with open(filename, 'r') as file:
        csv_reader = csv.DictReader(file)

        for csv_data in csv_reader:
            for key, val in csv_data.items():
                if key == "RMS":
                    rms.append(float(val))
                elif key == "ShapeIndicator":
                    shape_indcator.append(float(val))
                elif key == "Variance":
                    variance.append(float(val))
                elif key == "StandardDeviation":
                    standard_deviation.append(float(val))
                elif key == "SRAV":
                    srav.append(float(val))
                elif key == "AMAV":
                    amav.append(float(val))    
    
    # Health Index 
    rms                = np.expand_dims(rms, axis = 1)
    shape_indcator     = np.expand_dims(shape_indcator, axis = 1)
    variance           = np.expand_dims(variance, axis = 1)
    standard_deviation = np.expand_dims(standard_deviation, axis = 1)
    srav               = np.expand_dims(srav, axis = 1)
    amav               = np.expand_dims(amav, axis = 1)    
    
    # Health Index concat
    concat_data        = np.concatenate((rms, shape_indcator, 
                                        variance, standard_deviation, 
                                        srav, amav), axis =1)
    
    # segment_sec를 간격으로 하나의 데이터 세트로 생성
    X = np.array([ concat_data[i:i+5] for i in range(0, len(concat_data) - segment_sec, segment_sec) ])
    
    return X

# def dataframe_to_train_data(df, segment_sec = 5):
#     """
#     진동센서에서 받아진 HealthIndex 데이터를 5초 기준으로 하나의 훈련 데이터로 만드는 함수
#     연속된 5초의 데이터가 같은 라벨(정상 혹은 이상진동)일 때, 훈련 데이터로 생성
    
#     Argument:
#     df -- 시간 순으로 저장된 health index를 저장한 dataframe
    
#     Returns:
#     X -- 5초의 Health index 데이터
#     Y -- X와 매핑된 Label
#     """
    
#     # Health Index 
#     rms                = np.expand_dims(df['RMS'].to_list(), axis = 1)
#     shape_indcator     = np.expand_dims(df['ShapeIndicator'].to_list(), axis = 1)
#     variance           = np.expand_dims(df['Variance'].to_list(), axis = 1)
#     standard_deviation = np.expand_dims(df['StandardDeviation'].to_list(), axis = 1)
#     srav               = np.expand_dims(df['SRAV'].to_list(), axis = 1)
#     amav               = np.expand_dims(df['AMAV'].to_list(), axis = 1)
    
#     # Label
#     label              = np.expand_dims(df['Label'].to_list(), axis = 1)
    
#     # Health Index concat
#     concat_data        = np.concatenate((rms, shape_indcator, 
#                                         variance, standard_deviation, 
#                                         srav, amav), axis =1)
    
#     # segment_sec를 간격으로 하나의 데이터 세트로 생성
#     X = np.array([ concat_data[i:i+5] for i in range(0, len(concat_data) - segment_sec, segment_sec) ])
    
#     # segment_sec를 간격으로  하나의 Label을 생성. // 0 혹은 1이 아닌 경우 Positive와 Negative가 섞인 데이터기 때문에 삭제 필요
#     Y = np.expand_dims( [ np.mean(label[i:i+5])  for i in range(0, len(concat_data) - segment_sec, segment_sec) ], axis = 1)
    
#     # Label이 0 혹은 1이 아닌 경우 삭제
#     del_list = []
#     for idx in range(len(Y)):
#         if Y[idx] != 0.0 and Y[idx] != 1.0:
#             del_list.append(idx)

#     for i in range(len(del_list)-1, -1, -1):
#         Y = np.delete(Y, del_list[i], axis = 0)
#         X = np.delete(X, del_list[i], axis = 0)
        
#     # X와 Y의 위치를 유지하며 random으로 섞기
#     idxs = np.arange(X.shape[0])
#     np.random.shuffle(idxs)

#     Y = Y[idxs]
#     X = X[idxs]
    
#     return X, Y

# def dataframe_to_test_data(df, segment_sec = 5):
#     """
#     진동센서에서 받아진 HealthIndex 데이터를 5초 기준으로 하나의 테스트 데이터로 만드는 함수
    
#     Argument:
#     df -- 시간 순으로 저장된 health index를 저장한 dataframe
    
#     Returns:
#     X -- 5초의 Health index 데이터
#     """
    
#     # Health Index 
#     rms                = np.expand_dims(df['RMS'].to_list(), axis = 1)
#     shape_indcator     = np.expand_dims(df['ShapeIndicator'].to_list(), axis = 1)
#     variance           = np.expand_dims(df['Variance'].to_list(), axis = 1)
#     standard_deviation = np.expand_dims(df['StandardDeviation'].to_list(), axis = 1)
#     srav               = np.expand_dims(df['SRAV'].to_list(), axis = 1)
#     amav               = np.expand_dims(df['AMAV'].to_list(), axis = 1)
    
    
#     # Health Index concat
#     concat_data        = np.concatenate((rms, shape_indcator, 
#                                         variance, standard_deviation, 
#                                         srav, amav), axis =1)
    
#     # segment_sec를 간격으로 하나의 데이터 세트로 생성
#     X = np.array([ concat_data[i:i+5] for i in range(0, len(concat_data) - segment_sec, segment_sec) ])
    
#     return X

def get_avg(X):
    """
    입력 데이터의 Feature를 기준으로 평균을 계산하는 함수
    
    Argument:
    X -- 입력 데이터
    
    returns: 
    avg -- feature 축을 기준으로 한 평균 벡터
    """
    shape = X.shape
    reshape_X = np.reshape(X,(shape[0] * shape[1] ,shape[2]))

    avg = np.mean(reshape_X, axis=0)
    
    return avg

def get_std(X):
    """
    입력 데이터의 Feature를 기준으로 표준편차를 계산하는 함수
    
    Argument:
    X -- 입력 데이터
    
    returns: 
    std -- feature 축을 기준으로 한 표준편차 벡터
    """
    shape = X.shape
    reshape_X = np.reshape(X,(shape[0] * shape[1] ,shape[2]))

    std = np.std(reshape_X, axis=0)
    
    return std

def z_score_normalization(X, avg, std):
    """
    입력 데이터를 z score normalization을 수행하는 함수 
    
    * 데이터 분포에 따른 표준정규분포 정규화
    
    # z score normalization
    X = (X - mu) / sigma
    
    Argument:
    X -- 입력 데이터
    avg -- 입력 데이터의 feature 축을 기준으로 한 평균
    std -- 입력 데이터의 feature 축을 기준으로 한 표준 편차
    
    returns:
    X -- z score normalization이 적용된 입력 데이터
    """
    X = (X - avg) / std
    
    return X
    
def get_vibmodel(input_shape):
    """
    Keras의 model graph를 생성해주는 함수
    
    Argument:
    input_shape -- 모델 입력 데이터의 shape
    
    Returns:
    model -- Keras model instance  
    """
    
    X_input = Input(shape = (input_shape))
    X = TimeDistributed(Dense(20, activation = "relu"))(X_input)
    X = BatchNormalization()(X)
    X = LSTM(30, recurrent_dropout = 0.5, return_sequences = True)(X)
    X = LSTM(20, recurrent_dropout = 0.5)(X)
    X = Dense(10, activation = "relu")(X)
    X = Dense(1, activation = "sigmoid")(X)
    
    model = Model(inputs = X_input, outputs = X)
    
    opt = Adam(lr=1e-6, beta_1=0.9, beta_2=0.999)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
    
    return model

def train_model(model, X, Y, batch_size, epochs):   
    """
    model 훈련함수 함수
    
    Argument:
    X -- (N, Tx, n_feature) shape 의 입력 데이터
    Y -- X에 대한 Label 데이터
    batch_size -- 훈련 Batch size
    epochs -- 훈련 반복 횟수
    
    Returns:
    history -- 훈련 history 반환
    """
    
    history = model.fit(X, Y, batch_size = batch_size, epochs=epochs)
    
    return history
    
def save_model(model, avg, std, filename = "my_model"):
    """
    model 저장 함수
    
    Argument:
    model -- 저장할 keras 모델
    avg -- 입력데이터의 feature축을 기준으로 한 평균 벡터
    std -- 입력데이터의 feature축을 기준으로 한 표준편차 벡터
    
    filename -- 저장할 파일 이름
    
    Returns:
    None
    """
    
    # 훈련된 모델과 파라미터를 저장할 폴더 생성
    os.mkdir(filename)
    
    # inference 시 z score normalization에 사용될 평균과 표준편차 저장
    np.save(filename + "/" + filename + "_avg.npy", avg)
    np.save(filename + "/" + filename + "_std.npy", std)
    
    model.save(filename + "/" + filename + ".h5")
    
def load_trained_model(filename = "my_model"):
    """
    훈련된 model을 불러오는 함수
    
    Argument:
    filename -- 저장할 파일 이름
    
    Returns:
    model -- 훈련된 keras 모델
    avg -- 입력데이터의 feature축을 기준으로 한 평균 벡터
    std -- 입력데이터의 feature축을 기준으로 한 표준편차 벡터
    """
    
    model = load_model(filename + "/" + filename + ".h5")
    
    avg = np.load(filename + "/" + filename + "_avg.npy")
    std = np.load(filename + "/" + filename + "_std.npy")
    
    return model, avg, std

def inference(model, X):
    """
    훈련된 model을 사용하여 입력 데이터 X의 label과 confience를 추정하는 함수
    
    Argument:
    model -- 훈련된 keras 모델
    X -- 입력 데이터
    
    Returns:
    labels -- 입력 데이터에 대한 예측된 label 값
    confidence -- label에 대한 확률 값
    """
    
    confidences = model.predict(X)
    labels = np.round(confidences)
    
    return labels, confidences

def get_train_history(history):
    epochs = len(history.history['loss'])
    texts = []
    
    for epoch in range(epochs):
        text = "Epoch %d/%d - loss: %f - accuracy: %f" % (epoch, epochs, history.history['loss'][epoch], history.history['accuracy'][epoch])
        texts.append(text)
        
    return texts