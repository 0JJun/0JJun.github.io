# Abnormal vibration discrimination model

###      """
###       Input : 진동센서에서 받아진 HealthIndex 데이터를 5초 기준으로 하나의 훈련 데이터로 생성
###       >> 연속된 5초의 데이터가 같은 라벨(정상 혹은 이상진동)일 때, 훈련 데이터로 생성
### 
###       Argument:
###       filename -- 저장된 csv 파일 이름
### 
###      Returns:
###       X -- 5초의 Health index 데이터
###       Y -- X와 매핑된 Label
###       """

# Model

![Model](https://user-images.githubusercontent.com/105787074/169681219-47d248fb-b009-43bf-aff9-c6a53dc29f42.PNG)
