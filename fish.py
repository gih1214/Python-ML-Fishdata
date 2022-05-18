# 도미와 빙어를 분류하는 머신러닝 모델 훈련

# numpy, pandas 라이브러리 설치
# python -m pip install numpy
# python -m pip install pandas
# python -m pip install matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# CSV 파일 불러오기
bream_length = pd.read_csv("C:/Users/Administrator/MyML/bream_length.csv")
bream_weight = pd.read_csv("C:/Users/Administrator/MyML/bream_weight.csv")
smelt_length = pd.read_csv("C:/Users/Administrator/MyML/smelt_length.csv")
smelt_weight = pd.read_csv("C:/Users/Administrator/MyML/smelt_weight.csv")

# matplot으로 시각화
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel('length')
plt.ylabel('weight')
# plt.show()

# csv 데이터 배열로 받기
bl = np.array(bream_length).flatten().tolist()
bw = np.array(bream_weight).flatten().tolist()
sl = np.array(smelt_length).flatten().tolist()
sw = np.array(smelt_weight).flatten().tolist()

# 1차원 배열로 데이터 합치기
length = bl + sl
weight = bw + sw
# print(length)
# print(weight)

# 2차원 배열로 만들기
fish_data = np.column_stack((length, weight))
# shape - 배열의 크기 확인 (샘플 수, 특성 수)
# print(fish_data.shape)

# target data 만들기 (도미 1, 빙어 0)
fish_target = [1]*34 + [0]*13
# print(fish_target)

# 리스트를 numpy 배열로 변경
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)
# print(input_arr)

# arange() - 1씩 증가하는 인덱스 만들기 (N-1까지)
np.random.seed(42)
index = np.arange(47)

# shuffle로 데이터 섞기
np.random.shuffle(index)
# print(index)

# 훈련 데이터
train_input = input_arr[index[:34]]
train_target = target_arr[index[:34]]
# 테스트 데이터
test_input = input_arr[index[34:]]
test_target = target_arr[index[34:]]

# train data를 matplot으로 시각화
plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(test_input[:, 0], test_input[:, 1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
