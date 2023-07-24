import xgboost
import lightgbm as lgb
import catboost
import ngboost
import tensorflow as tf
import keras


from ngboost.learners import default_tree_learner
from ngboost.distns import Normal
from ngboost.scores import MLE

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mlp
import warnings
import sklearn
import joblib
import os
from random import randint

from tqdm import tqdm
warnings.filterwarnings('ignore')


"""data 불러오기"""

df_train = pd.read_csv('./train_data/master_data.csv')     # 7~8월
df_test = pd.read_csv('./train_data/test_data.csv')        # 9월1일

df_train[df_train['AIR_PRESSURE'] < 900] = np.nan
df_train[df_train['AIR_PRESSURE'] > 1100 ] = np.nan

df_train[df_train['AIR_TEMPERATURE'] < -50] = np.nan
df_train[df_train['AIR_TEMPERATURE'] > 70 ] = np.nan

df_train[df_train['HUMIDITY'] < 0] = np.nan
df_train[df_train['HUMIDITY'] > 100 ] = np.nan

df_train[df_train['WIND_SPEED'] < 0] = np.nan
df_train[df_train['WIND_SPEED'] > 80 ] = np.nan

df_test[df_test['AIR_PRESSURE'] < 900] = np.nan
df_test[df_test['AIR_PRESSURE'] > 1100 ] = np.nan

df_test[df_test['AIR_TEMPERATURE'] < -50] = np.nan
df_test[df_test['AIR_TEMPERATURE'] > 70 ] = np.nan

df_test[df_test['HUMIDITY'] < 0] = np.nan
df_test[df_test['HUMIDITY'] > 100 ] = np.nan

df_test[df_test['WIND_SPEED'] < 0] = np.nan
df_test[df_test['WIND_SPEED'] > 80 ] = np.nan

"""변동값 테스트"""

df_train = df_train.interpolate(method="linear")
df_test = df_test.interpolate(method="linear")


##minmaxscaler가 원래 (X-min(X))/ (max(X)-min(X))
#MinMaxScaler 전처리
df_train['AIR_PRESSURE'] = (lambda ap : (ap-900) / 200)(df_train['AIR_PRESSURE'])
df_test['AIR_PRESSURE'] = (lambda ap : (ap-900) / 200)(df_test['AIR_PRESSURE'])
df_train['AIR_TEMPERATURE'] = (lambda at : (at+50) / 120)(df_train['AIR_TEMPERATURE'])
df_test['AIR_TEMPERATURE'] = (lambda at : (at+50) / 120)(df_test['AIR_TEMPERATURE'])       
df_train['day_min'] = (lambda dm : dm / 1439)(df_train['day_min'])
df_test['day_min'] = (lambda dm : dm / 1439)(df_test['day_min'])
df_train['HUMIDITY'] = (lambda h : h / 100)(df_train['HUMIDITY'])
df_test['HUMIDITY'] = (lambda h : h / 100)(df_test['HUMIDITY'])
df_train['WIND_SPEED'] = (lambda ws : ws / 80)(df_train['WIND_SPEED'])
df_test['WIND_SPEED'] = (lambda ws : ws / 80)(df_test['WIND_SPEED'])
#StandardScaler 전처리


feature_cols = ['AIR_TEMPERATURE', 'AIR_PRESSURE', 'day_min', 'HUMIDITY', 'WIND_SPEED']
label_cols = ['AIR_TEMPERATURE']

y_train = df_train[label_cols].values
X_train = df_train[feature_cols].values
y_test = df_test[label_cols].values
X_test = df_test[feature_cols].values


"""data 불러오기"""
AE_X_test = X_test.copy()
AER_X_test = X_test.copy()
AEG_X_test = X_test.copy()
AEL_X_test = X_test.copy()      #검색해서 카피한거 쓰는 이유도 명확히 
AECL_X_test = X_test.copy()

"""model 불러오기"""
AE_model = keras.models.load_model('./AE/0.00003522-0.00008174-0900.h5')
AER_model = keras.models.load_model('./AE-RNN/0.00000191-0.00000167-0993.h5')
AEG_model = keras.models.load_model('./AE-GRU/0.00000192-0.00000154-0995.h5')
AEL_model = keras.models.load_model('./AE-LSTM/0.00000194-0.00000162-0991.h5')
AECL_model = keras.models.load_model('./CGBaNN-AE-LSTM/0.00000165-0.00000115-0906.h5')

"""model 불러오기 끝"""


#non data [1074:1095]

##드랍할 구간
con_drop_length = [20,40,60,150,300,450, 600]
# con_drop_length = [600]

fig_sv_sw = True

##신뢰성을 위해 한 모델당 50번 반복해서 이를 mape값 계산 
mape_AE_space = np.zeros((len(con_drop_length), 50))  ###################################
mape_AER_space = np.zeros((len(con_drop_length), 50))  ###################################
mape_AEG_space = np.zeros((len(con_drop_length), 50))  ###################################
mape_AEL_space = np.zeros((len(con_drop_length), 50))  ##################################
mape_AECL_space = np.zeros((len(con_drop_length), 50))  ##################################
mape_li_space = np.zeros((len(con_drop_length), 50))  ##################################

dp = np.zeros((len(con_drop_length), 50))  #con 뜻  기억안나면 control_drop_length라고 컨트롤 가능하게 인위적으로 드랍하는 길이라서 그렇다고 설명

for cdl in range(len(con_drop_length)):
    tn = 1438 - con_drop_length[cdl] - 2 ## 아마 tn = try_num
    ##1438인 이유는 아마 lstm이 여러개 값을 참조해서 넣기 떄문에2갠가 5개 뻈었고, 결측 구간 뺴고,
    ## 위에이유인지  아니면 앞부분은 예측에 쓸 데이터 없어서 줄이고 뒷부분은 예측할 구간이 없어서 제거
    """현재 디렉토리에 있는 모든 파일 리스트를 가져와서 해당 파일 형식이 없으면 생성, 있으면 pass"""
    if 'time_delta_2_{}'.format(con_drop_length[cdl]) in os.listdir():
        pass
    else:
        os.mkdir('./time_delta_2_{}/'.format(con_drop_length[cdl]))
    for try_num in tqdm(range(50)):
        drop_point = randint(25, tn)  ## 25이상 tn이하 난수 생성
        dp[cdl, try_num] = drop_point
        #drop_point = try_num   #continued drops start point 처음부터 끝까지(거의) 숫자 생성
        AE_X_test[drop_point: drop_point + con_drop_length[cdl] - 1, 0] = np.nan   # 생성된 정수 + 결측갯수 만큼 nun값 처리
        AER_X_test[drop_point: drop_point + con_drop_length[cdl] - 1, 0] = np.nan   # 생성된 정수 + 결측갯수 만큼 nun값 처리
        AEG_X_test[drop_point: drop_point + con_drop_length[cdl] - 1, 0] = np.nan   # 생성된 정수 + 결측갯수 만큼 nun값 처리
        AEL_X_test[drop_point: drop_point + con_drop_length[cdl] - 1, 0] = np.nan   # 생성된 정수 + 결측갯수 만큼 nun값 처리
        AECL_X_test[drop_point: drop_point + con_drop_length[cdl] - 1, 0] = np.nan   # 생성된 정수 + 결측갯수 만큼 nun값 처리
        for i in range(1438):
            X_test_li = X_test.copy()
            X_test_li[drop_point : drop_point + con_drop_length[cdl], 0] = np.linspace(X_test_li[drop_point-1, 0], X_test_li[drop_point + con_drop_length[cdl] - 1, 0], con_drop_length[cdl])

#            if str(AE_X_test[i, 1]) == 'nan' or str(AE_X_test[i, 1]) == 'na':
            if np.isnan(AE_X_test[i, 0]) == True:
                pred = AE_model.predict(AE_X_test[i - 24:i].reshape((1, 120)))[0][-1]
                AE_X_test[i, 0] = pred

            if np.isnan(AER_X_test[i, 0]) == True:
                pred = AER_model.predict(AER_X_test[i - 24:i].reshape((1, 120)))[0][-1]
                AER_X_test[i, 0] = pred

            if np.isnan(AEG_X_test[i, 0]) == True:
                pred = AEG_model.predict(AEG_X_test[i - 24:i].reshape((1, 120)))[0][-1]
                AEG_X_test[i, 0] = pred

            if np.isnan(AEL_X_test[i, 0]) == True:
                pred = AEL_model.predict(AEL_X_test[i - 24:i].reshape((1, 120)))[0][-1]
                AEL_X_test[i, 0] = pred

            if np.isnan(AECL_X_test[i, 0]) == True:
                pred = AECL_model.predict(AECL_X_test[i - 24:i].reshape((1, 120)))[0][-1]
                AECL_X_test[i, 0] = pred

            """이미 그래프가 그려져있으면 pass 아니면 저장"""
        if fig_sv_sw == True:
                ## 앞에 -1은 그래프 찍을때 첫 데이터는 예측값 y없음 맨뒤에 +1은 결측구간 예를 들어 20개면 20개째train data로 21번째 예측값이 결과로 나오니까 
            x_axis = np.arange(drop_point - 1, drop_point + con_drop_length[cdl] + 1)
            ### 원본,예측값 전체
            #fig, ax = plt.subplots(figsize=(10, 6))
            x = x_axis
            # (lambda at : ((at+1)*60)-50)
            y = (((y_test[drop_point - 1: drop_point + con_drop_length[cdl] + 1]) * 120) - 50)
            plt.plot(x, y, label='Actual')

            y1 = ((AE_X_test[drop_point - 1: drop_point + con_drop_length[cdl] + 1, 0] * 120) - 50)
            plt.plot(x, y1, label='AE')

            y2 = ((AER_X_test[drop_point - 1: drop_point + con_drop_length[cdl] + 1, 0] * 120) - 50)
            plt.plot(x, y2, label='AER')

            y3 = ((AEG_X_test[drop_point - 1: drop_point + con_drop_length[cdl] + 1, 0] * 120) - 50)
            plt.plot(x, y3, label='AEG')

            y4 = ((AEL_X_test[drop_point - 1: drop_point + con_drop_length[cdl] + 1, 0] * 120) - 50)
            plt.plot(x, y4, label='AEL')

            y5 = ((AECL_X_test[drop_point - 1: drop_point + con_drop_length[cdl] + 1, 0] * 120) - 50)
            plt.plot(x, y5, label='AECL')

            y7 = (((X_test_li[drop_point - 1: drop_point + con_drop_length[cdl] + 1, 0]) * 120) - 50)
            plt.plot(x, y7, label='linear_Predictionl')
            plt.legend()
            plt.savefig('./time_delta_2_{0}/interpolate_graph_{1} to {2}'.format(con_drop_length[cdl], drop_point, drop_point + con_drop_length[cdl]))
            plt.close()
        else:
            pass

        """AE 보간 mape값 계산"""
        AEP = AE_X_test[drop_point: drop_point + con_drop_length[cdl], 0]
        AEP = np.delete(AEP, np.where(AEP == 0))  ### ??? 왜 해놓은건지 이해 못함
        mape_AE = np.average(
            100 * np.abs((lambda at: (at * 120) - 50)(y_test[drop_point: drop_point + con_drop_length[cdl]]) -
                         (lambda at: (at * 120) - 50)(AEP))
            / (lambda at: (at * 120) - 50)(y_test[drop_point: drop_point + con_drop_length[cdl]]))

        """AER 보간 mape값 계산"""
        AERP = AER_X_test[drop_point: drop_point + con_drop_length[cdl], 0]
        AERP = np.delete(AERP, np.where(AEP == 0))
        mape_AER = np.average(
            100 * np.abs((lambda at: (at * 120) - 50)(y_test[drop_point: drop_point + con_drop_length[cdl]]) -
                         (lambda at: (at * 120) - 50)(AERP))
            / (lambda at: (at * 120) - 50)(y_test[drop_point: drop_point + con_drop_length[cdl]]))

        """AEG 보간 mape값 계산"""
        AEGP = AEG_X_test[drop_point: drop_point + con_drop_length[cdl], 0]
        AEGP = np.delete(AEGP, np.where(AEP == 0))
        mape_AEG = np.average(
            100 * np.abs((lambda at: (at * 120) - 50)(y_test[drop_point: drop_point + con_drop_length[cdl]]) -
                         (lambda at: (at * 120) - 50)(AEGP))
            / (lambda at: (at * 120) - 50)(y_test[drop_point: drop_point + con_drop_length[cdl]]))

        """AEL 보간 mape값 계산"""
        AELP = AEL_X_test[drop_point: drop_point + con_drop_length[cdl], 0]
        AELP = np.delete(AELP, np.where(AELP == 0))
        mape_AEL = np.average(
            100 * np.abs((lambda at: (at * 120) - 50)(y_test[drop_point: drop_point + con_drop_length[cdl]]) -
                         (lambda at: (at * 120) - 50)(AELP))
            / (lambda at: (at * 120) - 50)(y_test[drop_point: drop_point + con_drop_length[cdl]]))

        """AECL 보간 mape값 계산"""
        AECLP = AECL_X_test[drop_point: drop_point + con_drop_length[cdl], 0]
        AECLP = np.delete(AECLP, np.where(AECLP == 0))
        mape_AECL = np.average(
            100 * np.abs((lambda at: (at * 120) - 50)(y_test[drop_point: drop_point + con_drop_length[cdl]]) -
                         (lambda at: (at * 120) - 50)(AECLP))
            / (lambda at: (at * 120) - 50)(y_test[drop_point: drop_point + con_drop_length[cdl]]))

        """선형 보간 mape값 계산"""
        mape_li = np.average(100 * np.abs((lambda at : (at*120)-50)(y_test[drop_point: drop_point + con_drop_length[cdl]]) -
                          (lambda at : (at*120)-50)(X_test_li[drop_point: drop_point + con_drop_length[cdl], 1]))
             / (lambda at : (at*120)-50)(y_test[drop_point: drop_point + con_drop_length[cdl]]))

        """만들어놓은 저장공간에 20,40,60 mape값 정리"""
        mape_AE_space[cdl][try_num] = mape_AE
        mape_AER_space[cdl][try_num] = mape_AER
        mape_AEG_space[cdl][try_num] = mape_AEG
        mape_AEL_space[cdl][try_num] = mape_AEL
        mape_li_space[cdl][try_num] = mape_li

        AE_X_test = X_test.copy()
        AER_X_test = X_test.copy()
        AEG_X_test = X_test.copy()
        AEL_X_test = X_test.copy()
        AECL_X_test = X_test.copy()
        #


mape_mAE = []
mape_mAER = []
mape_mAEG = []
mape_mAEL = []
mape_mAECL = []
mape_mli = []

time_stamp = []
for cd in con_drop_length:
    time_stamp.append('time delta %d :'%cd)

np.savetxt('mape_AE_space.csv', mape_AE_space, delimiter=',')
np.savetxt('mape_AER_space.csv', mape_AER_space, delimiter=',')
np.savetxt('mape_AEG_space.csv', mape_AEG_space, delimiter=',')
np.savetxt('mape_AEL_space.csv', mape_AEL_space, delimiter=',')
np.savetxt('mape_AEL_space.csv', mape_AECL_space, delimiter=',')
np.savetxt('mape_li_space.csv', mape_li_space, delimiter=',')
np.savetxt('drop_space.csv', dp, delimiter=',')

"""XGB,Linear 보간법 별 반복횟수 만큼 생성된 모델의 평균,분산값"""
for i in range(len(con_drop_length)):
    mape_mAE.append('AE interpolate {0}\u03bc={1: .5f} \u03c3={2: .5f} '.format(time_stamp[i], np.average(mape_AE_space[i]), np.std(mape_AE_space[i])))
    mape_mAER.append('AER interpolate {0}\u03bc={1: .5f} \u03c3={2: .5f} '.format(time_stamp[i], np.average(mape_AER_space[i]), np.std(mape_AER_space[i])))
    mape_mAEG.append('AEG interpolate {0}\u03bc={1: .5f} \u03c3={2: .5f} '.format(time_stamp[i], np.average(mape_AEG_space[i]), np.std(mape_AEG_space[i])))
    mape_mAEL.append('AEL interpolate {0}\u03bc={1: .5f} \u03c3={2: .5f} '.format(time_stamp[i], np.average(mape_AEL_space[i]), np.std(mape_AEL_space[i])))
    mape_mAECL.append('AEL interpolate {0}\u03bc={1: .5f} \u03c3={2: .5f} '.format(time_stamp[i], np.average(mape_AEL_space[i]), np.std(mape_AEL_space[i])))
    mape_mli.append('linear interpolate {0}\u03bc={1: .5f} \u03c3={2: .5f}'.format(time_stamp[i], np.average(mape_li_space[i]), np.std(mape_li_space[i])))

"""위의 값을 가지고 정규분포를 따르는지 히스토그램을 그려 확인(신뢰성?)"""
for j in range(len(con_drop_length)):
    print('{}'.format(mape_mAE[j]))
    print('{}'.format(mape_mAER[j]))
    print('{}'.format(mape_mAEG[j]))
    print('{}'.format(mape_mAEL[j]))
    print('{}'.format(mape_mAECL[j]))
    print('{}\n'.format(mape_mli[j]))
#    fig, ax = plt.subplots(figsize=(10, 6))
#    plt.title('XGBoost_error(percent)_distribution')
#    plt.hist(mape_XG_space[j], bins=100, histtype='stepfilled')
#    plt.xticks(np.arange(0, np.max(mape_XG_space[j])+1, 0.5))
#    plt.xlim(0, 5)
#    plt.show()
#
#    plt.title('LGBoost_error(percent)_distribution')
#    plt.hist(mape_LG_space[j], bins=100, histtype='stepfilled')
#    plt.xticks(np.arange(0, np.max(mape_LG_space[j])+1, 0.5))
#    plt.xlim(0, 5)
#    plt.show()
#
#    plt.title('CatBoost_error(percent)_distribution')
#    plt.hist(mape_CB_space[j], bins=100, histtype='stepfilled')
#    plt.xticks(np.arange(0, np.max(mape_CB_space[j]) + 1, 0.5))
#    plt.xlim(0, 5)
#    plt.show()
#
#    plt.title('NGBoost_error(percent)_distribution')
#    plt.hist(mape_ng_space[j], bins=100, histtype='stepfilled')
#    plt.xticks(np.arange(0, np.max(mape_ng_space[j])+1, 0.5))
#    plt.xlim(0, 5)
#    plt.show()
#
#    plt.title('CGNN_error(percent)_distribution')
#    plt.hist(mape_CG_space[j], bins=100, histtype='stepfilled')
#    plt.xticks(np.arange(0, np.max(mape_CG_space[j])+1, 0.5))
#    plt.xlim(0, 5)
#    plt.show()
#
#    fig, ax = plt.subplots(figsize=(10, 6))
#    plt.title('linear_error(percent)_distribution')
#    plt.hist(mape_li_space[j], bins=100, histtype='stepfilled')
#    plt.xticks(np.arange(0, np.max(mape_li_space[j]) + 1, 0.1))
#    plt.xlim(0, 1)
#    plt.show()


"""
for i in range(1439):
#    if str(X_test[i][4]) == 'nan' or str(X_test[i][4]) == 'na':
#        X_test[i][4] = xgb_model.predict(np.expand_dims(X_test[i - 1][:], axis=0))
#        print(((X_test[i][4]) * 120) - 50)
if i>=1 and i<=1430:
X_test[i][4] = xgb_model.predict(np.expand_dims(X_test[i - 1][:], axis=0))
else:
pass

X_test[:, 4] = (lambda at: (at * 120) - 50)(X_test[:, 4])
y_val = (lambda at: (at * 120) - 50)(y_val)
#
x_axis = np.arange(1439)
## 원본,예측값 전체
fig, ax = plt.subplots(figsize=(10, 6))
x = x_axis
y = y_val
plt.plot(x, y, label='Actual')

x2 = x_axis
y2 = X_test[:, 4]
plt.plot(x2, y2, label='Predictionl')
plt.legend()
plt.show()
"""

####     x_axis_2 = np.arange(21)
####     # x_axis_2 = np.arange(41)
####     # x_axis_2 = np.arange(61)
####     fig, ax = plt.subplots(figsize=(20, 12))
####     x3 = x_axis_2
####     y3 = y_val[1073:1094]
####     #y3 = y_val[458:499]
####     #y3 = y_val[1346:1407]
####     plt.plot(x3, y3, label='Actual')
####     ## 예측값
####     x4 = x_axis_2
####     y4 = X_test[1073:1094, 4]
####     #y4 = X_test[458:499, 4]
####     #y4 = X_test[1346:1407, 4]
####     plt.plot(x4, y4, label='Predictionl')
####     plt.show()



#print(mse)
#print(mae)