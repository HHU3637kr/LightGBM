# 导入需要的包
import os
import pandas as pd
import numpy as np
from joblib import load as load_model
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import warnings

warnings.filterwarnings('ignore')


import warnings

warnings.filterwarnings('ignore')

selected_features = ['WINDSPEED', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE']
selected_features_extended = selected_features + ['HOUR', 'DAY', 'MONTH'] + [f + '_lag_' + str(i) for f in selected_features for i in range(1, 4)] + \
                        ['WINDSPEED_WINDDIRECTION','WINDSPEED_TEMPERATURE' ,'WINDSPEED_PRESSURE'] + ['WINDSPEED_24H_MEAN' , 'WINDSPEED_24H_MAX' ,'WINDSPEED_24H_MIN' ]

#添加交互特征
def add_interaction_features(X):
    X['WINDSPEED_WINDDIRECTION'] = X['WINDSPEED'] * X['WINDDIRECTION']
    X['WINDSPEED_TEMPERATURE'] = X['WINDSPEED'] * X['TEMPERATURE']
    X['WINDSPEED_PRESSURE'] = X['WINDSPEED'] * X['PRESSURE']
    return X

# 添加聚合特征
def add_aggregation_features(X):
    # 添加过去24小时的平均风速
    X['WINDSPEED_24H_MEAN'] = X['WINDSPEED'].rolling(window=24 * 4).mean()

    # 添加过去24小时的最高风速
    X['WINDSPEED_24H_MAX'] = X['WINDSPEED'].rolling(window=24 * 4).max()

    # 添加过去24小时的最低风速
    X['WINDSPEED_24H_MIN'] = X['WINDSPEED'].rolling(window=24 * 4).min()
    return X

def clean_data(df):
    selected_features = ['WINDSPEED', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE']

    # 异常值处理
   # df = df[~((df['WINDSPEED'] > 6.0) & (df['YD15'] < 0))]

    # 利用WINDSPEED中的数据填补'ROUND(A.WS,1)'中的缺省值
    df['ROUND(A.WS,1)'].fillna(df['WINDSPEED'], inplace=True)

    # 其余缺失值，采用上一个不为空的数据进行填补
    df.fillna(method='ffill', inplace=True)

    # 特征选择
    selected_features_with_datatime = selected_features + ['DATATIME']  # 临时添加 'DATATIME'
    X = df[selected_features_with_datatime]

    # 添加时间特征
    X['DATATIME'] = pd.to_datetime(X['DATATIME'])
    X['HOUR'] = X['DATATIME'].dt.hour
    X['DAY'] = X['DATATIME'].dt.day
    X['MONTH'] = X['DATATIME'].dt.month

    # 添加延迟特征
    for feature in selected_features:
        for i in range(1, 4):
            X[feature + '_lag_' + str(i)] = X[feature].shift(i)
    X.fillna(method="ffill", inplace=True)  # 用前一个不为NaN的数据填充NaN
    X.reset_index(drop=True, inplace=True)  # 重新索引

    X = add_interaction_features(X)
    X = add_aggregation_features(X)

    X.fillna(method="ffill", inplace=True)  # 填补由于添加交互特征 和 聚合特征 产生的NaN值
    # 特征缩放
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[selected_features_extended])  # 只对 selected_features 进行缩放

    # 将缩放后的特征数据重新赋值给df
    df[selected_features_extended] = X_scaled

    return df

def forecast(df, turbine_id, out_file):
    # 创建输入数据的副本
    df_copy = df.copy()

    # 删除不必要的列
    df = df[selected_features_extended]

    # 加载模型
    loaded_models = [load_model(f'model/LightGBMmodel_{i}.joblib') for i in range(4)]

    # 使用模型进行预测
    results = [model.predict(df) for model in loaded_models]

    #这个地方应该还可以改进
    # 为模型设置权重
    weights = [6 / 18,7/18,3/18,2/18]

    # 确保权重和为1
    assert sum(weights) == 1.0, "Weights 1.0"

    # 使用权重计算平均预测结果
    result = np.average(results, axis=0, weights=weights)

    # 将预测结果转换为 DataFrame
    result_df = pd.DataFrame(result, columns=['YD15'])

    # 添加风场风机ID和DATATIME列
    result_df['TurbID'] = turbine_id
    result_df['DATATIME'] = df_copy['DATATIME'].values

    # 添加'ROUND(A.POWER,0)'列并设置为0
    result_df['ROUND(A.POWER,0)'] = 0

    # 调整列的顺序
    result_df = result_df[['TurbID', 'DATATIME', 'ROUND(A.POWER,0)', 'YD15']]

    # 输出预测结果
    result_df.to_csv(out_file, index=False)


if __name__ == "__main__":

    files = os.listdir('infile')
    if not os.path.exists('pred'):
        os.mkdir('pred')

    # 第一步，完成数据格式统一
    for f in files:
        print(f)
        # 获取文件路径
        data_file = os.path.join('infile', f)
        print(data_file)
        out_file = os.path.join('pred', f[:4] + 'out.csv')
        df = pd.read_csv(data_file,
                         parse_dates=['DATATIME'],
                         infer_datetime_format=True,
                         dayfirst=True,
                         dtype={
                             'WINDDIRECTION': np.float64,
                             'HUMIDITY': np.float64,
                             'PRESSURE': np.float64
                         })
        #数据清洗
        df = clean_data(df)

        # 获取风机号
        turbine_id = df.TurbID
        df = df.drop(['TurbID'], axis=1)
        # 裁剪倒数第二天5:00前的数据输入时间序列
        df = df.tail(4 * 24)

        forecast(df, turbine_id, out_file)