import os
import pandas as pd
import numpy as np
from paddlets.transform import Fill, StandardScaler
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import warnings
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from sklearn.neighbors import LocalOutlierFactor
warnings.filterwarnings('ignore')


# 文件夹路径
folder_path = "区域赛训练集"

# 获取文件夹中所有文件的路径
file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]




selected_features = ['ROUND_A_WS_1', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE']  # 修改这里


def remove_outliers(df, column_names):
    for column in column_names:
        df = df[(df[column] >= -1e8) & (df[column] <= 1e8)]
    return df
# 用于填充缺失值的函数
def fill_missing_values(df):
    df['ROUND_A_WS_1'].fillna(df['WINDSPEED'], inplace=True)
    #print(f"After filling ROUND_A_WS_1, df shape is {df.shape}")

    df['YD15'].fillna(df['ROUND(A.POWER,0)'], inplace=True)
    #print(f"After filling YD15 with ROUND(A.POWER,0), df shape is {df.shape}")

    # 如果YD15仍有缺失值，则使用PREPOWER列进行填充
    df['YD15'].fillna(df['PREPOWER'], inplace=True)
    #print(f"After filling YD15 with PREPOWER, df shape is {df.shape}")

    # 处理在一段时间内，其他字段正常变化时，YD15持续完全不变的情况
    df['YD15_shifted'] = df['YD15'].shift()
    df['YD15_diff'] = df['YD15'] - df['YD15_shifted']

    # 如果YD15在一段时间内持续完全不变，则使用ROUND(A.POWER,0)进行替换
    df.loc[df['YD15_diff'] == 0, 'YD15'] = df.loc[df['YD15_diff'] == 0, 'ROUND(A.POWER,0)']

    # 再次填充缺失值
    df['YD15'].fillna(df['PREPOWER'], inplace=True)

    # 删除用于计算差值的列
    df.drop(['YD15_shifted', 'YD15_diff'], axis=1, inplace=True)

    # 异常值处理
    df = df[~((df['ROUND_A_WS_1'] > 12.0) & (df['YD15'] <= 0))]
    df = df[~((df['ROUND_A_WS_1'] <= 0) & (df['YD15'] > 0))]

    df = remove_outliers(df, ['ROUND_A_WS_1', 'YD15'])
    #print(f"After removing outliers, df shape is {df.shape}")
    # 填充其他缺失值
    df.fillna(method='ffill', inplace=True)

    return df


# 特征选择和添加时间特征的函数
def select_and_add_time_features(df):
    selected_features_with_datatime = selected_features + ['DATATIME']  # 临时添加 'DATATIME'
    X = df[selected_features_with_datatime]

    X['DATATIME'] = pd.to_datetime(X['DATATIME'])
    X['HOUR'] = X['DATATIME'].dt.hour
    X['DAY'] = X['DATATIME'].dt.day
    X['MONTH'] = X['DATATIME'].dt.month

    return X


# 添加延迟特征的函数
def add_lag_features(X):
    for feature in selected_features:
        for i in range(1, 4):
            X[feature + '_lag_' + str(i)] = X[feature].shift(i)

    X.fillna(method="ffill", inplace=True)  # 移除由于创建延迟特征而产生的NaN行
    X.reset_index(drop=True, inplace=True)  # 重新索引

    return X


# 对选择的特征进行缩放的函数
def scale_features(X):
    # 保存 'DATATIME' 列
    datetime_col = X['DATATIME'] if 'DATATIME' in X.columns else None

    # 移除 'DATATIME' 列
    if datetime_col is not None:
        X = X.drop('DATATIME', axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # 对所有的特征进行缩放

    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)  # 创建新的 DataFrame

    # 将 'DATATIME' 列重新加回 DataFrame
    if datetime_col is not None:
        X = pd.concat([X, datetime_col], axis=1)

    return X

# 添加交互特征
def add_interaction_features(X):
    # 风速-风向
    X['ROUND_A_WS_1_WINDDIRECTION'] = X['ROUND_A_WS_1'] * X['WINDDIRECTION']  # 修改这里
    # 风速-温度
    X['ROUND_A_WS_1_TEMPERATURE'] = X['ROUND_A_WS_1'] * X['TEMPERATURE']  # 修改这里
    # 风速-压力
    X['ROUND_A_WS_1_PRESSURE'] = X['ROUND_A_WS_1'] * X['PRESSURE']  # 修改这里

    return X

# 添加聚合特征
def add_aggregation_features(X):
    # 添加过去24小时的平均风速
    X['ROUND_A_WS_1_24H_MEAN'] = X['ROUND_A_WS_1'].rolling(window=24 * 4).mean()  # 修改这里

    # 添加过去24小时的最高风速
    X['ROUND_A_WS_1_24H_MAX'] = X['ROUND_A_WS_1'].rolling(window=24 * 4).max()  # 修改这里

    # 添加过去24小时的最低风速
    X['ROUND_A_WS_1_24H_MIN'] = X['ROUND_A_WS_1'].rolling(window=24 * 4).min()  # 修改这里

    return X

# 主数据清洗函数，调用上述帮助函数
def clean_data(df):
    df = fill_missing_values(df)
    X = select_and_add_time_features(df)
    X = add_lag_features(X)
    y = df['YD15']
    X = add_interaction_features(X)
    X = add_aggregation_features(X)
    X.fillna(method="ffill", inplace=True)  # 填补由于添加交互特征 和 聚合特征 产生的NaN值
    X = scale_features(X)  # 移动到所有特征创建完毕之后

    return X, y



# 主数据清洗函数，调用上述帮助函数
def clean_data(df):
    df = fill_missing_values(df)
    X = select_and_add_time_features(df)
    X = add_lag_features(X)
    X = add_interaction_features(X)
    X = add_aggregation_features(X)
    X.fillna(method="ffill", inplace=True)  # 填补由于添加交互特征 和 聚合特征 产生的NaN值
    y = df['YD15']

    X = scale_features(X)  # 移动到所有特征创建完毕之后

    return X, y


# 重新读取和清洗数据，获得清洗后的特征X和目标变量y
dfs_X = []
dfs_y = []

for file_path in file_paths:
    df = pd.read_csv(file_path)
    df = df.rename(columns={'ROUND(A.WS,1)': 'ROUND_A_WS_1'})
    X, y = clean_data(df)
    dfs_X.append(X)
    dfs_y.append(y)

    # # 计算相关性并排序
    # correlations = X.corrwith(y).abs().sort_values(ascending=False)
    #
    # # 打印相关性最高的10个特征
    # print(f"For file {file_path}:")
    # print(correlations.head(10))

n_machines = len(dfs_X)  # 风机的数量
pearson_matrix = np.zeros((n_machines, n_machines))

# 修改交互特征名，以匹配 DataFrame 中的实际列名
selected_features.extend(['ROUND_A_WS_1_WINDDIRECTION'])

# 在计算皮尔逊相关系数时，只需要考虑特征X，不需要考虑目标变量y
for i in range(n_machines):
    for j in range(i + 1, n_machines):
        merged_data = pd.merge(dfs_X[i], dfs_X[j], on='DATATIME', how='inner', suffixes=('_i', '_j'))

        if merged_data.empty:
            continue

        for feature in selected_features:
            feature_i = merged_data[feature + '_i']
            feature_j = merged_data[feature + '_j']
            pearson_coef, _ = pearsonr(feature_i, feature_j)
            pearson_matrix[i, j] += pearson_coef
            pearson_matrix[j, i] += pearson_coef

# 对皮尔逊相关系数求平均，得到最终的皮尔逊相关系数
pearson_matrix /= len(selected_features)

# 使用SpectralClustering进行聚类
from sklearn.cluster import SpectralClustering

k = 3  # 可以根据实际情况调整
spectral = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='discretize', random_state=0)

# 对皮尔逊相关系数矩阵进行谱聚类
labels = spectral.fit_predict(pearson_matrix)

# 输出每个风机的类别
print(labels)

# 创建一个字典来存储每个类别的数据
clusters = {i: [] for i in range(k)}

# 将每个风机的特征和目标值分配到对应的类别中
for machine_id, label in enumerate(labels):
    clusters[label].append((dfs_X[machine_id], dfs_y[machine_id]))


# 确保保存模型的文件夹存在
if not os.path.exists('model'):
    os.makedirs('model')

# 定义优化的目标函数
def objective(params):
    params = {
        'num_leaves': int(params['num_leaves']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
        'min_child_samples': int(params['min_child_samples']),
        'max_depth': int(params['max_depth']),
        'reg_alpha': '{:.3f}'.format(params['reg_alpha']),
        'reg_lambda': '{:.3f}'.format(params['reg_lambda']),
    }

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.001,
        **params
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    score = mean_squared_error(y_test, y_pred)
    return {'loss': score, 'status': STATUS_OK}

# 定义超参数的空间
space = {
    'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
    'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    'max_depth': hp.quniform('max_depth', 5, 30, 1),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
}

# 对每个类别训练并保存一个 LightGBM 模型
for cluster_id, data in clusters.items():
    # 合并该类别下所有风机的特征和目标值
    X_cluster = pd.concat([d[0] for d in data], ignore_index=True)
    y_cluster = pd.concat([d[1] for d in data], ignore_index=True)

    # 移除'DATATIME'字段
    X_cluster = X_cluster.drop(columns=['DATATIME'])

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X_cluster, y_cluster, test_size=0.2, random_state=42)

    print(y_train.isnull().sum())  # 计算训练数据中NaN的数量
    print(y_test.isnull().sum())  # 计算测试数据中NaN的数量

    # 使用均值填充NaN
    y_train = y_train.fillna(y_train.mean())
    y_test = y_test.fillna(y_train.mean())  # 注意这里我们使用训练集的均值填充测试集

    # 使用 hyperopt 进行超参数优化
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20)
    print("Best hyperparameters:", best)


    # 使用最优的超参数训练模型
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.01,
        num_leaves=int(best['num_leaves']),
        colsample_bytree=best['colsample_bytree'],
        min_child_samples=int(best['min_child_samples']),
        max_depth=int(best['max_depth']),
        reg_alpha=best['reg_alpha'],
        reg_lambda=best['reg_lambda']
    )
    model.fit(X_train, y_train)

    # 验证模型
    y_pred = model.predict(X_test)
    print(f"Cluster {cluster_id}:")
    print(f"  MSE: {mean_squared_error(y_test, y_pred)}")
    print(f"  MAE: {mean_absolute_error(y_test, y_pred)}")

    # 保存模型
    joblib.dump(model, f'model/LightGBMmodel_{cluster_id}.joblib')

print("ok")