from flask import Flask, request, jsonify
import requests, json
from io import StringIO
import mysql.connector
import pandas as pd
import numpy as np
import threading
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from tensorflow import keras
from keras.layers import Input, Reshape, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dropout, Dense
from keras.models import Model
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold



app = Flask(__name__)
# 创建数据库连接
config = {
    'user': 'root',
    'password': 'GJSE8YqDww',
    'host': 'localhost',
    'database': 'artificial_train',
    'raise_on_warnings': True
}
cnx = mysql.connector.connect(**config)  # 连接到 MySQL 数据库
cursor = cnx.cursor(dictionary=True)  # 创建一个可以返回字典格式数据的游标


@app.route('/describe/<int:file_id>', methods=['GET'])
def getFileDescribe(file_id):
    """
    根据文件的 ID 获取文件的描述信息
    :param file_id: 文件的 ID，应为整数
    :return: 文件描述的信息，包括前20行数据、列名列表、数据信息摘要和统计描述，如果文件ID不存在则返回错误信息
    """
    query = f"SELECT file_path FROM file_info WHERE file_id={file_id}"
    cursor.execute(query)
    result = cursor.fetchone()

    # 如果查询结果为空，返回错误信息
    if result is None:
        return jsonify({'status': 'error', 'message': 'No such file_id in database'}), 400

    file_path = result['file_path']  # 从查询结果中获取文件路径
    df = pd.read_csv(file_path)  # 读取文件为pandas dataframe

    # 获取前二十行数据并转换为字典
    top_20_rows = df.head(20).to_dict('records')
    # 获取所有列的 header 的名称列表
    column_names = list(df.columns)
    # 获取 dataframe.info() 的信息，并转换为字符串
    buffer = StringIO()
    df.info(buf=buffer)
    info = buffer.getvalue()
    # 获取 dataframe.describe() 的信息并转换为字典
    describe = df.describe().to_dict()

    # 将结果包装在一个字典中并返回
    result = {
        'top_20_rows': top_20_rows,
        'column_names': column_names,
        'info': info,
        'describe': describe
    }
    return jsonify({'status': 'completed', 'result': result}), 200



@app.route('/train', methods=['POST'])
def train():
    """
    提交训练任务的函数
    :return: 提交任务的反馈信息，格式可能会因具体情况而异
    """
    task_data = request.get_json()
    training_id = task_data.get('trainingId')
    task_id = task_data.get('taskId')

    task_results[task_id] = "BEGIN TRAINING"

    thread = threading.Thread(target=train_task, args=(training_id, task_id))
    thread.start()

    return jsonify({'status': 'processing', 'taskId': task_id}), 200



def train_task(training_id, task_id):
    """
    处理具体训练任务的函数
    :param training_id: 训练的 ID，应为整数
    :param task_id: 任务的 ID，应为整数
    :return: 任务处理的反馈信息，格式可能会因具体情况而异
    """
    # 使用trainingId查询operate_msg表，获取对应的所有列
    query = f"SELECT * FROM operate_msg WHERE operate_id={training_id}"
    cursor.execute(query)
    operate_msg = cursor.fetchone()
    if operate_msg is None:
        return jsonify({'status': 'error', 'message': 'No such trainingId in database'}), 400

    file_id = operate_msg['file_id']
    outlier_id = operate_msg['outlier_id']
    model_id = operate_msg['model_id']
    missing_id = operate_msg['missing_id']

    # 使用文件ID查询file_info表，获取文件路径
    query = f"SELECT file_path FROM file_info WHERE file_id={file_id}"
    cursor.execute(query)
    result = cursor.fetchone()
    if result is None:
        return jsonify({'status': 'error', 'message': 'No such fileId in database'}), 400
    file_path = result['file_path']

    # 读取CSV文件并转换为DataFrame
    dataframe = pd.read_csv(file_path)

    # 用outlier_id, model_id, missing_id分别在相应的表中查询并获取对应的所有列
    query = f"SELECT * FROM outlier_msg WHERE outlier_id={outlier_id}"
    cursor.execute(query)
    outlier_msg = cursor.fetchone()

    query = f"SELECT * FROM model WHERE model_id={model_id}"
    cursor.execute(query)
    model = cursor.fetchone()

    query = f"SELECT * FROM missing_msg WHERE missing_id={missing_id}"
    cursor.execute(query)
    missing_msg = cursor.fetchone()

    # 解析JSON字符串，获取列名列表
    missing_columns = json.loads(missing_msg['columns_json'])
    outlier_columns = json.loads(outlier_msg['cols_json'])

    # 根据missing_msg的type决定是删除缺失值还是填充缺失值
    if missing_msg['type'] == 0:  # 删除缺失值
        dataframe = delete_missing_values(dataframe, axis=0, columns=missing_columns)
    elif missing_msg['type'] == 1:  # 填充缺失值
        dataframe = fill_missing_values(dataframe, method=missing_msg['missing_method'], axis=0,
                                        columns=missing_columns)
    else:
        raise ValueError("无法识别的type值")

    # 根据outlier_msg的type决定是删除异常值还是替换异常值
    if outlier_msg['type'] == 0:  # 删除异常值
        dataframe = remove_outliers(dataframe, col=outlier_columns, method=outlier_msg['delete_method'],
                                    threshold=outlier_msg['threshold'])
    elif outlier_msg['type'] == 1:  # 替换异常值
        dataframe = replace_outliers(dataframe, threshold=outlier_msg['threshold'],
                                     replace_method=outlier_msg['replace_method'], col=outlier_columns,
                                     method=outlier_msg['detect_method'])
    else:
        raise ValueError("无法识别的type值")

    if dataframe.isnull().any().any():
        return jsonify({'status': 'error', 'message': 'Dataframe contains null values'}), 400
    else:
        print("DataFrame 中不存在空值")

    # 根据你的需求，这里可能需要对dataframe进行一些操作，然后将结果保存到task_results中
    task_results[task_id] = dataframe

    # 解析JSON字符串，将其转换为Python对象
    x_names = json.loads(model['x_names_json'])
    y_names = json.loads(model['y_names_json'])
    deep_cnn_hy_para = json.loads(model['deep_cnn_hy_para_json'])
    knn_hy_para = json.loads(model['knn_hy_para_json'])
    nn_hy_para = json.loads(model['nn_hy_para_json'])

    # 打印转换后的Python对象，以便检查
    print(f"x_names: {x_names}")
    print(f"y_names: {y_names}")
    print(f"deep_cnn_hy_para: {deep_cnn_hy_para}")
    print(f"knn_hy_para: {knn_hy_para}")
    print(f"nn_hy_para: {nn_hy_para}")

    # 打印获取到的消息
    print("OperateMsg:")
    print(operate_msg)
    print("\nOutlierMsg:")
    print(outlier_msg)
    print("\nMissingMsg:")
    print(missing_msg)
    print("\nModel:")
    print(model)

    # 当 nn_hy_para 不为空，knn_hy_para 和 deep_cnn_hy_para 为空的时候，把 nn_hy_para 转换成字典
    if nn_hy_para and not knn_hy_para and not deep_cnn_hy_para:
        hyperparameters = nn_hy_para
        print(hyperparameters)
        response = nn(dataframe, x_names, y_names, hyperparameters)

    # 当 knn_hy_para 不为空，nn_hy_para 和 deep_cnn_hy_para 为空的时候，把 knn_hy_para 转换成字典
    elif knn_hy_para and not nn_hy_para and not deep_cnn_hy_para:
        hyperparameters = knn_hy_para
        print(hyperparameters)
        response = knn(dataframe, x_names, y_names, hyperparameters)

    elif deep_cnn_hy_para and not knn_hy_para and not nn_hy_para:
        deep_cnn_dict = deep_cnn_hy_para

        # 从 deepCnnLayers 中提取出 convFilters、kernelSizes、poolSizes、activations 的列表
        deep_cnn_layers = deep_cnn_dict.get('deepCnnLayers', [])
        conv_filters = [layer.get('convFilter') for layer in deep_cnn_layers]
        kernel_sizes = [layer.get('kernelSize') for layer in deep_cnn_layers]
        pool_sizes = [layer.get('poolSize') for layer in deep_cnn_layers]
        activations = [layer.get('activation').lower() for layer in deep_cnn_layers]

        # 提取其他超参数，如果未提供则使用默认值
        hyperparameters = {
            'loss': deep_cnn_dict.get('lossMethod', 'binary_crossentropy').lower(),
            'convFilters': conv_filters,
            'kernelSizes': kernel_sizes,
            'poolSizes': pool_sizes,
            'lstmUnit': deep_cnn_dict.get('lstmUnit', 64),
            'dropoutRate': deep_cnn_dict.get('dropoutRate', 0.5),
            'batchSize': deep_cnn_dict.get('batchSize', 32),
            'epoch': deep_cnn_dict.get('epoch', 10),
            'activations': activations
        }

        # 输出最终的超参数字典
        print(hyperparameters)
        response = deepcnn(dataframe, x_names, y_names, hyperparameters)

    # 检查是否有所需的键
    if 'average_f1' in response and 'average_recall' in response and 'average_precision' in response and 'average_accuracy' in response:
        task_results[task_id] = response
        cnx.start_transaction()
        try:
            update_query = f"UPDATE operate_msg SET f1 = {response['average_f1']}, recall = {response['average_recall']}, pre = {response['average_precision']}, accuracy = {response['average_accuracy']} WHERE task_id = {training_id}"
            cursor.execute(update_query)
            cnx.commit()
        except mysql.connector.Error as err:
            print(f"Something went wrong: {err}")
            task_results[task_id] = f"Something went wrong: {err}"
            cnx.rollback()
    else:
        task_results[task_id] = response



@app.route('/result/<int:task_id>', methods=['GET'])
def get_result(task_id):
    """
    获取训练结果的函数
    :param task_id: 任务的 ID，应为整数
    :return: 训练结果的反馈信息，格式可能会因具体情况而异
    """
    # 检查结果是否已经准备好
    result = task_results.get(task_id, "no result")
    if result is not None:
        # 如果结果已经准备好，返回给客户端
        return jsonify({'status': 'completed', 'result': result}), 200
    else:
        # 否则，告诉客户端结果还未准备好
        return jsonify({'status': 'processing'}), 202



def delete_missing_values(df, axis=0, indices=None, columns=None):
    """
    删除缺失值的函数
    :param df: pandas数据集
    :param axis: 删除缺失值的方向，0表示按行删除，1表示按列删除，默认为0
    :param indices: 要删除的行的索引列表，默认为None，即不删除任何行
    :param columns: 要删除的列的名称列表，默认为None，即不删除任何列
    :return: 删除缺失值后的数据集
    """
    if axis == 0:
        # 按行删除缺失值
        if indices is None and columns is None:
            df_cleaned = df.dropna(axis=0)
        elif indices is not None and columns is None:
            df_cleaned = df.drop(index=indices)
        elif indices is None and columns is not None:
            df_cleaned = df.drop(columns=columns)
        else:
            df_cleaned = df.drop(index=indices, columns=columns)
    elif axis == 1:
        # 按列删除缺失值
        if indices is None and columns is None:
            df_cleaned = df.dropna(axis=1)
        elif indices is not None and columns is None:
            df_cleaned = df.drop(index=indices, axis=1)
        elif indices is None and columns is not None:
            df_cleaned = df.drop(columns=columns, axis=1)
        else:
            raise ValueError("不能同时指定行和列")
    else:
        raise ValueError("axis参数必须为0或1")

    return df_cleaned



def fill_missing_values(df, method='mean', axis=0, indices=None, columns=None, n_neighbors=5):
    """
    填充缺失值的函数
    :param df: pandas数据集
    :param method: 填充缺失值的方式，'mean'表示使用均值填充，'median'表示使用中位数填充，'mode'表示使用众数填充，
      'knn'表示使用最近邻插值填充，'rf'表示使用随机森林填充，'lr'表示使用线性回归填充，默认为'mean'
    :param axis: 填充缺失值的方向，0表示按行填充，1表示按列填充，默认为0
    :param indices: 要填充的行的索引列表，默认为None，即不填充任何行
    :param columns: 要填充的列的名称列表，默认为None，即不填充任何列
    :return: 填充缺失值后的数据集
    """
    if method == 'mean':
        # 使用均值填充缺失值
        if axis == 0:
            if indices is None and columns is None:
                return df.fillna(df.mean(axis=0))
            elif indices is not None and columns is None:
                return df.loc[indices].fillna(df.mean(axis=0))
            elif indices is None and columns is not None:
                return df[columns].fillna(df.mean(axis=0)[columns])
            else:
                return df.loc[indices, columns].fillna(df.mean(axis=0)[columns])
        elif axis == 1:
            if indices is None and columns is None:
                return df.fillna(df.mean(axis=1))
            elif indices is not None and columns is None:
                return df.loc[indices].fillna(df.mean(axis=1))
            elif indices is None and columns is not None:
                return df[columns].fillna(df.mean(axis=1))
            else:
                return df.loc[indices, columns].fillna(df.mean(axis=1))
        else:
            raise ValueError("axis参数必须为0或1")
    elif method == 'median':
        # 使用中位数填充缺失值
        if axis == 0:
            if indices is None and columns is None:
                return df.fillna(df.median(axis=0))
            elif indices is not None and columns is None:
                return df.loc[indices].fillna(df.median(axis=0))
            elif indices is None and columns is not None:
                return df[columns].fillna(df.median(axis=0)[columns])
            else:
                return df.loc[indices, columns].fillna(df.median(axis=0)[columns])
        elif axis == 1:
            if indices is None and columns is None:
                return df.fillna(df.median(axis=1))
            elif indices is not None and columns is None:
                return df.loc[indices].fillna(df.median(axis=1))
            elif indices is None and columns is not None:
                return df[columns].fillna(df.median(axis=1))
            else:
                return df.loc[indices, columns].fillna(df.median(axis=1))
        else:
            raise ValueError("axis参数必须为0或1")
    elif method == 'mode':
        # 使用众数填充缺失值
        if axis == 0:
            if indices is None and columns is None:
                return df.fillna(df.mode(axis=0).iloc[0])
            elif indices is not None and columns is None:
                return df.loc[indices].fillna(df.mode(axis=0).iloc[0])
            elif indices is None and columns is not None:
                return df[columns].fillna(df.mode(axis=0).iloc[0][columns])
            else:
                return df.loc[indices, columns].fillna(df.mode(axis=0).iloc[0][columns])
        elif axis == 1:
            if indices is None and columns is None:
                return df.fillna(df.mode(axis=1).iloc[:, 0])
            elif indices is not None and columns is None:
                return df.loc[indices].fillna(df.mode(axis=1).iloc[:, 0])
            elif indices is None and columns is not None:
                return df[columns].fillna(df.mode(axis=1).iloc[:, 0])
            else:
                raise ValueError("不能同时指定行和列")
        else:
            raise ValueError("axis参数必须为0或1")
    elif method == 'knn':
        # 使用最近邻插值填充缺失值
        if indices is not None and columns is not None:
            raise ValueError("最近邻插值法不能同时指定行和列")
        else:
            imputer = KNNImputer(n_neighbors=n_neighbors)
            if indices is None and columns is None:
                return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
            elif indices is not None and columns is None:
                return pd.DataFrame(imputer.fit_transform(df.loc[indices]), columns=df.columns)
            elif indices is None and columns is not None:
                return pd.DataFrame(imputer.fit_transform(df[columns]), columns=columns)
            else:
                return pd.DataFrame(imputer.fit_transform(df.loc[indices, columns]), columns=columns)
    elif method == 'rf':
        # 使用随机森林填充缺失值
        if indices is not None and columns is not None:
            raise ValueError("随机森林填充法不能同时指定行和列")
        else:
            rf = RandomForestRegressor(n_estimators=10, random_state=0)
            if indices is None and columns is None:
                missing = df.isnull()
                known = df[missing == False]
                unknown = df[missing]
                rf.fit(known.dropna(axis=1), known[unknown.columns])
                pred = rf.predict(unknown.dropna(axis=1))
                df[unknown.columns] = pred
                return df
            elif indices is not None and columns is None:
                missing = df.loc[indices].isnull()
                known = df.loc[indices][missing == False]
                unknown = df.loc[indices][missing]
                rf.fit(known.dropna(axis=1), known[unknown.columns])
                pred = rf.predict(unknown.dropna(axis=1))
                df.loc[indices, unknown.columns] = pred
                return df
            elif indices is None and columns is not None:
                missing = df[columns].isnull()
                known = df[columns][missing == False]
                unknown = df[columns][missing]
                rf.fit(known.dropna(axis=1), known[unknown.columns])
                pred = rf.predict(unknown.dropna(axis=1))
                df[unknown.columns] = pred
                return df
            else:
                missing = df.loc[indices, columns].isnull()
                known = df.loc[indices, columns][missing == False]
                unknown = df.loc[indices, columns][missing]
                rf.fit(known.dropna(axis=1), known[unknown.columns])
                pred = rf.predict(unknown.dropna(axis=1))
                df.loc[indices, unknown.columns] = pred
                return df
    else:
        raise ValueError("method参数必须为'mean', 'median', 'mode', ‘knn’, ‘rf’")



def remove_outliers(df, col=None, row=None, method='mean_std', threshold=3):
    """
    对数据集进行异常值删除的函数
    :param df: pandas数据集
    :param col: 待处理的列名，如果不指定，则对整个数据集进行处理。默认为None。
    :param row: 待处理的行名，如果不指定，则对整个数据集进行处理。默认为None。
    :param method: 异常值删除方法，可以选择'mean_std', 'boxplot', 'cluster'中的任意一个。默认为'mean_std'，表示基于均值和标准差的方法。
    :param threshold: 阈值，用于判断是否为异常值。默认为3。
    :return: 异常值删除后的数据集
    """
    # 根据指定的异常值删除方法对数据集进行处理
    if method == 'mean_std':
        if col is not None:
            mean = np.mean(df[col])
            std = np.std(df[col])
            df_processed = df[(df[col] >= mean - threshold * std) & (df[col] <= mean + threshold * std)]
        else:
            mean = np.mean(df)
            std = np.std(df)
            df_processed = df[(df >= mean - threshold * std) & (df <= mean + threshold * std)]
    elif method == 'boxplot':
        if col is not None:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            df_processed = df[(df[col] >= q1 - threshold * iqr) & (df[col] <= q3 + threshold * iqr)]
        else:
            q1 = df.quantile(0.25)
            q3 = df.quantile(0.75)
            iqr = q3 - q1
            df_processed = df[(df >= q1 - threshold * iqr) & (df <= q3 + threshold * iqr)]
    elif method == 'cluster':
        if col is not None:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=2, random_state=0).fit(df[col].values.reshape(-1, 1))
            df_processed = df[kmeans.labels_ == 0]
        else:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=2, random_state=0).fit(df.values.reshape(-1, 1))
            df_processed = df[kmeans.labels_ == 0]
    else:
        raise ValueError("Invalid outlier removal method. Please choose from 'mean_std', 'boxplot', 'cluster'.")

    # 根据指定的行、列删除对应的异常值
    if row is not None:
        df_processed = df_processed.drop(index=row)
    if col is not None:
        df_processed = df_processed.drop(columns=col)

    return df_processed



def replace_outliers(df, threshold=3, replace_method='median', col=None, row=None, replace_value=None,
                     method='z-score'):
    """
    替换异常值的函数
    :param df: pandas数据集
    :param threshold: 阈值，表示超过几个标准差为异常值，默认为3
    :param replace_method: 替换方法，可以选择'mean', 'median', 'mode', 'constant', 'random', 'model'中的任意一个，默认为'median'
    :param col: 待处理的列名，如果不指定，则对整个数据集进行处理。默认为None。
    :param row: 待处理的行名，如果不指定，则对整个数据集进行处理。默认为None。
    :param method: 识别异常值的方法，可以选择'zscore', 'boxplot', 'elliptic', 'local', 'iqr'中的任意一个，默认为'zscore'
    :return: 替换异常值后的数据集
    """
    # 根据用户选择的方法识别异常值
    if method == 'zscore':
        z_scores = pd.DataFrame()
        if col is not None:
            if df[col].dtype in [np.float64, np.int64]:
                z_scores[col] = (df[col] - df[col].mean()) / df[col].std()
            outliers = (z_scores > threshold) | (z_scores < -threshold)
        else:
            for col in df.columns:
                if df[col].dtype in [np.float64, np.int64]:
                    z_scores[col] = (df[col] - df[col].mean()) / df[col].std()
            outliers = (z_scores > threshold) | (z_scores < -threshold)
    elif method == 'boxplot':
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = (df < lower_bound) | (df > upper_bound)
    elif method == 'elliptic':
        ee = EllipticEnvelope()
        ee.fit(df)
        outliers = ee.predict(df) == -1
    elif method == 'local':
        lof = LocalOutlierFactor()
        outliers = lof.fit_predict(df) == -1
    elif method == 'iqr':
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = (df < lower_bound) | (df > upper_bound)
    else:
        raise ValueError(
            "Invalid method. Please choose from 'zscore', 'boxplot', 'elliptic', 'local', 'iqr'.")

    # 根据指定的行、列、数值替换对应的异常值
    if row is not None:
        outliers = outliers.drop(index=row)
    if col is not None:
        outliers = outliers.drop(columns=col)

    # 将异常值替换为指定值
    if replace_method == 'mean':
        replace_value = df.mean() if replace_value is None else replace_value
    elif replace_method == 'median':
        replace_value = df.median() if replace_value is None else replace_value
    elif replace_method == 'mode':
        replace_value = df.mode().iloc[0] if replace_value is None else replace_value
    elif replace_method == 'constant':
        replace_value = replace_value if replace_value is not None else 0
    elif replace_method == 'random':
        replace_value = replace_value if replace_value is not None else np.random.choice(df.values.flatten())
    elif replace_method == 'model':
        model = LinearRegression()
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                model.fit(df.loc[~outliers[col], df.columns != col], df.loc[~outliers[col], col])
                df.loc[outliers[col], col] = model.predict(df.loc[outliers[col], df.columns != col])
        return df
    else:
        raise ValueError(
            "Invalid replace method. Please choose from 'mean', 'median', 'mode', 'constant', 'random', 'model'.")

    df[outliers] = replace_value

    return df



def create_model(dim, conv_filters, kernel_sizes, pool_sizes, lstm_units, dropout_rate, num_conv_pool_layers, activations):
    """
    创建一个深度卷积神经网络模型
    :param dim: 输入维度，即特征的数量
    :param conv_filters: 卷积层的过滤器数量列表
    :param kernel_sizes: 卷积核大小列表
    :param pool_sizes: 池化层的池化大小列表
    :param lstm_units: LSTM单元数量
    :param dropout_rate: dropout层的丢弃比率
    :param num_conv_pool_layers: 卷积池化层的数量
    :param activations: 激活函数类型列表
    :return: 定义好的深度卷积神经网络模型
    """
    # 定义输入层
    input_layer = Input(shape=(dim,))
    reshape_layer = Reshape((dim, 1))(input_layer)
    x = reshape_layer

    # 使用循环构建卷积层和池化层
    for i in range(num_conv_pool_layers):
        x = Conv1D(conv_filters[i], kernel_sizes[i], activation=activations[i])(x)
        x = MaxPooling1D(pool_sizes[i])(x)

    # 定义BiLSTM层和dropout层
    bilstm_layer = Bidirectional(LSTM(lstm_units))(x)
    dropout_layer = Dropout(dropout_rate)(bilstm_layer)

    # 定义输出层
    output_layer = Dense(1, activation='sigmoid')(dropout_layer)

    # 定义模型
    model = Model(inputs=input_layer, outputs=output_layer)
    return model



def deepcnn(data, X_name, y_name, hyperparameters):
    """
    基于深度卷积神经网络的分类器
    :param data: pandas数据集
    :param X_name: 特征列名称列表
    :param y_name: 标签列名称
    :param hyperparameters: 模型超参数，包括损失函数类型、卷积层参数、LSTM单元数量、dropout比率、批大小、迭代次数和激活函数类型等
    :return: 各项性能指标的平均值，包括F1值、召回率、精度和准确率
    """
    if data is None or hyperparameters is None:
        return jsonify({"error": "Missing data or hyperparameters"}), 400

    X = data[X_name]
    y = data[y_name]
    dim = X.shape[1]

    loss = hyperparameters.get('loss', 'binary_crossentropy')
    conv_filters = hyperparameters.get('conv_filters', [64, 128, 256])
    kernel_sizes = hyperparameters.get('kernel_sizes', [3, 3, 3])
    pool_sizes = hyperparameters.get('pool_sizes', [2, 2, 2])
    lstm_units = hyperparameters.get('lstmUnit', 64)
    dropout_rate = hyperparameters.get('dropoutRate', 0.5)
    num_conv_pool_layers = len(conv_filters)
    batch_size = hyperparameters.get('batchSize', 32)
    epochs = hyperparameters.get('epoch', 10)
    activations = hyperparameters.get('activations', ['relu', 'relu', 'relu'])

    skf = StratifiedKFold(n_splits=10, random_state=40, shuffle=True)
    f1_scores = []
    recall_scores = []
    precision_scores = []
    accuracy_scores = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = create_model(dim, conv_filters, kernel_sizes, pool_sizes, lstm_units, dropout_rate,
                             num_conv_pool_layers, activations)
        model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])

        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)

        y_pred = model.predict(X_test)
        y_pred = np.round(y_pred)

        f1 = f1_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        accuracy = accuracy_score(y_test, y_pred)

        f1_scores.append(f1)
        recall_scores.append(recall)
        precision_scores.append(precision)
        accuracy_scores.append(accuracy)

    average_f1 = np.mean(f1_scores)
    average_recall = np.mean(recall_scores)
    average_precision = np.mean(precision_scores)
    average_accuracy = np.mean(accuracy_scores)

    return {
        "average_f1": average_f1,
        "average_recall": average_recall,
        "average_precision": average_precision,
        "average_accuracy": average_accuracy
    }



def knn(data, X_name, y_name, hyperparameters):
    #################################需要从外部掺入数据df,和超参数集hyperparameters###############################
    """
    n_neighbors: 范围可以从1到较大的整数，如50或100，具体取决于数据集的大小和问题的复杂性。通常需要选择一个奇数，以避免平局。
    weights: 范围： uniform（所有邻居的权重相等）和distance（权重与距离成反比）。
    algorithm: 范围： ball_tree、kd_tree、brute和auto。auto将会根据数据集的情况自动选择最合适的算法。
    leaf_size: 范围：大于零的整数。 传入可以尝试的范围是从5到100，具体取决于数据集的大小。
    p: 范围:1或2。
    metric: 范围 ：minkowski,euclidean、manhattan、chebyshev、haversine、hamming、jaccard、cosine。
    n_jobs: 范围： 1或-1
    """

    # 检查数据和超参数是否存在，如果不存在则返回错误
    if data is None or hyperparameters is None:
        return jsonify({"error": "Missing data or hyperparameters"}), 400

    # 通过X_name和y_name获取X和y的列索引
    header_row = data[0]
    X_index = header_row.index(X_name)
    y_index = header_row.index(y_name)

    # 将数据转换为NumPy数组
    data = np.array(data[1:])

    # 超参数
    n_neighbors = hyperparameters.get('nneighbor', 5)
    weights = hyperparameters.get('weight', 'uniform')
    algorithm = hyperparameters.get('algorithm', 'auto')
    leaf_size = hyperparameters.get('leafSize', 30)
    p = hyperparameters.get('p', 2)
    metric = hyperparameters.get('metric', 'minkowski')
    n_jobs = hyperparameters.get('njob', 1)

    # 提取X和y数据
    X = data[:, X_index].reshape(-1, 1)
    y = data[:, y_index]

    # 超参数
    # n_neighbors: 邻近数,默认为5,可选的值为奇数
    # weights: 邻近样本的权重,默认为uniform,可选值为‘distance’使用距离权重
    # algorithm: 用于查找最近邻居的算法,可选为'ball_tree'、'kd_tree'、'brute',默认为'brute'(朴素算法)
    # leaf_size: 传入的样本点中叶子节点的最小样本数,默认为30
    # p: 距离度量,默认为2(欧式距离),可选为1(曼哈顿距离)
    # metric: 距离度量指标,默认为minkowski,可选值为['euclidean'、'manhattan'、'chebyshev'、'haversine']等
    # n_jobs: 并行任务的数量,-1表示使用所有的CPUs

    # Initialize KNN model with provided parameters
    knn = KNeighborsClassifier(n_neighbors=n_neighbors,
                               weights=weights,
                               algorithm=algorithm,
                               leaf_size=leaf_size,
                               p=p,
                               metric=metric,
                               n_jobs=n_jobs)

    # 10-fold cross-validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

    accuracy = []
    f1 = []
    recall = []
    precision = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)

        accuracy.append(knn.score(X_test, y_test))
        f1.append(f1_score(y_test, y_pred, average='weighted'))
        recall.append(recall_score(y_test, y_pred, average='weighted'))
        precision.append(precision_score(y_test, y_pred, average='weighted'))

    result = {
        'average_f1': np.mean(f1),
        'average_recall': np.mean(recall),
        'average_precision': np.mean(precision),
        'average_accuracy': np.mean(accuracy)
    }

    return jsonify(result)



class IDSModel(nn.Module):
    """
    基于 LSTM 的模型定义
    :param input_size: 输入层大小
    :param hidden_size: 隐藏层大小
    :param num_layers: LSTM 层数
    :param output_size: 输出层大小
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(IDSModel, self).__init__()
        self.hidden_size = hidden_size  # 隐藏层神经元数量，范围建议：32-512
        self.num_layers = num_layers  # LSTM 层数，范围建议：1-4

        # 构建 LSTM 层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入特征
        :return: 输出结果
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]

        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out



def train_and_evaluate(X, y, input_size, hidden_size, num_layers, output_size, num_epochs, learning_rate, batch_size):
    """
    模型训练与评估函数
    :param X: 特征矩阵
    :param y: 目标向量
    :param input_size: 输入层大小
    :param hidden_size: 隐藏层大小
    :param num_layers: LSTM 层数
    :param output_size: 输出层大小
    :param num_epochs: 迭代次数，范围建议：10-100
    :param learning_rate: 学习率，范围建议：1e-5 - 1e-2
    :param batch_size: 批大小，范围建议：16-128
    :return: 一个字典，包含平均精度、召回率、F1 分数以及准确率
    """
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    f1_scores = []
    recall_scores = []
    precision_scores = []
    accuracy_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = np.expand_dims(X_train, axis=1)
        X_test = np.expand_dims(X_test, axis=1)

        model = IDSModel(input_size, hidden_size, num_layers, output_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 学习率，范围建议：1e-5 - 1e-2

        X_train_batches = torch.split(torch.tensor(X_train, dtype=torch.float32), batch_size)  # 批大小，范围建议：16-128
        y_train_batches = torch.split(torch.tensor(y_train, dtype=torch.long), batch_size)

        for epoch in range(num_epochs):  # 迭代次数，范围建议：10-100
            for i, (inputs, targets) in enumerate(zip(X_train_batches, y_train_batches)):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        inputs = torch.tensor(X_test, dtype=torch.float32)
        outputs = model(inputs)

        _, predicted = torch.max(outputs.data, 1)
        predicted_np = predicted.numpy()
        y_test_np = y_test

        accuracy = accuracy_score(y_test_np, predicted_np)
        precision = precision_score(y_test_np, predicted_np, average='weighted')
        recall = recall_score(y_test_np, predicted_np, average='weighted')
        f1 = f1_score(y_test_np, predicted_np, average='weighted')

        f1_scores.append(f1)
        recall_scores.append(recall)
        precision_scores.append(precision)
        accuracy_scores.append(accuracy)

    average_f1 = np.mean(f1_scores)
    average_recall = np.mean(recall_scores)
    average_precision = np.mean(precision_scores)
    average_accuracy = np.mean(accuracy_scores)

    return {
        'average_f1': average_f1,
        'average_recall': average_recall,
        'average_precision': average_precision,
        'average_accuracy': average_accuracy
    }



def nn(data, X_name, y_name, hyperparameters):
    """
    定义神经网络模型nn的API
    hidden_size: 范围：32-512。较小的值可能会导致模型的表示能力不足，而较大的值可能会导致过度拟合、增加计算成本和训练时间。
    num_layers: 范围：1-4。较少的层数可能会导致模型的表达能力不足，而较多的层数可能会导致梯度消失或梯度爆炸问题、增加计算成本和训练时间。
    num_epochs: 范围：10-100。较少的轮数可能会导致模型未充分训练，而较多的轮数可能会导致过度拟合和训练时间过长。
    learning_rate: 范围：0.00001 - 0.01。较小的学习率可能会导致模型收敛速度较慢，而较大的学习率可能会导致模型无法收敛或在最优解附近震荡。
    batch_size: 范围：16-128。较小的批次大小可能会导致梯度更新的方差较大，从而影响模型收敛速度，而较大的批次大小可能会导致内存需求增加和训练时间过长。
    """

    # 检查数据和超参数是否存在，如果不存在则返回错误
    if data is None or hyperparameters is None:
        return jsonify({"error": "Missing data or hyperparameters"}), 400

    # 通过X_name和y_name获取X和y的列索引
    header_row = data[0]
    X_index = header_row.index(X_name)
    y_index = header_row.index(y_name)

    # 将数据转换为NumPy数组
    data = np.array(data[1:])

    # 提取X和y数据
    X = data[:, X_index].reshape(-1, 1)
    y = data[:, y_index]

    # 超参数，范围过大可能造成时间太长和过度拟合
    input_size = X.shape[1]  # 输入特征的数量
    output_size = len(np.unique(y))  # 输出类别的数量
    hidden_size = hyperparameters.get('hiddenSize', 128)  # 隐藏层中神经元的数量 范围建议：32-512
    num_layers = hyperparameters.get('numLayers', 2)  # LSTM层的数量  范围建议：1-4
    num_epochs = hyperparameters.get('numEpochs', 20)  # 训练轮数 迭代次数，范围建议：10-100
    learning_rate = hyperparameters.get('learningRate', 0.001)  # 学习率   范围建议：1e-5 - 1e-2
    batch_size = hyperparameters.get('batchSize', 64)  # 每个批次的大小  范围建议：16-128

    metrics = train_and_evaluate(X, y, input_size, hidden_size, num_layers, output_size, num_epochs, learning_rate,
                                 batch_size)

    return jsonify(metrics)



if __name__ == '__main__':
    task_results = {}  # 存储处理结果的全局字典
    app.run(port=5000)

# 记得在程序结束时关闭数据库连接
cnx.close()
