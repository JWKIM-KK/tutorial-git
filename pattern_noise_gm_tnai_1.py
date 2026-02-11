#!/usr/bin/env python
# coding: utf-8
#김진웅입니다.
## IMPORTS

# General
import sys
import os
import argparse
import pandas as pd
import numpy as np
import joblib
import io
import base64

# Internal modules
# "data_processing" 이 로컬에도 있고, 공통폴더에도 있어서 강제로 위치를 지정하여 가져옴
import importlib.util
module_path = 'Z:\\_DATA_Training\\_common\\data_processing.py'
spec = importlib.util.spec_from_file_location("data_processing", module_path)
data_processing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_processing)
sys.path.append('Z:\\_DATA_Training\\_common\\')
import report_generation

# Modeling
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.decomposition import PCA
from xgboost import XGBRegressor

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# # Azure ML
# from azureml.core import Workspace, Run, Dataset
# from azureml.core.run import _OfflineRun  # For local development

# Oracle DB
sys.path.append('Z:/_DATA_Training/_db')
from rcx import get_rcx_connection
from queries import pattern_noise_gm_tnai_query, ctb_compound_specs_query

data_source = "oracle_db"
output_model_dir = "Z:\_DATA_Training\_output_model\TEMP"
output_model_date = "20251103_TNAI"
# training_mode = "fixed_hyperparameters"
training_mode = "random_search"

# Concatenate the output model path with the date of the newly build model
output_model_dir = os.path.join(output_model_dir, output_model_date)

# Create the output model folder
os.makedirs(output_model_dir, exist_ok = True)

## DATA COLLECTION

# Retrieve data from Oracle DB
if data_source == 'oracle_db':
    tests_df = pd.read_sql_query(pattern_noise_gm_tnai_query, get_rcx_connection()) # Test data

    # Get pattern embeddings from local server (Old)
    # pattern_df = pd.read_parquet("http://10.82.79.72:9998/gzip/PATTERN")

    # Get pattern embeddings from local server (New_2025.08.14)
    performance = pd.read_parquet(r"http://10.82.79.72:9998/gzip/20250715_07000000_PERFORMANCE/PATTERN_EMBEDDING")
    emb = pd.read_parquet(r"http://10.82.79.72:9998/gzip/20250705_09141105_CLS-TOKEN/PATTERN_EMBEDDING")

    # 공통 컬럼 이름 자동 추출
    common_cols = list(set(performance.columns) & set(emb.columns))
    # 각 데이터프레임의 공통 컬럼 중 Null을 특정 문자열로 치환
    for col in common_cols:
        performance[f'_merge_{col}'] = performance[col].apply(lambda x: '__P_NULL__' if pd.isna(x) else x)
        emb[f'_merge_{col}'] = emb[col].apply(lambda x: '__P_NULL__' if pd.isna(x) else x)
    # 조인용 임시 컬럼명 리스트
    merge_cols = [f'_merge_{col}' for col in common_cols]
    # inner join 수행
    result = pd.merge(
        performance, emb[[i for i in emb.columns if i not in common_cols]],
        on=merge_cols,
        how='inner'
    )
    # 임시로 추가한 컬럼 삭제
    pattern_df = result.drop(columns=merge_cols)

# Convert dataframe columns to uppercase
tests_df.columns = tests_df.columns.str.upper()
pattern_df.columns = pattern_df.columns.str.upper()

# Convert air pressure and load units (★★★ Already converted)
# tests_df['AIR'] = tests_df.apply(lambda x: data_processing.convert_air_pressure_unit(x, 'AIR_PRESS_UNIT', 'AIR'), axis = 1).round(2)
# tests_df['TEST_LOAD'] = tests_df.apply(lambda x: data_processing.convert_load_unit(x, 'LOAD_UNIT', 'LOAD_1'), axis = 1).round(0)

# Fix column types
tests_df['PRODUCT_CODE'] = tests_df['PRODUCT_CODE'].astype(int)
tests_df['GM_SP_80'] = tests_df['GM_SP_80'].astype(float)
tests_df['GM_TNAI_80'] = tests_df['GM_TNAI_80'].astype(float)

# Preprocess data
print(f'Dataset size before preprocessing data : {tests_df.shape}')
tests_df= data_processing.process_specs(
    #(DataFrame, 'Cat or Numb', Replace CTB_Comp True or False, Validation Date)
    tests_df,
    cat_or_numb = 'n',
    replace_ctb_bool = True,
    ref_month = False)
# tests_df = data_processing.process_specs(tests_df, cat_or_numb='n', replace_bool=True)
print(f'Dataset size after preprocessing data : {tests_df.shape}')

# Remove duplicates request no. and test no.
tests_df = tests_df.drop_duplicates(['REQ_NO', 'TEST_NO', 'TIRE_NO', 'TEST_COND_NO'], keep="last")
print(f'Dataset size after removing duplicates : {tests_df.shape}')
tests_df = tests_df[~tests_df['REQ_NO'].isin(['22OE040197KI','22OE040205KI'])]

## EXTRACT PATTERN INFORMATION FROM EMBEDDINGS

# Number of dimensions for PCA reduction
n_components = 14

# EMB_VIT_MAE 데이터로 PCA 하기
df_EMB_VIT_MAE = pattern_df.iloc[:,pattern_df.columns.get_loc('EMB_VIT_MAE_0001'):pattern_df.columns.get_loc('EMB_VIT_MAE_0768') + 1]
df_EMB_VIT_MAE = df_EMB_VIT_MAE.astype("float32")
pca = PCA(n_components=n_components)
pca.fit(df_EMB_VIT_MAE)

# Save PCA model to file
joblib.dump(pca, os.path.join(output_model_dir, 'pattern_noise_gm_tnai_pca_model.joblib'))

# EMB_SELF_PATCH 데이터는 PCA component 21개로 진행 시, variance ratio 합이 0.99 이상 확보가능
# EMB_SELF_PATCH 데이터는 PCA component 14개로 진행 시, variance ratio 합이 0.95 이상 확보가능
df_EMB_VIT_MAE_compressed = pd.DataFrame(pca.fit_transform(df_EMB_VIT_MAE))
df_EMB_VIT_MAE_compressed = df_EMB_VIT_MAE_compressed.astype("float32")

# Define column names for PCA dimensions
pca_features = [f'GM_TNAI_EMB_VIT_MAE_PCA_{i + 1}' for i in range(n_components)]
df_EMB_VIT_MAE_compressed.columns = pca_features

# CNN data에 PCA된 결과를 행방향으로 합치기
pattern_df = pd.concat([pattern_df, df_EMB_VIT_MAE_compressed], axis=1)

# Keep only useful columns
pattern_df = pattern_df[['SPEC_NO', 'PRODUCT_CODE', 'DRW_NO', 'DRW_REV', 'LAND_RATIO', 'TPI', 'PKX_N'] + pca_features]
pattern_df = pattern_df.rename(columns={'PRODUCT_CODE': 'PRODUCT_CODE_DRW'})

# Merge test data with pattern data
## No.1 : SPEC_NO 기준으로 Pattern embedding DB 매칭 (tests_df1_f 로 저장)
tests_df1 = pd.merge(tests_df, pattern_df, how='left', on='SPEC_NO')
tests_df1_f = pd.merge(tests_df, pattern_df, how='inner', on='SPEC_NO')

## No.2 : SPEC_NO 기준으로 매칭이 안된 것은 PRODUCT_CODE 기준으로 매칭 (tests_df2_f 로 저장)
pattern_df_unique = pattern_df.drop_duplicates(subset=['PRODUCT_CODE_DRW'])
pattern_df_unique['PRODUCT_CODE'] = pattern_df_unique['PRODUCT_CODE_DRW']
pattern_df_unique = pattern_df_unique.iloc[:, 1:]
df_missing_DRW_NO = tests_df1[tests_df1['DRW_NO'].isna()]
# pattern_df의 feature를 제거
pattern_df_feature_no = pattern_df.drop('SPEC_NO', axis=1).shape[1]
df_missing_DRW_NO = df_missing_DRW_NO.iloc[:, :-pattern_df_feature_no]
tests_df2 = pd.merge(df_missing_DRW_NO, pattern_df_unique, how='left', on='PRODUCT_CODE')
tests_df2_f = pd.merge(df_missing_DRW_NO, pattern_df_unique, how='inner', on='PRODUCT_CODE')

## No.3 : DRW_NO 기준으로 매칭 (tests_df3_f 로 저장)
df_combined_missing = tests_df2[tests_df2['DRW_NO'].isna()]
df_combined_missing2 = df_combined_missing.iloc[:, :-pattern_df_feature_no]
df_combined_missing3 = data_processing.process_DRW_NO(df_combined_missing2)
df_combined_missing3 = df_combined_missing3.rename(columns={'DRW_NO_Manual': 'DRW_NO'})
pattern_df_unique2 = pattern_df.drop_duplicates(subset=['DRW_NO'])
pattern_df_unique2 = pattern_df_unique2.iloc[:, 1:]
tests_df3_f = pd.merge(df_combined_missing3, pattern_df_unique2, how='inner', on='DRW_NO')

## 모든 방법으로도 매칭되는 결과가 없는 M-code를 print 하기
temp_df3_f = pd.merge(df_combined_missing3, pattern_df_unique2, how='left', on='DRW_NO')
missing_rows = temp_df3_f[temp_df3_f['LAND_RATIO'].isnull()]
missing_data = missing_rows[['PRODUCT_CODE','DRW_NO']]
missing_data_isnull = missing_data[missing_data['DRW_NO'].isnull()]
if missing_data_isnull is not None:
    print('Below is no connection drawing data with M-code')
    for value in missing_data_isnull['PRODUCT_CODE']:
        print(value)

## 총 3개의 dataframe을 합침
df_combined_final = pd.concat([tests_df1_f, tests_df2_f, tests_df3_f], ignore_index=True)
tests_df = df_combined_final
print(f'Dataset size after merging with pattern data : {tests_df.shape}')

validation_month = '2025-10-01'
tests_df_valmonth = tests_df.loc[tests_df['RESULT_DATE'] < validation_month]
tests_df_refmonth = tests_df.loc[tests_df['RESULT_DATE'] >= validation_month]
tests_df = tests_df_valmonth

# 'SEASON' 열에서 동일 항목의 개수를 카운트하여, 10개 이하인 행 제거
# tests_df = tests_df.groupby('SEASON').filter(lambda x: len(x) > 10)

# 'PATTERN' 열에서 'Q'를 포함하는 값을 가진 행 제거
# tests_df = tests_df[~tests_df['PATTERN'].str.contains('Q', na=False)]

# TEST_TIRE_STATUS 의 값이 없거나 'New'인 행만 남기기
tests_df = tests_df[tests_df['TEST_TIRE_STATUS'].isnull() | (tests_df['TEST_TIRE_STATUS'] == '') | (tests_df['TEST_TIRE_STATUS'] == 'NEW')]

# COMPOUND GUIDE 에 따른 트레드 컴파운드 HS 정의
tests_df = tests_df.reset_index(drop=True)
tests_df = data_processing.process_ctb_compound(tests_df)
tests_df_refmonth = data_processing.process_ctb_compound(tests_df_refmonth)

# 인덱스 재설정
tests_df = tests_df.reset_index(drop=True)
tests_df_refmonth = tests_df_refmonth.reset_index(drop=True)

tests_df_HS = tests_df[['CTB_COMPOUND', 'HS' , 'HS_GUIDE']]
tests_df_unique = tests_df_HS.drop_duplicates()
tests_df_unique = tests_df_unique.dropna()
tests_df_unique = tests_df_unique.drop_duplicates('CTB_COMPOUND')

# 반복되는 숫자 식별 및 저장
label_counts = tests_df_unique['CTB_COMPOUND'].value_counts()
repeated_labels = label_counts[label_counts > 1].index.tolist()

# 두 열의 차이를 계산하여 새로운 열 추가
tests_df_unique['difference'] = tests_df_unique['HS_GUIDE'] - tests_df_unique['HS']
tests_df_unique = tests_df_unique.reset_index(drop=True)

# 결과 출력
print(tests_df_unique)

# 막대 그래프 시각화
plt.figure(figsize=(10, 6))
plt.bar(tests_df_unique['CTB_COMPOUND'], tests_df_unique['difference'], color='green')
plt.title('Difference between HS_GUIDE and HS')
plt.xlabel('Index')
plt.ylabel('Difference')
plt.xticks(tests_df_unique['CTB_COMPOUND'], rotation=90)
# 반복되는 숫자를 그래프의 오른쪽에 표시
for i, label in enumerate(repeated_labels):
    plt.annotate(f'Repeated: {label}', xy=(1, 1), xycoords='axes fraction', fontsize=10,
                 xytext=(15, -30 - i*15), textcoords='offset points', ha='right')
plt.grid(True)
plt.show()

## MODELING

# Dictionary to store the information to include in the training report
# Text data will be stored as strings in 'text' dictionary, and figures as base64 in 'plot' dictionary
report_output = {
                 'Data processing': {'text' : dict(), 'plot': dict()},
                 'Input features' : {'size_features' : tuple(),
                                     'numeric' : pd.DataFrame(), 'categoric': pd.DataFrame(),
                                     'plot': dict()},
                 'Hyper parameters' : {'train mode' : training_mode,
                                       'parameters' : dict(),
                                       'plot' : dict()},
                 'Training results': {'text': dict(), 'text2' : dict(),  'text3' : dict(),
                                      'plot': dict()}
                }

def add_tooltip(column_name):
    if column_name in report_generation.HYPERPARAMETERS:
        return f"""{column_name} 
            <span class="tooltip" onclick="openModal('{column_name}')">?</span>"""
    return column_name  # 툴팁 정보가 없는 경우 원래 컬럼명 유지

# Best random search parameters to include in the training report if needed
best_params = ''

# Model features
features = ['TEST_LOAD', 'AIR', 'RIM_WIDTH', # 시험조건
            'SIZE_INCH', # 사이즈 'SIZE_NSW_MM' 'SIZE_SERIES_MM'
            'HS_GUIDE',  # 트래드 컴파운드 특성 'M10' 'SUT_STEP_GAUGE' 'TREAD_TOTAL_GAUGE'
            'CC_PLY', 'C01_MAT', # 카카스 PLY
            'RBT_TYPE', 'RBT_MAT',#  # 보강벨트 특성
            'BT1_WIDTH', 'BT1_ANGLE', 'BT1_MAT_EPI' , # 벨트 특성
            'MOLD_SD', 'MOLD_TDW', # Mold dimension
            'LAND_RATIO', # LAND RATIO
            'TPI', # TPI (Snow index, 높을수록 Snow 성능에 유리, Lateral/Kerf void에 따라 결정)
            'PKX_N' # PKX_N
            ] + pca_features # VIT-MAE 모델로 패턴 이미지를 숫자로 임베딩 한 것

# Model targets and default hyperparameters
# Hyperparameters are in the following order : n_estimators, min_child_weight, max_depth, learning_rate, gamma, colsample_bytree
targets = {
            'GM_SP_80': [500, 10, 11, 0.1, 0.0, 0.3],
            'GM_TNAI_80': [500, 5, 11, 0.1, 0.3, 0.5]
          }

report_sweetviz = {}

for target in targets:

    print(f'Start model for target = {target}')

    # Get default hyperparameters
    n_estimators = targets[target][0]
    min_child_weight = targets[target][1]
    max_depth = targets[target][2]
    learning_rate = targets[target][3]
    gamma = targets[target][4]
    colsample_bytree = targets[target][5]

    # Keep only useful columns
    tests_df_target = tests_df[features + [target]]
    dataset_size_original = tests_df.shape

    # Remove rows with null values
    tests_df_target = tests_df_target.dropna()
    dataset_size_null_values = tests_df_target.shape

    # Calculate target average for duplicate features
    first_idx = tests_df_target.groupby(features).apply(lambda x: x.index[0])
    tests_df_target = tests_df_target.groupby(features).mean().reset_index()
    tests_df_target.index = first_idx
    dataset_size_duplicates = tests_df_target.shape

    # Remove target outliers based on IQR rule
    tests_df_target = data_processing.remove_outliers_iqr(tests_df_target, target, 3)
    dataset_size_outliers = tests_df_target.shape

    # Add dataset size information to report
    report_output['Data processing']['text'][(target)] = {'Original dataset size' : dataset_size_original,
                                                          'Remove null values' : dataset_size_null_values,
                                                          'Remove duplicates' : dataset_size_duplicates,
                                                          'Remove target outliers' : dataset_size_outliers
                                                         }
    report_sweetviz[f"{target}"] = tests_df_target

    # Plot target data distribution
    fig, ax = plt.subplots()
    tests_df_target[target].plot(kind='hist', figsize=(6, 3), ax=ax)
    ax.set_title(f"{target} - Distribution")
    fig.tight_layout()

    # # Log plot image to the Azure ML experiment
    # if type(run_context) != _OfflineRun:
    #     run_context.log_image(name= target + '_target_distrib', plot=plt, description= target + ' : Data distribution')
    # else:
    #     pass #plt.show()

    # Write image to base64 bytes for the training report
    stringIObytes = io.BytesIO()
    plt.savefig(stringIObytes, format = 'png')
    stringIObytes.seek(0)
    base64_data = base64.b64encode(stringIObytes.getvalue()).decode('utf8')

    # Add target distribution plot to report
    # report_output[f'Data processing']['plot'].update({f'{target}_target_distribution': base64_data})
    report_output[f'Data processing']['plot'].setdefault(f'{target}', []).append(base64_data)

    # Separate input features and target feature
    X = tests_df_target[features]
    y = tests_df_target[target]

    # Split dataset into train/test sets

    # RANDOM SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    # SPLIT BASED ON SPEC_NO (Make sure the same SPEC_NO doesn't get into train and test datasets)
    # test_samples = random.sample(tests_df_target['SPEC_NO'].unique().tolist(), int(tests_df_target.shape[0] * 0.2))
    # train_samples = [s for s in tests_df_target['SPEC_NO'].unique().tolist() if s not in test_samples]

    # df_test = tests_df_target[tests_df_target['SPEC_NO'].isin(test_samples)]
    # df_train = tests_df_target[tests_df_target['SPEC_NO'].isin(train_samples)]

    # X_train = df_train[features]
    # X_test = df_test[features]
    # y_train = df_train[target]
    # y_test = df_test[target]

    # Determine numeric and categorical features
    numeric_features = X_train.select_dtypes(exclude=['object']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    ordinal_features = []

    report_output['Input features']['numeric'] = X[numeric_features].agg(['max', 'min', 'mean', 'median']).round(2)
    report_output['Input features']['categoric'] = X[categorical_features].describe()
    report_output['Input features']['size_features'] = (len(numeric_features), len(categorical_features))

    # Create separate transformation pipelines for numeric and categorical features
    num_transform_pipeline = Pipeline(steps = [('scaler', MinMaxScaler())])
    cat_transform_pipeline = Pipeline(steps = [('onehotencoding', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    ord_transform_pipeline = Pipeline(steps = [('ordinalencoding', OrdinalEncoder(handle_unknown='error'))])

    # Create the overall transformation pipeline
    column_transformer = ColumnTransformer(
        transformers=[
            ("num_column_transformer", num_transform_pipeline, numeric_features),
            ("cat_column_transformer", cat_transform_pipeline, categorical_features),
            ("ord_column_transformer", ord_transform_pipeline, ordinal_features)
        ], verbose_feature_names_out = False
    )

    # Create the final pipeline including data transformation and regressor
    # The regressor is given default hyperparameters based on prior experimentation
    pipeline = Pipeline([('column_transformer', column_transformer),
                         ('regressor', XGBRegressor(n_estimators = n_estimators, min_child_weight = min_child_weight, max_depth = max_depth, learning_rate = learning_rate, gamma = gamma, colsample_bytree = colsample_bytree))])

    # If we want to search for optimal hyperparameters
    if training_mode == "random_search" :

        # Hyperparameter values to feed to the RandomizedSearchCV
        param_grid = {
                    "regressor__n_estimators"     : [50, 100, 500, 900],
                    "regressor__learning_rate"    : [0.001, 0.01, 0.1, 0.3, 1],
                    "regressor__max_depth"        : np.arange(2, 20, 3),
                    "regressor__min_child_weight" : [1, 5, 10, 50],
                    "regressor__gamma"            : [0.0, 0.01, 0.1, 0.3, 0.5],
                    "regressor__colsample_bytree" : [0.3, 0.5, 0.7, 0.9, 1]
                    }

        # Instantiate a RandomizedSearchCV on an XGBRegressor model with n iterations
        random_cv = RandomizedSearchCV(estimator=pipeline, param_distributions = param_grid, n_iter = 100, scoring='neg_mean_absolute_error', verbose = 0, n_jobs = -1)

        # Fit model
        random_cv.fit(X_train, y_train)

        # Print training results
        bp = random_cv.best_params_
        print(f"Best score : {random_cv.best_score_}")
        print(f"Best parameters : {bp}")

        n_estimators = bp['regressor__n_estimators']
        min_child_weight = bp['regressor__min_child_weight']
        max_depth = bp['regressor__max_depth']
        learning_rate = bp['regressor__learning_rate']
        gamma = bp['regressor__gamma']
        colsample_bytree = bp['regressor__colsample_bytree']

        # Get best estimator
        pipeline = random_cv.best_estimator_

    # If we just want to use default hyperparameters
    elif training_mode == "fixed_hyperparameters" :

        # Fit pipeline
        pipeline.fit(X_train, y_train)

    ## Calculate metrics on training data
    y_pred = pipeline.predict(X_train)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred))
    mae_train = mean_absolute_error(y_train, y_pred)
    mape_train = mean_absolute_percentage_error(y_train, y_pred)
    r2_train = r2_score(y_train, y_pred)
    Train_real_data = y_train
    Train_pred_data = y_pred

    print(f'Dataset size Train: {X_train.shape[0]}')
    print(f'RMSE Train: {rmse_train}')
    print(f'MAE Train: {mae_train}')
    print(f'MAPE Train: {mape_train}')
    print(f'R^2 Train: {r2_train}')

    ## Calculate metrics on test data

    y_pred = pipeline.predict(X_test)

    if target == "GM_SP_80":
        now_tolerance = 1
        now_AP = 'A'
    elif target == "GM_TNAI_80":
        now_tolerance = 4
        now_AP = 'A'

    # GM-PV Format Validation Code
    # Return 받는 인자 개수가 늘어남을 고려, Dictionary 형태로 return됨.
    result_gmpv_validation = data_processing.validation_gmpv_mode(y_test, y_pred, now_tolerance, now_AP)

    # return 값 : (1)성공률 개수[int], (2)성공률 퍼센트[float], (3)평균오차[char]
    now_err_boundary_ea = result_gmpv_validation['success_ea']
    now_err_boundary_percent = result_gmpv_validation['success_percentage']

    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
    mae_test = mean_absolute_error(y_test, y_pred)
    mape_test = mean_absolute_percentage_error(y_test, y_pred)
    r2_test = r2_score(y_test, y_pred)
    Test_real_data = y_test
    Test_pred_data = y_pred

    if target == "GM_TNAI_80":
        diff_TNAI_80 = Test_real_data - Test_pred_data
        temp1 = pd.DataFrame()
        temp1['Test_real_data'] = Test_real_data
        temp1['Test_pred_data'] = Test_pred_data
        temp1['diff'] = diff_TNAI_80
        # 최대값 3개에 해당하는 인덱스 찾기
        max_indices = diff_TNAI_80.nlargest(3).index
        print(max_indices)
        df_max = tests_df_target.loc[max_indices]
        df_max_value = temp1.loc[max_indices]
        # 두 데이터프레임을 행 방향으로 합치기
        df_max_combined = pd.concat([df_max, df_max_value], axis=1)

        # 최소값 3개에 해당하는 인덱스 찾기
        min_indices = diff_TNAI_80.nsmallest(3).index
        print(min_indices)
        df_min = tests_df_target.loc[min_indices]
        df_min_value = temp1.loc[min_indices]
        # 두 데이터프레임을 행 방향으로 합치기
        df_min_combined = pd.concat([df_min, df_min_value], axis=1)

    print(f'Dataset size Test: {X_test.shape[0]}')
    print(f'RMSE Test: {rmse_test}')
    print(f'MAE Test: {mae_test}')
    print(f'MAPE Test: {mape_test}')
    print(f'R^2 Test: {r2_test}')

    if target == 'GM_TNAI_80':

        # GM P-V CHART VALIDATION ----------------------------------------------------------------
        PVchart_temp0 = pd.read_csv('./nvh_tool_training_y2024/training_scripts/2024_GM_PV_physics_data.txt',
                                    sep='\t')
        PVchart_temp0['Air_kgf_cm2'] = PVchart_temp0['Air_kgf_cm2'].astype(float)
        PVchart_temp0 = PVchart_temp0.rename(columns={'Air_kgf_cm2': 'AIR'})
        PVchart_temp0['TEST_LOAD'] = PVchart_temp0['TEST_LOAD'].astype(float)
        PVchart_spec_list = PVchart_temp0['Spec_no_GMPV']
        PVchart_spec_list = PVchart_spec_list.values.tolist()

        import _auth_config as auth
        import cx_Oracle

        # 오라클 DB 접속 아이디 및 접속 주소 등 정보 입력
        credentials = f"{auth.username}/{auth.password}@{auth.host}:{auth.port}/{auth.servicename}/"
        connection = cx_Oracle.connect(credentials, encoding="UTF-8", nencoding="UTF-8")

        df_result_PV = pd.DataFrame()
        for i in range(len(PVchart_spec_list)):
            spec_no = PVchart_spec_list[i]
            query1 = f"SELECT * \
                        FROM RCXUSER.da_spec\
                        WHERE da_spec.spec_no = '{spec_no}'"
            temp0 = pd.read_sql(query1, connection)

            air = PVchart_temp0['AIR'][i]
            test_load = PVchart_temp0['TEST_LOAD'][i]
            rim_width = PVchart_temp0['RIM_WIDTH'][i]
            Drawing_no = PVchart_temp0['Drawing_no'][i]

            # [temp1, Pattern_Image, df_mold_pattern, df_product] = Data_Cleaning(temp0, air, test_load, rim_width, use_cols, scale_cols)
            #
            # df_result = Prediction(temp1, BestModel)
            df_result_PV = pd.concat([df_result_PV, temp0], ignore_index=True)

            print(f"{round((i + 1) / (len(PVchart_spec_list)) * 100)}% 진행됨.")

        # 'Product' 열에서 'CX'를 'CQ'로 변경
        df_result_PV['CTB_COMPOUND'] = df_result_PV['CTB_COMPOUND'].str.replace('CX', 'CQ')

        df_result_PV2 = data_processing.process_specs(df_result_PV, cat_or_numb='n', replace_ctb_bool=True)
        df_result_PV2 = data_processing.process_ctb_compound(df_result_PV2)
        df_result_PV2 = df_result_PV2.reset_index(drop=True)
        # 두 데이터프레임을 열 방향으로 병합
        df_result_PV3 = pd.concat([PVchart_temp0, df_result_PV2], axis=1)

        # 빈 데이터 추가
        df_result_PV3.loc[df_result_PV3['SPEC_NO'] == 'DPKT2001863S00000', 'MOLD_SD'] = 10.5
        df_result_PV3.loc[df_result_PV3['SPEC_NO'] == 'DPKT2001863S00000', 'MOLD_TDW'] = 190
        df_result_PV3.loc[df_result_PV3['SPEC_NO'] == 'JMKT1014801S00000', 'MOLD_SD'] = 8.3
        df_result_PV3.loc[df_result_PV3['SPEC_NO'] == 'JMKT1014801S00000', 'MOLD_TDW'] = 204

        merged_pattern_df = pd.DataFrame()
        for i in range(len(df_result_PV3['SPEC_NO'])):
            # Spec_No 와 정확하게 일치하는 CNN DB 정보를 별도로 저장
            target_value = df_result_PV3['SPEC_NO'][i]
            matching_df = pattern_df[pattern_df['SPEC_NO'] == target_value]
            matching_df = matching_df.reset_index(drop=True)
            # 만약 매칭되는 SPEC_NO 가 없을 경우, PRODUCT_CODE 로 CNN DB 정보 중 가장 최신 정보를 별도로 저장
            target_value2 = int(df_result_PV3['PRODUCT_CODE'][i])
            if matching_df.empty:
                print("SPEC_NO와 매칭되는 CNN data 없기 때문에, 가장 최신 정보로 저장시도")
                matching_df = pattern_df[
                    pattern_df['SPEC_NO'].str.contains(f"{df_result_PV3['PRODUCT_CODE'][i]}", na=False)]
                matching_df = matching_df[:1]
                matching_df = matching_df.reset_index(drop=True)
            else:
                pass

            if matching_df.empty:
                target_value3 = df_result_PV3['Drawing_no'][i]
                print("PRODUCT_CODE 기준으로도 CNN data 정보가 없음. Drawing_no로 저장시도.")
                matching_df = pattern_df[pattern_df['DRW_NO'] == target_value3]
                matching_df = matching_df[:1]
                matching_df = matching_df.reset_index(drop=True)
            else:
                pass

            if matching_df.empty:
                print("CNN data 정보가 없음. 종료.")
            else:
                pass

            matching_df['target_value'] = target_value
            merged_pattern_df = pd.concat([merged_pattern_df, matching_df], axis=0)

        merged_pattern_df = merged_pattern_df.rename(columns={'SPEC_NO': 'SPEC_NO_drawing', 'target_value': 'SPEC_NO'})
        merged_pattern_df = merged_pattern_df.reset_index(drop=True)

        merged_df = pd.merge(df_result_PV3, merged_pattern_df, on='SPEC_NO', how='outer')

        extracted_df = merged_df[features]
        # NaN 값을 포함하는 행 제거
        extracted_df_cleaned = extracted_df.dropna()

        extracted_df_pred = pipeline.predict(extracted_df_cleaned)
        extracted_df_cleaned['Predicted level'] = extracted_df_pred
        extracted_df_final = extracted_df.join(extracted_df_cleaned['Predicted level'], how='outer')
        extracted_df_final = extracted_df_final.join(df_result_PV3['Physical(GM)'], how='outer')
        extracted_df_final['diff'] = extracted_df_final['Physical(GM)'] - extracted_df_final['Predicted level']
        # 평균값 계산
        TNAI_diff_withGM = extracted_df_final['diff'].mean()

        print(f"'diff' 열의 평균값: {TNAI_diff_withGM:.2f}")

        extracted_df_final['Predicted level for GM'] = extracted_df_final['Predicted level'] + TNAI_diff_withGM

        # NaN 값을 포함하는 행 제거
        extracted_df_final_cleaned = extracted_df_final.dropna()

        # A열과 B열의 값 차이가 4보다 크고, -4보다 작은 행 필터링
        condition = (extracted_df_final['Physical(GM)'] - extracted_df_final['Predicted level for GM'] > 4) | (
                    extracted_df_final['Physical(GM)'] - extracted_df_final['Predicted level for GM'] < -4)

        # 조건을 만족하는 행의 개수 구하기
        count_outlier = len(extracted_df_final[condition])
        count_all = len(extracted_df_final['Predicted level for GM'])
        Percent_under = (count_all - count_outlier) / count_all * 100

    y_pred_s = pd.Series(y_pred, index=y_test.index)
    seasons_test = tests_df.loc[list(y_test.index), "SEASON"].unique()
    report_output['Training results']['text3'].setdefault(target, {})
    for now_season in seasons_test:
        X_test_season = X_test.loc[tests_df.SEASON == now_season]
        y_test_season = y_test.loc[tests_df.SEASON == now_season]
        y__pred_season = y_pred_s.loc[tests_df.SEASON == now_season]
        result_gmpv_validation = data_processing.validation_gmpv_mode(y_test_season, y__pred_season, now_tolerance,
                                                                      now_AP)
        now_Seg_err_boundary_ea = result_gmpv_validation['success_ea']
        now_Seg_err_boundary_percent = result_gmpv_validation['success_percentage']
        now_avg_error = result_gmpv_validation['average_error']
        r2_season = r2_score(y_test_season, y__pred_season)

        report_output['Training results']['text3'][target][now_season] = {
            'Dataset Size': X_test_season.shape[0],
            'Success Rate[%]': data_processing.format_with_circle(round(now_Seg_err_boundary_percent, 2)),
            'P-V Chart R^2': round(r2_season, 3),
            'Prediction Error [Avg]': now_avg_error
        }

    # Fit pipeline on all data
    pipeline.fit(X, y)

    # Save model
    joblib.dump(pipeline, os.path.join(output_model_dir, f'{target}.joblib'))

    ## Calculate feature importance on all data

    # Feature importance based on feature permutation
    feature_names = pipeline['column_transformer'].get_feature_names_out()
    perm_imp = permutation_importance(
        pipeline._final_estimator, pipeline['column_transformer'].transform(X), y, n_repeats=10, random_state=42, n_jobs=-1
    )
    forest_importances = pd.Series(perm_imp.importances_mean, index=feature_names)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(figsize=(16, 5.5), yerr=perm_imp.importances_std, ax=ax)
    ax.set_title(f"{target} - Feature importance using permutation")
    ax.set_ylabel("Mean accuracy decrease")
    ax.tick_params(axis='x', which='major', labelsize=10)
    fig.tight_layout()

    # Write image to base64 bytes for the training report
    stringIObytes = io.BytesIO()
    plt.savefig(stringIObytes, format = 'png')
    stringIObytes.seek(0)
    base64_data = base64.b64encode(stringIObytes.getvalue()).decode('utf8')

    # Add training plot to report
    # report_output['Training results']['plot'].update({f'{target}_feature_importance': base64_data})
    report_output['Training results']['plot'].setdefault(f'{target}+', []).append(base64_data)

    # # Log plot image to the Azure ML experiment
    # if type(run_context) != _OfflineRun:
    #     run_context.log_image(name=target + '_feat_import_perm', plot=plt, description='Model ' + target + ' : Feature importance based on permutation')
    # else:
    #     pass #plt.show()

    # Calculate feature importance after grouping categorical features
    forest_importances_grouped = forest_importances[numeric_features + ordinal_features]
    for cat_feat in categorical_features:
        cols = [x for x in feature_names if x.startswith(cat_feat)]
        forest_importances_grouped[cat_feat] = forest_importances[cols].sum()

    # Sort from most important to least important
    forest_importances_grouped = forest_importances_grouped.sort_values(ascending = False)

    fig, ax = plt.subplots()
    forest_importances_grouped.plot.bar(figsize=(16, 5.5), ax=ax)
    ax.set_title(f"{target} - Feature importance using permutation (grouped)")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()

    # Write image to base64 bytes for the training report
    stringIObytes = io.BytesIO()
    plt.savefig(stringIObytes, format = 'png')
    stringIObytes.seek(0)
    base64_data = base64.b64encode(stringIObytes.getvalue()).decode('utf8')

    report_output['Training results']['plot'].setdefault(f'{target}+', []).append(base64_data)

    # Scatter plot
    plt.figure(figsize=(7, 6))
    plt.scatter(Train_real_data, Train_pred_data, color='blue',
                label=f"Train correlation ({Train_real_data.shape[0]} EA)")
    plt.scatter(Test_real_data, Test_pred_data, color='orange',
                label=f"Test correlation ({Test_real_data.shape[0]} EA), R-square {r2_test:.2f}")

    # 상관계수 계산
    r2_test = r2_score(Test_real_data, Test_pred_data)

    # 상관계수 표기
    plt.title(f'Prediction Model Validation with Real Result')
    plt.xlabel('Real value')
    plt.ylabel('Predicted value')
    plt.legend()
    plt.show()

    # Write image to base64 bytes for the training report
    stringIObytes = io.BytesIO()
    plt.savefig(stringIObytes, format='png')
    stringIObytes.seek(0)
    base64_data = base64.b64encode(stringIObytes.getvalue()).decode('utf8')

    # Add training plot to report
    # report_output['Training results']['plot'].update({f'{target}_feature_importance': base64_data})
    report_output['Training results']['plot'].setdefault(f'{target}+', []).append(base64_data)

    if target == 'GM_TNAI_80':
        # Scatter plot (GM PV chart)
        plt.figure(figsize=(7, 6))

        # 상관계수 계산
        r2_validataion = r2_score(extracted_df_final_cleaned['Physical(GM)'],
                           extracted_df_final_cleaned['Predicted level for GM'])
        plt.scatter(extracted_df_final_cleaned['Physical(GM)'], extracted_df_final_cleaned['Predicted level for GM'],
                    color='blue',
                    label=f"Test correlation ({extracted_df_final_cleaned['Predicted level for GM'].shape[0]} EA), R-square {r2_validataion:.2f}, TNAI within 4 : {Percent_under:.0f}%")
        # y = x 실선 추가 (axline 사용)
        plt.axline((0, 0), slope=1, color='green', linestyle='-')

        # y = x + 4 점선 추가 (axline 사용)
        plt.axline((0, 4), slope=1, color='orange', linestyle='--')
        plt.axline((0, -4), slope=1, color='orange', linestyle='--')

        plt.xlim(60, 100)
        plt.ylim(60, 100)

        # 상관계수 표기
        plt.title(f'GM TNAI Validation')
        plt.xlabel('Real TNAI(from GM)')
        plt.ylabel('Predicted TNAI')
        plt.legend()
        plt.show()

        # Write image to base64 bytes for the training report
        stringIObytes = io.BytesIO()
        plt.savefig(stringIObytes, format='png')
        stringIObytes.seek(0)
        base64_data = base64.b64encode(stringIObytes.getvalue()).decode('utf8')

        # Add training plot to report
        # report_output['Training results']['plot'].update({f'{target}_feature_importance': base64_data})
        report_output['Training results']['plot'].setdefault(f'{target}+', []).append(base64_data)

    ## Calculate metrics on all data

    y_pred = pipeline.predict(X)

    rmse_all = np.sqrt(mean_squared_error(y, y_pred))
    mae_all = mean_absolute_error(y, y_pred)
    mape_all = mean_absolute_percentage_error(y, y_pred)
    r2_all = r2_score(y, y_pred)

    print(f'Dataset size All : {X.shape[0]}')
    print(f'RMSE All : {rmse_all}')
    print(f'MAE All : {mae_all}')
    print(f'MAPE All : {mape_all}')
    print(f'R^2 All : {r2_all}')

    # Visualize the predictions (in blue) against the actual values (in red)
    fig, ax = plt.subplots()
    ax = sns.kdeplot(y, color='r', label='actual')
    sns_plot = sns.kdeplot(y_pred, color='b',label='prediction', ax=ax)

    # 기준 월로 필터링된 데이터셋으로 학습데이터셋 구성
    tests_df_target_refmonth = tests_df_refmonth[features + [target]]
    tests_df_target_refmonth = tests_df_target_refmonth.dropna()  ## ←중요! Drop NaN 꼭 해야함,,
    X_refmonth = tests_df_target_refmonth[features]
    y_refmonth = tests_df_target_refmonth[target]

    # 기준월 데이터셋 예측 진행
    y_pred = pipeline.predict(X_refmonth)

    # 출력을 위한 검증결과 변수할당
    result_gmpv_validation = data_processing.validation_gmpv_mode(y_refmonth, y_pred, now_tolerance, now_AP)
    refmonth_err_boundary_ea = result_gmpv_validation['success_ea']
    refmonth_err_boundary_percent = result_gmpv_validation['success_percentage']
    refmonth_avg_error = result_gmpv_validation['average_error']
    r2_refmonth = r2_score(y_refmonth, y_pred)

    plt.figure(figsize=(7, 6))

    # 상관계수 계산
    plt.scatter(y_refmonth, y_pred,
                color='blue',
                label=f"Test correlation ({y_refmonth.shape[0]} EA), R-square {r2_refmonth:.2f}, Success_percentage : {refmonth_err_boundary_percent:.0f}%")
    # y = x 실선 추가 (axline 사용)
    plt.axline((0, 0), slope=1, color='green', linestyle='-')

    # y = x + 4 점선 추가 (axline 사용)
    plt.axline((0, now_tolerance), slope=1, color='orange', linestyle='--')
    plt.axline((0, -now_tolerance), slope=1, color='orange', linestyle='--')

    if target == 'GM_SP_80':
        plt.xlim(85, 110)
        plt.ylim(85, 110)
    else:
        plt.xlim(60, 100)
        plt.ylim(60, 100)

    # 상관계수 표기
    plt.title(f'Monthly Verification - {target}')
    plt.xlabel('Physics')
    plt.ylabel('Predicted')
    plt.legend()
    plt.show()

    # Write image to base64 bytes for the training report
    stringIObytes = io.BytesIO()
    plt.savefig(stringIObytes, format='png')
    stringIObytes.seek(0)
    base64_data = base64.b64encode(stringIObytes.getvalue()).decode('utf8')

    # Add training plot to report
    # report_output['Training results']['plot'].update({f'{target}_feature_importance': base64_data})
    report_output['Training results']['plot'].setdefault(f'{target}+', []).append(base64_data)

    # # Log plot image to the Azure ML experiment
    # if type(run_context) != _OfflineRun:
    #     run_context.log_image(name=target + '_actual_vs_pred', plot=plt, description='Model ' + target + ' : Predictions (in blue) against actual values (in red)')
    # else:
    #     pass #plt.show()

    # Add training results to report
    report_output['Training results']['text'][(target)] = {'Training dataset size' : X_train.shape[0],
                                                           'Test dataset size' : X_test.shape[0],
                                                           'Training MAE': round(mae_train, 3),
                                                           'Training MAPE': round(mape_train, 3),
                                                           'Training RMSE': round(rmse_train, 3),
                                                           'Training R2': round(r2_train, 3),
                                                           'Test MAE': round(mae_test, 3),
                                                           'Test MAPE': round(mape_test, 3),
                                                           'Test RMSE': round(rmse_test, 3),
                                                           'Test R2': round(r2_test, 3),
                                                           'Overall MAE': round(mae_all, 3),
                                                           'Overall MAPE': round(mape_all, 3),
                                                           'Overall RMSE': round(rmse_all, 3),
                                                           'Overall R2': round(r2_all, 3),
                                                           'Tolerance' : f"{round(now_tolerance,1)}%" if now_AP == 'P' else f"±{round(now_tolerance,1)}",
                                                           'Size of Under Tol' :  now_err_boundary_ea,
                                                           'Percentage of Under Tol' : data_processing.format_with_circle(round(now_err_boundary_percent,2))
                                                           }

    report_output['Hyper parameters']['parameters'][(target)] = {
        'n_estimators': n_estimators,
        'min_child_weight': min_child_weight,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'gamma': gamma,
        'colsample_bytree': colsample_bytree
    }

    # 월간 보고 출력 양식
    report_output['Training results']['text2'][(target)] = {
        'Validation Month': validation_month,
        'Dataset Size': X_refmonth.shape[0],
        'Success Rate[%]': data_processing.format_with_circle(round(refmonth_err_boundary_percent, 2)),
        'P-V Chart R^2': round(r2_refmonth, 3),
        'Prediction Error [Avg]': refmonth_avg_error
    }

## Generate the final report

# Convert nested dictionaries to dataframes and then to HTML code
transpose_numeric_table = report_output['Input features']["numeric"].transpose()
report_output['Input features']["numeric"]  = transpose_numeric_table.to_html(justify="center")
report_output['Input features']["categoric"]  = report_output['Input features']["categoric"].to_html(justify="center")

report_output["Hyper parameters"]["parameters"] = pd.DataFrame.from_dict(report_output["Hyper parameters"]["parameters"], orient='index')
report_output["Hyper parameters"]["parameters"].columns = [add_tooltip(col) for col in report_output["Hyper parameters"]["parameters"].columns]
report_output["Hyper parameters"]["parameters"] = report_output["Hyper parameters"]["parameters"].to_html(classes="dataframe", escape=False, justify="center")  # escape=False 필수

report_output['Data processing']['text']  = pd.DataFrame.from_dict(report_output['Data processing']['text'], orient='index').to_html(justify="center")
report_output['Training results']['text']  = pd.DataFrame.from_dict(report_output['Training results']['text'], orient='index').to_html(justify="center")
report_output['Training results']['text2']  = pd.DataFrame.from_dict(report_output['Training results']['text2'], orient='index').to_html(justify="center")

# Generate HTML report
report_generation.generate_report("PATTERN_NOISE_GM_TNAI", output_model_date, report_output, output_model_dir)

###################################################################################################
import sweetviz as sv

for now_target in report_sweetviz.keys():

    # Sweetviz 분석: 단일 데이터셋
    train_report = sv.analyze(report_sweetviz[now_target])
    train_report.show_html(os.path.join(output_model_dir,f"PATTERN_NOISE_GM_TNAI_{now_target}_EDA.html"))

###################################################################################################
