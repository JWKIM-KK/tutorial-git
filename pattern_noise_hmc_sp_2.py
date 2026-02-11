#!/usr/bin/env python
# coding: utf-8

# EXECUTE LOCALLY FROM src/ folder : cmd /v:on /c "pushd ..\.. && set ROOT=!cd! && popd && set PYTHONPATH=!cd!;!ROOT! && python training_scripts\pattern_noise_gm_tnai.py"

## IMPORTS

# General
import json
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import io
import base64
from datetime import datetime
from pytz import timezone
from dotenv import load_dotenv

# Modeling
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# Visualization
import matplotlib.pyplot as plt

# Utility functions
from common.utils import drdpirs, helpers as common_helpers, model_explanation
from utils import helpers, report_generation, sql_queries


## DATA PROCESSING

prediction_item = "pattern_noise_hmc_sp"

# Load environment variables from .env file
load_dotenv(override=False)

# Get today's date in YYYYMMDD format
today = datetime.now(timezone('Asia/Seoul')).date().strftime('%Y%m%d')

# Parse arguments passed to the script execution
parser = argparse.ArgumentParser()
parser.add_argument("--output-model-dir", type=str, dest="output_model_dir", default="training_output", help="Path at which to store all trained output models")
parser.add_argument("--output-model-date", type=str, dest="output_model_date", default=today, help="Date in format YYYYMMDD to use as the name of the subfolder containing the newly trained model")
parser.add_argument("--training-mode", type=str, dest="training_mode", default="fixed_hyperparameters", help="Train model with fixed hyperparameters (fixed_hyperparameters) or execute a randomized search (random_search)")
args = parser.parse_args()
output_model_dir = args.output_model_dir
output_model_date = args.output_model_date
training_mode = args.training_mode

# Create the output model folder
output_model_dir = os.path.join(output_model_dir, output_model_date)
os.makedirs(output_model_dir, exist_ok = True)

# Read configuration file
filepath = os.path.join('common', 'config', 'process_specs.yaml')
prediction_item_config = common_helpers.read_yaml_file(filepath).get(prediction_item)


## DATA PROCESSING


# Retrieve data from Oracle DB
tests_df = pd.read_sql_query(sql_queries.pattern_noise_hmc_sp_query, drdpirs.get_db_connection()) # Test data

vit_mae_url = f"{os.getenv("PATTERN_API_URL")}/gzip/{os.getenv("PATTERN_EMBEDDINGS_VERSION")}//PATTERN_EMBEDDING"
performance_url = f"{os.getenv("PATTERN_API_URL")}/gzip/20250715_07000000_PERFORMANCE/PATTERN_EMBEDDING"

pattern_embeddings_vit_mae_df = pd.read_parquet(vit_mae_url)
pattern_embeddings_performance_df = pd.read_parquet(performance_url)

df_result_PV = pd.read_sql_query(sql_queries.da_spec_from_spec_no_list_query(tuple(PVchart_spec_list)), drdpirs.get_db_connection())

# Convert dataframe columns to uppercase
tests_df.columns = tests_df.columns.str.upper()
pattern_embeddings_vit_mae_df.columns = pattern_embeddings_vit_mae_df.columns.str.upper()
pattern_embeddings_performance_df.columns = pattern_embeddings_performance_df.columns.str.upper()
df_result_PV.columns = df_result_PV.columns.str.upper()

# Determine the start and end of cutoff date's month (month of output_model_date)
current_month_start = pd.to_datetime(output_model_date, format='%Y%m%d').replace(day=1)
current_month_end = (current_month_start + pd.DateOffset(months=1)) - pd.Timedelta(days=1)

# Filter tests_df based on the cutoff date
tests_df = tests_df.loc[tests_df['RESULT_DATE'] <= current_month_end]

# Process pattern embeddings
pattern_df = helpers.get_pattern_embeddings(pattern_embeddings_vit_mae_df, pattern_embeddings_performance_df)

# Convert air pressure and load units
tests_df['AIR'] = tests_df.apply(lambda x: helpers.convert_air_pressure_unit(x, 'AIR_PRESS_UNIT', 'AIR_PRESS_1'), axis = 1).round(2)
tests_df['TEST_LOAD'] = tests_df.apply(lambda x: helpers.convert_load_unit(x, 'LOAD_UNIT', 'LOAD_1'), axis = 1).round(0)

# Fix column types
tests_df['PRODUCT_CODE'] = tests_df['PRODUCT_CODE'].astype(int)
tests_df['HMC_SP_60'] = tests_df['HMC_SP_60'].astype(float)
tests_df['HMC_SP_80'] = tests_df['HMC_SP_80'].astype(float)

# Preprocess data
print(f'Dataset size before preprocessing data : {tests_df.shape}')
tests_df = common_helpers.process_specs(
    tests_df,
    training = True,
    cat_or_numb = prediction_item_config['cat_or_numb'],
    replace_ctb_bool = prediction_item_config['replace_ctb_bool']
)
print(f'Dataset size after preprocessing data : {tests_df.shape}')

# Remove duplicates request no. and test no.
tests_df = tests_df.drop_duplicates(['REQ_NO', 'TEST_NO', 'TIRE_NO', 'TEST_COND_NO'], keep="last")
print(f'Dataset size after removing duplicates : {tests_df.shape}')

## EXTRACT PATTERN INFORMATION FROM EMBEDDINGS

# Number of dimensions for PCA reduction
n_components = 14

# EMB_VIT_MAE 데이터로 PCA 하기
df_EMB_VIT_MAE = pattern_df.iloc[:,pattern_df.columns.get_loc('EMB_VIT_MAE_0001'):pattern_df.columns.get_loc('EMB_VIT_MAE_0768') + 1]
pca = PCA(n_components=n_components)
pca.fit(df_EMB_VIT_MAE)

# Save PCA model to file
joblib.dump(pca, os.path.join(output_model_dir, 'pattern_noise_hmc_sp_pca_model.joblib'))

# EMB_SELF_PATCH 데이터는 PCA component 21개로 진행 시, variance ratio 합이 0.99 이상 확보가능
# EMB_SELF_PATCH 데이터는 PCA component 14개로 진행 시, variance ratio 합이 0.95 이상 확보가능
df_EMB_VIT_MAE_compressed = pd.DataFrame(pca.fit_transform(df_EMB_VIT_MAE))

# Define column names for PCA dimensions
pca_features = [f'HMC_SP_EMB_VIT_MAE_PCA_{i + 1}' for i in range(n_components)]
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
df_combined_missing3 = helpers.process_DRW_NO(df_combined_missing2)
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

# TEST_TIRE_STATUS 의 값이 없거나 'New'인 행만 남기기
tests_df = tests_df[tests_df['TEST_TIRE_STATUS'].isnull() | (tests_df['TEST_TIRE_STATUS'] == '') | (tests_df['TEST_TIRE_STATUS'].str.upper() == 'NEW')]

# COMPOUND GUIDE 에 따른 트레드 컴파운드 HS 정의
tests_df = tests_df.reset_index(drop=True)
tests_df = common_helpers.process_ctb_compound(tests_df)

# Create a separate dataframe for monthly verification (data from current month)
tests_df_refmonth = tests_df.loc[(tests_df['RESULT_DATE'] >= current_month_start) & 
                                 (tests_df['RESULT_DATE'] <= current_month_end)]

# Determine data training cutoff date (get last day of previous month)
data_cutoff_date = (pd.to_datetime(output_model_date, format='%Y%m%d').replace(day=1) - pd.Timedelta(days=1))

# Filter out test data past the cutoff date
tests_df = tests_df[tests_df['RESULT_DATE'] <= data_cutoff_date]

# 인덱스 재설정
tests_df = tests_df.reset_index(drop=True)
tests_df_refmonth = tests_df_refmonth.reset_index(drop=True)

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
            'HMC_SP_60': [900, 5, 17, 0.01, 0.5, 0.5],
            'HMC_SP_80': [900, 10, 14, 0.1, 0.3, 0.3]
          }

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
    tests_df_target = helpers.remove_outliers_iqr(tests_df_target, target, 3)
    dataset_size_outliers = tests_df_target.shape

    # Add dataset size information to report
    report_output['Data processing']['text'][(target)] = {'Original dataset size' : dataset_size_original,
                                                          'Remove null values' : dataset_size_null_values,
                                                          'Remove duplicates' : dataset_size_duplicates,
                                                          'Remove target outliers' : dataset_size_outliers
                                                         }

    # Plot target data distribution
    fig, ax = plt.subplots()
    tests_df_target[target].plot(kind='hist', figsize=(6, 3), ax=ax)
    ax.set_title(f"{target} - Distribution")
    fig.tight_layout()

    # Write image to base64 bytes for the training report
    stringIObytes = io.BytesIO()
    plt.savefig(stringIObytes, format = 'png')
    stringIObytes.seek(0)
    base64_data = base64.b64encode(stringIObytes.getvalue()).decode('utf8')

    # Add target distribution plot to report
    # report_output[f'Data processing']['plot'].update({f'{target}_target_distribution': base64_data})
    report_output[f'Data processing']['plot'].setdefault(f'{target}', []).append(base64_data)
    plt.close(fig)

    # Separate input features and target feature
    X = tests_df_target[features]
    y = tests_df_target[target]

    # Split dataset into train/test sets

    # RANDOM SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

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
    if ordinal_features:  # Only include ordinal transformer if there are ordinal features
        column_transformer.append(("ord_column_transformer", ord_transform_pipeline, ordinal_features))

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

        # Get best estimator
        pipeline = random_cv.best_estimator_

    # If we just want to use default hyperparameters
    elif training_mode == "fixed_hyperparameters" :

        # Fit pipeline
        pipeline.fit(X_train, y_train)

    ## Generate model metrics and explanation data

    # Model name used for filename prefix
    model_name = target

    # Get hyperparameters used in final model
    regressor = pipeline.named_steps['regressor']
    hyperparameters = {
        'n_estimators': regressor.n_estimators,
        'min_child_weight': regressor.min_child_weight,
        'max_depth': regressor.max_depth,
        'learning_rate': regressor.learning_rate,
        'gamma': regressor.gamma,
        'colsample_bytree': regressor.colsample_bytree
    }

    # Calculate metrics for training data
    y_pred = pipeline.predict(X_train)
    train_metrics = helpers.calculate_metrics(X_train, y_train, y_pred)

    # Save datasets for training report
    Train_real_data = y_train
    Train_pred_data = y_pred

    # Calculate metrics for test data
    y_pred = pipeline.predict(X_test)
    test_metrics = helpers.calculate_metrics(X_test, y_test, y_pred)

    # Save datasets for training report
    Test_real_data = y_test
    Test_pred_data = y_pred
    
    if target == "HMC_SP_60":
        now_tolerance = 1
        now_AP = 'A'
    elif target == "HMC_SP_80":
        now_tolerance = 1
        now_AP = 'A'

    # GM-PV Format Validation Code
    # Return 받는 인자 개수가 늘어남을 고려, Dictionary 형태로 return됨.
    result_gmpv_validation = helpers.validation_gmpv_mode(y_test, y_pred, now_tolerance, now_AP)

    # return 값 : (1)성공률 개수[int], (2)성공률 퍼센트[float], (3)평균오차[char]
    now_err_boundary_ea = result_gmpv_validation['success_ea']
    now_err_boundary_percent = result_gmpv_validation['success_percentage']

    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
    mae_test = mean_absolute_error(y_test, y_pred)
    mape_test = mean_absolute_percentage_error(y_test, y_pred)
    r2_test = r2_score(y_test, y_pred)
    Test_real_data = y_test
    Test_pred_data = y_pred

    if target == "HMC_SP_80":
        diff_SP_80 = Test_real_data - Test_pred_data
        temp1 = pd.DataFrame()
        temp1['Test_real_data'] = Test_real_data
        temp1['Test_pred_data'] = Test_pred_data
        temp1['diff'] = diff_SP_80
        # 최대값 3개에 해당하는 인덱스 찾기
        max_indices = diff_SP_80.nlargest(3).index
        print(max_indices)
        df_max = tests_df_target.loc[max_indices]
        df_max_value = temp1.loc[max_indices]
        # 두 데이터프레임을 행 방향으로 합치기
        df_max_combined = pd.concat([df_max, df_max_value], axis=1)

        # 최소값 3개에 해당하는 인덱스 찾기
        min_indices = diff_SP_80.nsmallest(3).index
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

    y_pred_s = pd.Series(y_pred, index=y_test.index)
    seasons_test = tests_df.loc[list(y_test.index), "SEASON"].unique()
    report_output['Training results']['text3'].setdefault(target, {})
    for now_season in seasons_test:
        X_test_season = X_test.loc[tests_df.SEASON == now_season]
        y_test_season = y_test.loc[tests_df.SEASON == now_season]
        y__pred_season = y_pred_s.loc[tests_df.SEASON == now_season]
        result_gmpv_validation = helpers.validation_gmpv_mode(y_test_season, y__pred_season, now_tolerance,
                                                                      now_AP)
        now_Seg_err_boundary_ea = result_gmpv_validation['success_ea']
        now_Seg_err_boundary_percent = result_gmpv_validation['success_percentage']
        now_avg_error = result_gmpv_validation['average_error']
        r2_season = r2_score(y_test_season, y__pred_season)

        report_output['Training results']['text3'][target][now_season] = {
            'Dataset Size': X_test_season.shape[0],            'Success Rate[%]': helpers.format_with_circle(round(now_Seg_err_boundary_percent, 2)),
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
    plt.close(fig)

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
    plt.savefig(stringIObytes, format='png')
    stringIObytes.seek(0)
    base64_data = base64.b64encode(stringIObytes.getvalue()).decode('utf8')

    report_output['Training results']['plot'].setdefault(f'{target}+', []).append(base64_data)
    plt.close(fig)

    # Scatter plot
    plt.figure(figsize=(7, 6))
    plt.scatter(Train_real_data, Train_pred_data, color='blue',
                label=f"Train correlation ({Train_real_data.shape[0]} EA)")
    plt.scatter(Test_real_data, Test_pred_data, color='orange',
                label=f"Test correlation ({Test_real_data.shape[0]} EA), R-square {test_metrics.get('r2'):.2f}")

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

    ## Calculate metrics on all data

    y_pred = pipeline.predict(X)
    full_metrics = helpers.calculate_metrics(X, y, y_pred)

    # 기준 월로 필터링된 데이터셋으로 학습데이터셋 구성
    tests_df_target_refmonth = tests_df_refmonth[features + [target]]
    tests_df_target_refmonth = tests_df_target_refmonth.dropna()  ## ←중요! Drop NaN 꼭 해야함,,

    if not tests_df_target_refmonth.empty:
        X_refmonth = tests_df_target_refmonth[features]
        y_refmonth = tests_df_target_refmonth[target]

        # 기준월 데이터셋 예측 진행
        y_pred = pipeline.predict(X_refmonth)

        # 출력을 위한 검증결과 변수할당
        result_gmpv_validation = helpers.validation_gmpv_mode(y_refmonth, y_pred, now_tolerance, now_AP)
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

        plt.xlim(80, 105)
        plt.ylim(80, 105)

        # 상관계수 표기
        plt.title(f'Monthly Verification - {target}')
        plt.xlabel('Physics')
        plt.ylabel('Predicted')
        plt.legend()

        # Write image to base64 bytes for the training report
        stringIObytes = io.BytesIO()
        plt.savefig(stringIObytes, format='png')
        stringIObytes.seek(0)
        base64_data = base64.b64encode(stringIObytes.getvalue()).decode('utf8')

        # Add training plot to report
        # report_output['Training results']['plot'].update({f'{target}_feature_importance': base64_data})
        report_output['Training results']['plot'].setdefault(f'{target}+', []).append(base64_data)

    # Generate and save training info to JSON file
    training_info = helpers.generate_training_info(
                        training_date=output_model_date,
                        data_cutoff_date=data_cutoff_date.strftime('%Y%m%d'),
                        train_metrics=train_metrics,
                        test_metrics=test_metrics,
                        full_metrics=full_metrics,
                        hyperparameters=hyperparameters)
    with open(os.path.join(output_model_dir, f'{model_name}_training_info.json'), 'w') as f:
        json.dump(training_info, f, indent=2, cls=common_helpers.NumpyEncoder)

    ## Generate and save SHAP explanation objects

    # Define PCA feature name mappings for aggregation of SHAP values across different PCA components of a single feature
    # {Prefix of PCA components: Name of final aggregated feature}
    pca_feature_mappings = {
        'GM_TNAI_EMB_VIT_MAE_PCA_': 'PATTERN_EMBEDDINGS'
    }
    shap_explainer, shap_explanation = model_explanation.compute_shap_explanation(pipeline, X, pca_feature_mappings=pca_feature_mappings)
    joblib.dump(shap_explanation, os.path.join(output_model_dir, f'{model_name}_shap_explanation.joblib'))
    joblib.dump(shap_explainer, os.path.join(output_model_dir, f'{model_name}_shap_explainer.joblib'))

    # Add training results to report
    report_output['Training results']['text'][(target)] = {'Training dataset size' : X_train.shape[0],
                                                           'Test dataset size' : X_test.shape[0],
                                                           'Training MAE': round(train_metrics.get('mae'), 3),
                                                           'Training MAPE': round(train_metrics.get('mape'), 3),
                                                           'Training RMSE': round(train_metrics.get('rmse'), 3),
                                                           'Training R2': round(train_metrics.get('r2'), 3),
                                                           'Test MAE': round(test_metrics.get('mae'), 3),
                                                           'Test MAPE': round(test_metrics.get('mape'), 3),
                                                           'Test RMSE': round(test_metrics.get('rmse'), 3),
                                                           'Test R2': round(test_metrics.get('r2'), 3),
                                                           'Overall MAE': round(full_metrics.get('mae'), 3),
                                                           'Overall MAPE': round(full_metrics.get('mape'), 3),
                                                           'Overall RMSE': round(full_metrics.get('rmse'), 3),
                                                           'Overall R2': round(full_metrics.get('r2'), 3),
                                                           'Tolerance' : f"{round(now_tolerance,1)}%" if now_AP == 'P' else f"±{round(now_tolerance,1)}",
                                                           'Size of Under Tol' :  now_err_boundary_ea,
                                                           'Percentage of Under Tol' : helpers.format_with_circle(round(now_err_boundary_percent,2))
                                                           }

    report_output['Hyper parameters']['parameters'][(target)] = hyperparameters

    # 월간 보고 출력 양식
    if not tests_df_target_refmonth.empty:
        report_output['Training results']['text2'][(target)] = {
            'Validation Month': current_month_start.strftime('%Y-%m'),
            'Dataset Size': X_refmonth.shape[0],
            'Success Rate[%]': helpers.format_with_circle(round(refmonth_err_boundary_percent, 2)),
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
report_generation.generate_report(prediction_item, output_model_date, report_output, output_model_dir)
