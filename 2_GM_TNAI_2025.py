import streamlit as st
from PIL import Image
from io import BytesIO
import requests
import json
import os
import math
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from scipy.fftpack import fft
import cx_Oracle
import numpy as np
import _auth_config as auth
import io
import _PLM_SPEC_CHANGE  # 필요한 스펙으로 변환하는 함수 정의
import _PLM_SPEC_CHANGE_2
import _use_cols
import seaborn as sns
from datetime import date
pd.set_option('mode.chained_assignment',  None)

import random
import sklearn
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from group_lasso import GroupLasso

class IsplineGenerator_2:
    def __init__(self):
        self.knot_vec = None

    def fit(self, X):
        import numpy as np
        import pandas as pd

        train_knot_vec = []
        for i in range(X.shape[1]):
            min_i = np.min(X.iloc[:, i])
            max_i = np.max(X.iloc[:, i])
            mid_i = np.median(X.iloc[:, i])
            dx_i = (max_i - min_i) / 2

            vec_i = [min_i - dx_i, min_i, mid_i, max_i, max_i + dx_i]
            train_knot_vec.append(vec_i)

        train_knot_vec = np.array(train_knot_vec)
        self.knot_vec = train_knot_vec

    def transform(self, X):
        import numpy as np
        import pandas as pd

        #### I-spline bases for one variable function
        def I_spline_bases(X_var, knot_vec):
            I_spline = np.zeros((3, len(X_var)))

            for j in range(I_spline.shape[0]):
                I_j = I_spline[j].copy()

                knot_j = knot_vec[j]
                knot_j1 = knot_vec[j + 1]
                knot_j2 = knot_vec[j + 2]

                spline_mid_1 = ((X_var - knot_j) / ((knot_j1 - knot_j) * (knot_j2 - knot_j))) * (X_var - knot_j)
                spline_mid_2 = 1 - ((X_var - knot_j2) / ((knot_j2 - knot_j) * (knot_j2 - knot_j1))) * (
                        X_var - knot_j2)

                I_j[X_var > knot_j] = spline_mid_1[X_var > knot_j]
                I_j[X_var > knot_j1] = spline_mid_2[X_var > knot_j1]
                I_j[X_var > knot_j2] = 1

                I_spline[j] = I_j

            return (I_spline.T)

        #### I-spline bases for all data function
        def I_spline_matrix(X, knot_vec):
            X_numeric = X
            for i in range(X_numeric.shape[1]):
                Z_i = pd.DataFrame(I_spline_bases(X_numeric.iloc[:, i], knot_vec[i]))
                col_i = X_numeric.columns[i]
                Z_i.columns = [col_i + "_1", col_i + "_2", col_i + "_3"]
                if i == 0:
                    Z = Z_i
                else:
                    Z = pd.concat([Z, Z_i], axis=1)

            return (Z)

        ############################################
        Z = I_spline_matrix(X, self.knot_vec)
        return (Z)

class cv_grouplasso_fit:
    def __init__(self):
        self.cv_grouplasso = None

    def fit(self, X, y, cv, group_lst, lambda_lst):
        ## Function generation tunning best lambda
        # from group_lasso import GroupLasso
        #################################################################
        # function definition
        def group_cv(xx, yy, lambda_lst, cv, seed, group_lst):
            # import numpy as np
            # import pandas as pd
            # from sklearn.metrics import r2_score
            # from group_lasso import GroupLasso

            cv_r2 = list(np.array(lambda_lst) * 0)
            ss = seed
            xx = xx.reset_index(drop=True)
            yy = yy.reset_index(drop=True)

            for j in range(cv):

                random.seed(j * ss + 11)  # to generate different samples for each iteration
                train_sam = random.sample(range(xx.shape[0]), round(xx.shape[0] * (1 - 1 / cv)))

                X_train = np.array(xx.iloc[train_sam,])
                X_test = np.array(xx.drop(train_sam, axis=0))

                y_train = np.array(yy[train_sam]).reshape(-1, 1)
                y_test = np.array(yy.drop(train_sam)).reshape(-1, 1)

                cv_j_r2 = []

                for i in range(len(lambda_lst)):
                    gl = GroupLasso(groups=group_lst,
                                    group_reg=lambda_lst[i],
                                    l1_reg=0,
                                    frobenius_lipschitz=True,
                                    scale_reg="None",
                                    subsampling_scheme=1,
                                    supress_warning=True,
                                    n_iter=1000,
                                    tol=1e-3)

                    gl.fit(X_train, y_train)
                    yhat_test = gl.predict(X_test)
                    r2_i = r2_score(y_test, yhat_test)
                    cv_j_r2.append(r2_i)

                cv_r2 = list(np.array(cv_r2) + np.array(cv_j_r2))

            cv_r2_mean = list(np.array(cv_r2) / cv)
            lambda_best = lambda_lst[cv_r2_mean.index(max(cv_r2_mean))]
            return (lambda_best)

        ##################################################################
        # Fitting
        best_lambda = group_cv(xx=X, yy=y, cv=cv, lambda_lst=lambda_lst,
                               seed=10, group_lst=group_lst)

        gl = GroupLasso(groups=group_lst, group_reg=0.001,
                        l1_reg=0, frobenius_lipschitz=True, scale_reg="None",
                        subsampling_scheme=1, supress_warning=True, n_iter=1000, tol=1e-3)

        gl.fit(X, y)
        self.cv_grouplasso = gl

    def predict(self, X):
        yhat = self.cv_grouplasso.predict(X)
        return (yhat)

class GAM_grouplasso_RF:
    def __init__(self):
        self.I_spline = None
        self.rf_re = None
        self.gam = None
        self.dummy = None

    def fit(self, X_numeric, X_dummy, y, lambda_lst, cv):
        # sampling-> mix -> CV에서의 상관성 제거
        sam = random.sample(range(X_numeric.shape[0]), round(X_numeric.shape[0]))

        X_all = pd.concat([X_numeric, X_dummy], axis=1)

        X_all = X_all.iloc[sam,]
        y = y[sam]
        X_numeric = X_numeric.iloc[sam,]
        ## model
        I_gene = IsplineGenerator_2()
        cv_gam = cv_grouplasso_fit()
        rf_model = RandomForestRegressor()

        ## hyper parameter for RF
        param_rf = {'n_estimators': [100],
                    'max_depth': [2, 3, 5, 6, 10],
                    'min_samples_split': [3, 5, 9, 10, 13]}

        ## spline generation
        I_gene.fit(X_numeric);
        self.I_spline = I_gene  # save
        Z = I_gene.transform(X_numeric)
        spline_groups = np.repeat(range(X_numeric.shape[1]), 3).reshape(-1, 1)  # groups for group lasso

        ### group lasso GAM fitting
        cv_gam.fit(X=Z, y=y, cv=cv, lambda_lst=lambda_lst, group_lst=spline_groups)
        self.gam = cv_gam  # save

        y_hat_gam = cv_gam.predict(Z)
        res_gam = y - y_hat_gam

        ### RF for res
        grid_cv_rf = GridSearchCV(rf_model, param_grid=param_rf, cv=cv, scoring='r2', n_jobs=-1)

        grid_cv_rf.fit(X_all, res_gam)

        self.rf = grid_cv_rf.best_estimator_  # save

    def predict(self, X_numeric, X_dummy):
        X_all_test = pd.concat([X_numeric, X_dummy], axis=1)
        Z_test = self.I_spline.transform(X_numeric)

        y_hat_gam = self.gam.predict(Z_test)
        res_hat_rf = self.rf.predict(X_all_test)

        y_hat_test = y_hat_gam + res_hat_rf

        return (y_hat_test)

def get_image_from_DB(drawing_no,resize_factor):
    # image_url = f"http://10.82.79.72:9991/image/?image_type=catia_fullassy&image_name={filename}&image_extension=png"
    # resize_factor = 5 # 이미지 해상도를 0.1mm --> 0.1 * 5 = 0.5mm 로 변경
    #             http://10.82.79.72:9991/img/?img_name=RPR-210405_K127_275-40R19Y%20XL_FullAssy_2D&img_extension=png
    # image_url = f"http://10.82.79.72:9991/img/?img_name={filename}&img_extension=png"
    image_url = f"http://10.82.79.72:9998/pattern/drw_no/2.5d/{drawing_no}"
    try:
        input_image = Image.open(BytesIO(requests.get(image_url).content))
    except:
        image_url = f"http://10.82.79.72:9998/pattern/drw_no/2d/{drawing_no}"
    input_image = Image.open(BytesIO(requests.get(image_url).content))
    input_info = eval(input_image.info["Description"])

    # RGB 이미지
    try:
        a_rgb = np.array(input_image, dtype="uint8")[:, :, :3]
    except:
        a_rgb = np.reshape(np.repeat(input_image, 3), (input_image.size[1], input_image.size[0], 3))
    if resize_factor > 1:
        a_rgb = np.array(
            Image.fromarray(np.array(a_rgb, dtype="uint8")).resize(
                (
                    int(a_rgb.shape[1] / resize_factor),
                    int(a_rgb.shape[0] / resize_factor),
                ),
                Image.NEAREST,
            )
        )

    # 깊이
    a_gray = np.amax(a_rgb, axis=2)
    a_gray[a_gray > 240] = 240
    a_depth = ((240 - a_gray).astype(int)) / 10

    # 강성
    #  패턴  성능  정보  불러오기
    list_of_performance = json.loads(requests.get(f"http://10.82.79.72:9998/performance/drw_no/2.5d/{drawing_no}").text)
    if list_of_performance is None:
        list_of_performance = json.loads(
            requests.get(f"http://10.82.79.72:9998/performance/drw_no/2d/{drawing_no}").text)
    table_of_performance = pd.DataFrame(list_of_performance)

    # ITEM, VALUE 열 추출
    selected_df_perf = table_of_performance[['ITEM', 'VALUE']]
    # ITEM 열을 인덱스로 설정하고, VALUE 열을 값으로 설정
    selected_df_perf = selected_df_perf.set_index('ITEM').T

    selected_df_perf_pkx_step = selected_df_perf[['PKX_STEP']]
    # 일부 performance 데이터가 이중으로 들어간 데이터를 걸러내기 위한 작업
    if selected_df_perf_pkx_step.shape[1] == 1:
        pass
    else:
        selected_df_perf_pkx_step = selected_df_perf_pkx_step.iloc[:, :1]

    import ast

    l_pkx = ast.literal_eval(selected_df_perf_pkx_step.at['VALUE', 'PKX_STEP'])
    l_stiffness = [l_pkx[0] for i in range(round((input_image.size[1] - len(l_pkx)) / 2) - 1)]
    l_stiffness.extend(l_pkx)
    l_stiffness.extend([l_pkx[-1] for i in range(input_image.size[1] - len(l_stiffness))])
    a_stiffness = np.tile(l_stiffness, (input_image.size[0], 1)).T
    if resize_factor > 1:
        a_stiffness = np.array(
            Image.fromarray(np.array(a_stiffness, dtype="uint16")).resize(
                (
                    int(a_stiffness.shape[1] / resize_factor),
                    int(a_stiffness.shape[0] / resize_factor),
                ),
                Image.NEAREST,
            )
        )
    a_stiffness[a_gray < 240] = 0

    # 강성 2
    coe_stiffness = sum(l_stiffness) / len(l_stiffness)
    l_stiffness_normalization = [i / coe_stiffness for i in l_stiffness]
    a_stiffness_normalization = np.tile(l_stiffness_normalization, (input_image.size[0], 1)).T
    if resize_factor > 1:
        a_stiffness_normalization = np.array(
            Image.fromarray(np.array(a_stiffness_normalization * 10000000, dtype="uint32")).resize(
                (
                   int(
                       a_stiffness_normalization.shape[1] / resize_factor),
                   int(
                       a_stiffness_normalization.shape[0] / resize_factor),
                ),
               Image.NEAREST,
            )
        ) / 10000000
    a_stiffness_normalization[a_gray < 240] = 0

    return a_rgb, a_stiffness, a_stiffness_normalization

def get_image_from_manual_upload(uploaded_pattern,resize_factor):

    input_image = uploaded_pattern
    input_info = eval(input_image.info["Description"])

    # Manual pattern image 사용 시
    d_info = fn_dict_to_flat(input_info)
    t_temp = pd.DataFrame.from_dict([d_info]).rename(columns=d_table, inplace=False)
    # t_temp = pd.DataFrame(d_info, columns=d_table.keys(), index=[0]).rename(columns=d_table, inplace=False)

    # RGB 이미지
    try:
        a_rgb = np.array(input_image, dtype="uint8")[:, :, :3]
    except:
        a_rgb = np.reshape(np.repeat(input_image, 3), (input_image.size[1], input_image.size[0], 3))
    if resize_factor > 1:
        a_rgb = np.array(
            Image.fromarray(np.array(a_rgb, dtype="uint8")).resize(
                (
                    int(a_rgb.shape[1] / resize_factor),
                    int(a_rgb.shape[0] / resize_factor),
                ),
                Image.NEAREST,
            )
        )

    # 깊이
    a_gray = np.amax(a_rgb, axis=2)
    a_gray[a_gray > 240] = 240
    a_depth = ((240 - a_gray).astype(int)) / 10

    # 강성
    l_pkx = input_info["Performance"]["PatternStiffness"]["PKXstep"]
    l_stiffness = [l_pkx[0] for i in range(round((input_image.size[1] - len(l_pkx)) / 2) - 1)]
    l_stiffness.extend(l_pkx)
    l_stiffness.extend([l_pkx[-1] for i in range(input_image.size[1] - len(l_stiffness))])
    a_stiffness = np.tile(l_stiffness, (input_image.size[0], 1)).T
    if resize_factor > 1:
        a_stiffness = np.array(
            Image.fromarray(np.array(a_stiffness, dtype="uint16")).resize(
                (
                    int(a_stiffness.shape[1] / resize_factor),
                    int(a_stiffness.shape[0] / resize_factor),
                ),
                Image.NEAREST,
            )
        )
    a_stiffness[a_gray < 240] = 0

    # 강성 2
    coe_stiffness = sum(l_stiffness) / len(l_stiffness)
    l_stiffness_normalization = [i / coe_stiffness for i in l_stiffness]
    a_stiffness_normalization = np.tile(l_stiffness_normalization, (input_image.size[0], 1)).T
    if resize_factor > 1:
        a_stiffness_normalization = np.array(
            Image.fromarray(np.array(a_stiffness_normalization * 10000000, dtype="uint32")).resize(
                (
                   int(
                       a_stiffness_normalization.shape[1] / resize_factor),
                   int(
                       a_stiffness_normalization.shape[0] / resize_factor),
                ),
               Image.NEAREST,
            )
        ) / 10000000
    a_stiffness_normalization[a_gray < 240] = 0
    return a_rgb, a_stiffness, a_stiffness_normalization

def get_image_info(M_code, DRW_NO_keyin, img_pattern):
    if manual_pattern_key == 0:
        performance_url = f"http://10.82.79.72:9998/performance/product_code/2.5d/{M_code}"
        list_of_info = json.loads(requests.get(performance_url).text)
        if list_of_info is None:
            performance_url = f"http://10.82.79.72:9998/performance/product_code/2d/{M_code}"
            list_of_info = json.loads(requests.get(performance_url).text)
        if list_of_info is None:
            performance_url = f"http://10.82.79.72:9998/performance/drw_no/2d/{DRW_NO_keyin}"
            list_of_info = json.loads(requests.get(performance_url).text)
        table_of_info = pd.DataFrame(list_of_info)
        index_number_land_ratio = table_of_info.index[(table_of_info['ITEM'] == 'LAND_RATIO')]
        index_number_main_ratio = table_of_info.index[(table_of_info['ITEM'] == 'MAIN_RATIO')]
        # image에서 Land, Groove Sum ratio 추출
        a_landsea = []
        a_landsea.append(table_of_info['FILENAME'][0])
        a_landsea.append(table_of_info["VALUE"][index_number_land_ratio[0]])
        a_landsea.append(table_of_info["VALUE"][index_number_main_ratio[0]])
        a_landsea = pd.DataFrame(a_landsea)
    elif manual_pattern_key == 1:
        input_info = eval(img_pattern.info["Description"])
        a_landsea = []
        a_landsea.append(input_info["Information"]["FileName"])
        a_landsea.append(input_info["Performance"]["Info"]["Land"]["Ratio"])
        a_landsea.append(input_info["Performance"]["BlockRatio"]["GrooveSumRatio"])
        a_landsea = pd.DataFrame(a_landsea)

    return a_landsea

def fn_dict_to_flat(d_in):
    d_out = {}
    for key_1, value_1 in d_in.items():
        try:
            for key_2, value_2 in value_1.items():
                try:
                    for key_3, value_3 in value_2.items():
                        try:
                            for key_4, value_4 in value_3.items():
                                try:
                                    for key_5, value_5 in value_4.items():
                                        try:
                                            for key_6, value_6 in value_5.items():
                                                print("-----")
                                        except:
                                            d_out[
                                                "_".join(
                                                    list(
                                                        (
                                                            key_1,
                                                            key_2,
                                                            key_3,
                                                            key_4,
                                                            key_5,
                                                        )
                                                    )
                                                )
                                            ] = value_5
                                except:
                                    d_out[
                                        "_".join(list((key_1, key_2, key_3, key_4)))
                                    ] = value_4
                        except:
                            d_out["_".join(list((key_1, key_2, key_3)))] = value_3
                except:
                    d_out["_".join(list((key_1, key_2)))] = value_2
        except:
            d_out[key_1] = value_1
    return d_out

d_table = {
    "Information_Version": "VERSION",
    "Information_CreateTime": "CREATETIME",
    "Information_PC": "PC",
    "Information_ID": "ID",
    "Information_FileName": "FILENAME",
    "Information_ImageSource": "IMAGESOURCE",
    "Information_ImageType": "IMAGETYPE",
    "Information_ImageResolution": "IMAGERESOLUTION",
    "Parameters_Product_ProductCode": "PRODUCTCODE",
    "Parameters_Product_Revision": "REVISION",
    "Parameters_Product_TireSize": "TIRESIZE",
    "Parameters_Product_TireType": "TIRETYPE",
    "Parameters_Product_LoadIndex": "LOADINDEX",
    "Parameters_Product_SpeedSymbol": "SPEEDSYMBOL",
    "Parameters_Product_PatternName": "PATTERNNAME",
    "Parameters_Product_PlyRating": "PLYRATING",
    "Parameters_Profile_OveralDiameter": "OVERALDIAMETER",
    "Parameters_Profile_MainGrooveDepth": "MAINGROOVEDEPTH",
    "Parameters_Profile_TreadWidth": "TREADWIDTH",
    "Parameters_Profile_DecorationWidth": "DECORATIONWIDTH",
    "Parameters_Profile_DecorationBarWidth": "DECORATIONBARWIDTH",
    "Parameters_Profile_DecorationGrooveWidth": "DECORATIONGROOVEWIDTH",
    "Parameters_Pattern_PitchDesignType": "PITCHDESIGNTYPE",
    "Parameters_Pattern_MainGrooveType": "MAINGROOVETYPE",
    "Parameters_Pattern_KerfThickness": "KERFTHICKNESS",
    "Parameters_Pattern_PitchDesignBox_MoldShift": "MOLDSHIFT",
    "Parameters_Pattern_PitchDesignBox_PitchCenterLineOffset": "PITCHCENTERLINEOFFSET",
    "Parameters_Pattern_PitchDesignBox_Lower_BlockAngle1st": "BLOCKANGLE1ST",
    "Parameters_Pattern_PitchDesignBox_Lower_BlockAngle2nd": "BLOCKANGLE2ND",
    "Parameters_Pattern_PitchDesignBox_Lower_BlockAngle3rd": "BLOCKANGLE3RD",
    "Parameters_Pattern_PitchDesignBox_Lower_ShoulderBlockAngle": "SHOULDERBLOCKANGLE",
    "Parameters_Pattern_PitchDesignBox_Lower_LowerDecorationBlockAngle": "LOWERDECORATIONBLOCKANGLE",
    "Parameters_Pattern_PitchDesignBox_Lower_BlockShift1st": "BLOCKSHIFT1ST",
    "Parameters_Pattern_PitchDesignBox_Lower_BlockShift2nd": "BLOCKSHIFT2ND",
    "Parameters_Pattern_PitchDesignBox_Lower_BlockShift3rd": "BLOCKSHIFT3RD",
    "Parameters_Pattern_PitchDesignBox_Lower_CirCenterGrooveWidth": "CIRCENTERGROOVEWIDTH",
    "Parameters_Pattern_PitchDesignBox_Lower_Cir1stGrooveDistance": "CIR1STGROOVEDISTANCE",
    "Parameters_Pattern_PitchDesignBox_Lower_Cir1stGrooveWidth": "CIR1STGROOVEWIDTH",
    "Parameters_Pattern_PitchDesignBox_Lower_Cir2ndGrooveDistance": "CIR2NDGROOVEDISTANCE",
    "Parameters_Pattern_PitchDesignBox_Lower_Cir2ndGrooveWidth": "CIR2NDGROOVEWIDTH",
    "Parameters_Pattern_PitchDesignBox_Lower_Cir3rdGrooveDistance": "CIR3RDGROOVEDISTANCE",
    "Parameters_Pattern_PitchDesignBox_Lower_Cir3rdGrooveWidth": "CIR3RDGROOVEWIDTH",
    "Parameters_Pattern_PitchDesignBox_Upper_UpperBlockAngle1st": "UPPERBLOCKANGLE1ST",
    "Parameters_Pattern_PitchDesignBox_Upper_UpperBlockAngle2nd": "UPPERBLOCKANGLE2ND",
    "Parameters_Pattern_PitchDesignBox_Upper_UpperBlockAngle3rd": "UPPERBLOCKANGLE3RD",
    "Parameters_Pattern_PitchDesignBox_Upper_UpperDecorationBlockAngle": "UPPERDECORATIONBLOCKANGLE",
    "Parameters_Pattern_PitchDesignBox_Upper_UpperShoulderBlockAngle": "UPPERSHOULDERBLOCKANGLE",
    "Parameters_Pattern_PitchDesignBox_Upper_UpperBlockShift1st": "UPPERBLOCKSHIFT1ST",
    "Parameters_Pattern_PitchDesignBox_Upper_UpperBlockShift2nd": "UPPERBLOCKSHIFT2ND",
    "Parameters_Pattern_PitchDesignBox_Upper_UpperBlockShift3rd": "UPPERBLOCKSHIFT3RD",
    "Parameters_Pattern_PitchDesignBox_Upper_UpperCir1stGrooveDistance": "UPPERCIR1STGROOVEDISTANCE",
    "Parameters_Pattern_PitchDesignBox_Upper_UpperCir1stGrooveWidth": "UPPERCIR1STGROOVEWIDTH",
    "Parameters_Pattern_PitchDesignBox_Upper_UpperCir2ndGrooveDistance": "UPPERCIR2NDGROOVEDISTANCE",
    "Parameters_Pattern_PitchDesignBox_Upper_UpperCir2ndGrooveWidth": "UPPERCIR2NDGROOVEWIDTH",
    "Parameters_Pattern_PitchDesignBox_Upper_UpperCir3rdGrooveDistance": "UPPERCIR3RDGROOVEDISTANCE",
    "Parameters_Pattern_PitchDesignBox_Upper_UpperCir3rdGrooveWidth": "UPPERCIR3RDGROOVEWIDTH",
    "Parameters_Pattern_PitchDesignBox_Calculated_BWL1": "BWL1",
    "Parameters_Pattern_PitchDesignBox_Calculated_BWL2": "BWL2",
    "Parameters_Pattern_PitchDesignBox_Calculated_BWL3": "BWL3",
    "Parameters_Pattern_PitchDesignBox_Calculated_BWLShoulder": "BWLSHOULDER",
    "Parameters_Pattern_PitchDesignBox_Calculated_BWLDecoration": "BWLDECORATION",
    "Parameters_Pattern_PitchDesignBox_Calculated_BWU1": "BWU1",
    "Parameters_Pattern_PitchDesignBox_Calculated_BWU2": "BWU2",
    "Parameters_Pattern_PitchDesignBox_Calculated_BWU3": "BWU3",
    "Parameters_Pattern_PitchDesignBox_Calculated_BWUShoulder": "BWUSHOULDER",
    "Parameters_Pattern_PitchDesignBox_Calculated_BWUDecoration": "BWUDECORATION",
    "Parameters_Pattern_PitchDesignBox_Calculated_UpperInterAngle1": "UPPERINTERANGLE1",
    "Parameters_Pattern_PitchDesignBox_Calculated_UpperInterAngle2": "UPPERINTERANGLE2",
    "Parameters_Pattern_PitchDesignBox_Calculated_UpperInterAngle3": "UPPERINTERANGLE3",
    "Parameters_Pattern_PitchDesignBox_Calculated_UpperInterAngleTW": "UPPERINTERANGLETW",
    "Parameters_Pattern_PitchDesignBox_Calculated_LowerInterAngle1": "LOWERINTERANGLE1",
    "Parameters_Pattern_PitchDesignBox_Calculated_LowerInterAngle2": "LOWERINTERANGLE2",
    "Parameters_Pattern_PitchDesignBox_Calculated_LowerInterAngle3": "LOWERINTERANGLE3",
    "Parameters_Pattern_PitchDesignBox_Calculated_LowerInterAngleTW": "LOWERINTERANGLETW",
    "Parameters_Pattern_PitchExpansion_B1_PitchNumber": "B1_PITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_B1_NickNameOfPitchNumber": "B1_NICKNAMEOFPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_B1_PitchLength": "B1_PITCHLENGTH",
    "Parameters_Pattern_PitchExpansion_B1_PitchBorderGeometryType": "B1_PITCHBORDERGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_B1_PitchDesignGeometryType": "B1_PITCHDESIGNGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_B1_UpperPitchNumber": "B1_UPPERPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_B2_PitchNumber": "B2_PITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_B2_NickNameOfPitchNumber": "B2_NICKNAMEOFPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_B2_PitchLength": "B2_PITCHLENGTH",
    "Parameters_Pattern_PitchExpansion_B2_PitchBorderGeometryType": "B2_PITCHBORDERGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_B2_PitchDesignGeometryType": "B2_PITCHDESIGNGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_B2_UpperPitchNumber": "B2_UPPERPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_B3_PitchNumber": "B3_PITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_B3_NickNameOfPitchNumber": "B3_NICKNAMEOFPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_B3_PitchLength": "B3_PITCHLENGTH",
    "Parameters_Pattern_PitchExpansion_B3_PitchBorderGeometryType": "B3_PITCHBORDERGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_B3_PitchDesignGeometryType": "B3_PITCHDESIGNGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_B3_UpperPitchNumber": "B3_UPPERPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_B4_PitchNumber": "B4_PITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_B4_NickNameOfPitchNumber": "B4_NICKNAMEOFPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_B4_PitchLength": "B4_PITCHLENGTH",
    "Parameters_Pattern_PitchExpansion_B4_PitchBorderGeometryType": "B4_PITCHBORDERGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_B4_PitchDesignGeometryType": "B4_PITCHDESIGNGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_B4_UpperPitchNumber": "B4_UPPERPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_B5_PitchNumber": "B5_PITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_B5_NickNameOfPitchNumber": "B5_NICKNAMEOFPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_B5_PitchLength": "B5_PITCHLENGTH",
    "Parameters_Pattern_PitchExpansion_B5_PitchBorderGeometryType": "B5_PITCHBORDERGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_B5_PitchDesignGeometryType": "B5_PITCHDESIGNGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_B5_UpperPitchNumber": "B5_UPPERPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_B6_PitchNumber": "B6_PITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_B6_NickNameOfPitchNumber": "B6_NICKNAMEOFPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_B6_PitchLength": "B6_PITCHLENGTH",
    "Parameters_Pattern_PitchExpansion_B6_PitchBorderGeometryType": "B6_PITCHBORDERGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_B6_PitchDesignGeometryType": "B6_PITCHDESIGNGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_B6_UpperPitchNumber": "B6_UPPERPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_B7_PitchNumber": "B7_PITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_B7_NickNameOfPitchNumber": "B7_NICKNAMEOFPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_B7_PitchLength": "B7_PITCHLENGTH",
    "Parameters_Pattern_PitchExpansion_B7_PitchBorderGeometryType": "B7_PITCHBORDERGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_B7_PitchDesignGeometryType": "B7_PITCHDESIGNGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_B7_UpperPitchNumber": "B7_UPPERPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_B8_PitchNumber": "B8_PITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_B8_NickNameOfPitchNumber": "B8_NICKNAMEOFPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_B8_PitchLength": "B8_PITCHLENGTH",
    "Parameters_Pattern_PitchExpansion_B8_PitchBorderGeometryType": "B8_PITCHBORDERGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_B8_PitchDesignGeometryType": "B8_PITCHDESIGNGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_B8_UpperPitchNumber": "B8_UPPERPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_B9_PitchNumber": "B9_PITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_B9_NickNameOfPitchNumber": "B9_NICKNAMEOFPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_B9_PitchLength": "B9_PITCHLENGTH",
    "Parameters_Pattern_PitchExpansion_B9_PitchBorderGeometryType": "B9_PITCHBORDERGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_B9_PitchDesignGeometryType": "B9_PITCHDESIGNGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_B9_UpperPitchNumber": "B9_UPPERPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_BA_PitchNumber": "BA_PITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_BA_NickNameOfPitchNumber": "BA_NICKNAMEOFPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_BA_PitchLength": "BA_PITCHLENGTH",
    "Parameters_Pattern_PitchExpansion_BA_PitchBorderGeometryType": "BA_PITCHBORDERGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_BA_PitchDesignGeometryType": "BA_PITCHDESIGNGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_BA_UpperPitchNumber": "BA_UPPERPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_L1_PitchNumber": "L1_PITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_L1_NickNameOfPitchNumber": "L1_NICKNAMEOFPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_L1_PitchLength": "L1_PITCHLENGTH",
    "Parameters_Pattern_PitchExpansion_L1_PitchBorderGeometryType": "L1_PITCHBORDERGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_L1_PitchDesignGeometryType": "L1_PITCHDESIGNGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_L1_UpperPitchNumber": "L1_UPPERPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_L2_PitchNumber": "L2_PITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_L2_NickNameOfPitchNumber": "L2_NICKNAMEOFPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_L2_PitchLength": "L2_PITCHLENGTH",
    "Parameters_Pattern_PitchExpansion_L2_PitchBorderGeometryType": "L2_PITCHBORDERGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_L2_PitchDesignGeometryType": "L2_PITCHDESIGNGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_L2_UpperPitchNumber": "L2_UPPERPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_L3_PitchNumber": "L3_PITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_L3_NickNameOfPitchNumber": "L3_NICKNAMEOFPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_L3_PitchLength": "L3_PITCHLENGTH",
    "Parameters_Pattern_PitchExpansion_L3_PitchBorderGeometryType": "L3_PITCHBORDERGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_L3_PitchDesignGeometryType": "L3_PITCHDESIGNGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_L3_UpperPitchNumber": "L3_UPPERPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_L4_PitchNumber": "L4_PITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_L4_NickNameOfPitchNumber": "L4_NICKNAMEOFPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_L4_PitchLength": "L4_PITCHLENGTH",
    "Parameters_Pattern_PitchExpansion_L4_PitchBorderGeometryType": "L4_PITCHBORDERGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_L4_PitchDesignGeometryType": "L4_PITCHDESIGNGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_L4_UpperPitchNumber": "L4_UPPERPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_L5_PitchNumber": "L5_PITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_L5_NickNameOfPitchNumber": "L5_NICKNAMEOFPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_L5_PitchLength": "L5_PITCHLENGTH",
    "Parameters_Pattern_PitchExpansion_L5_PitchBorderGeometryType": "L5_PITCHBORDERGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_L5_PitchDesignGeometryType": "L5_PITCHDESIGNGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_L5_UpperPitchNumber": "L5_UPPERPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_L6_PitchNumber": "L6_PITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_L6_NickNameOfPitchNumber": "L6_NICKNAMEOFPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_L6_PitchLength": "L6_PITCHLENGTH",
    "Parameters_Pattern_PitchExpansion_L6_PitchBorderGeometryType": "L6_PITCHBORDERGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_L6_PitchDesignGeometryType": "L6_PITCHDESIGNGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_L6_UpperPitchNumber": "L6_UPPERPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_L7_PitchNumber": "L7_PITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_L7_NickNameOfPitchNumber": "L7_NICKNAMEOFPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_L7_PitchLength": "L7_PITCHLENGTH",
    "Parameters_Pattern_PitchExpansion_L7_PitchBorderGeometryType": "L7_PITCHBORDERGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_L7_PitchDesignGeometryType": "L7_PITCHDESIGNGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_L7_UpperPitchNumber": "L7_UPPERPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_L8_PitchNumber": "L8_PITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_L8_NickNameOfPitchNumber": "L8_NICKNAMEOFPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_L8_PitchLength": "L8_PITCHLENGTH",
    "Parameters_Pattern_PitchExpansion_L8_PitchBorderGeometryType": "L8_PITCHBORDERGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_L8_PitchDesignGeometryType": "L8_PITCHDESIGNGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_L8_UpperPitchNumber": "L8_UPPERPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_L9_PitchNumber": "L9_PITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_L9_NickNameOfPitchNumber": "L9_NICKNAMEOFPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_L9_PitchLength": "L9_PITCHLENGTH",
    "Parameters_Pattern_PitchExpansion_L9_PitchBorderGeometryType": "L9_PITCHBORDERGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_L9_PitchDesignGeometryType": "L9_PITCHDESIGNGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_L9_UpperPitchNumber": "L9_UPPERPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_LA_PitchNumber": "LA_PITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_LA_NickNameOfPitchNumber": "LA_NICKNAMEOFPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_LA_PitchLength": "LA_PITCHLENGTH",
    "Parameters_Pattern_PitchExpansion_LA_PitchBorderGeometryType": "LA_PITCHBORDERGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_LA_PitchDesignGeometryType": "LA_PITCHDESIGNGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_LA_UpperPitchNumber": "LA_UPPERPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_U1_PitchNumber": "U1_PITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_U1_NickNameOfPitchNumber": "U1_NICKNAMEOFPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_U1_PitchLength": "U1_PITCHLENGTH",
    "Parameters_Pattern_PitchExpansion_U1_PitchBorderGeometryType": "U1_PITCHBORDERGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_U1_PitchDesignGeometryType": "U1_PITCHDESIGNGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_U1_UpperPitchNumber": "U1_UPPERPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_U2_PitchNumber": "U2_PITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_U2_NickNameOfPitchNumber": "U2_NICKNAMEOFPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_U2_PitchLength": "U2_PITCHLENGTH",
    "Parameters_Pattern_PitchExpansion_U2_PitchBorderGeometryType": "U2_PITCHBORDERGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_U2_PitchDesignGeometryType": "U2_PITCHDESIGNGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_U2_UpperPitchNumber": "U2_UPPERPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_U3_PitchNumber": "U3_PITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_U3_NickNameOfPitchNumber": "U3_NICKNAMEOFPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_U3_PitchLength": "U3_PITCHLENGTH",
    "Parameters_Pattern_PitchExpansion_U3_PitchBorderGeometryType": "U3_PITCHBORDERGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_U3_PitchDesignGeometryType": "U3_PITCHDESIGNGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_U3_UpperPitchNumber": "U3_UPPERPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_U4_PitchNumber": "U4_PITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_U4_NickNameOfPitchNumber": "U4_NICKNAMEOFPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_U4_PitchLength": "U4_PITCHLENGTH",
    "Parameters_Pattern_PitchExpansion_U4_PitchBorderGeometryType": "U4_PITCHBORDERGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_U4_PitchDesignGeometryType": "U4_PITCHDESIGNGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_U4_UpperPitchNumber": "U4_UPPERPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_U5_PitchNumber": "U5_PITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_U5_NickNameOfPitchNumber": "U5_NICKNAMEOFPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_U5_PitchLength": "U5_PITCHLENGTH",
    "Parameters_Pattern_PitchExpansion_U5_PitchBorderGeometryType": "U5_PITCHBORDERGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_U5_PitchDesignGeometryType": "U5_PITCHDESIGNGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_U5_UpperPitchNumber": "U5_UPPERPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_U6_PitchNumber": "U6_PITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_U6_NickNameOfPitchNumber": "U6_NICKNAMEOFPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_U6_PitchLength": "U6_PITCHLENGTH",
    "Parameters_Pattern_PitchExpansion_U6_PitchBorderGeometryType": "U6_PITCHBORDERGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_U6_PitchDesignGeometryType": "U6_PITCHDESIGNGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_U6_UpperPitchNumber": "U6_UPPERPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_U7_PitchNumber": "U7_PITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_U7_NickNameOfPitchNumber": "U7_NICKNAMEOFPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_U7_PitchLength": "U7_PITCHLENGTH",
    "Parameters_Pattern_PitchExpansion_U7_PitchBorderGeometryType": "U7_PITCHBORDERGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_U7_PitchDesignGeometryType": "U7_PITCHDESIGNGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_U7_UpperPitchNumber": "U7_UPPERPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_U8_PitchNumber": "U8_PITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_U8_NickNameOfPitchNumber": "U8_NICKNAMEOFPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_U8_PitchLength": "U8_PITCHLENGTH",
    "Parameters_Pattern_PitchExpansion_U8_PitchBorderGeometryType": "U8_PITCHBORDERGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_U8_PitchDesignGeometryType": "U8_PITCHDESIGNGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_U8_UpperPitchNumber": "U8_UPPERPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_U9_PitchNumber": "U9_PITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_U9_NickNameOfPitchNumber": "U9_NICKNAMEOFPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_U9_PitchLength": "U9_PITCHLENGTH",
    "Parameters_Pattern_PitchExpansion_U9_PitchBorderGeometryType": "U9_PITCHBORDERGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_U9_PitchDesignGeometryType": "U9_PITCHDESIGNGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_U9_UpperPitchNumber": "U9_UPPERPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_UA_PitchNumber": "UA_PITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_UA_NickNameOfPitchNumber": "UA_NICKNAMEOFPITCHNUMBER",
    "Parameters_Pattern_PitchExpansion_UA_PitchLength": "UA_PITCHLENGTH",
    "Parameters_Pattern_PitchExpansion_UA_PitchBorderGeometryType": "UA_PITCHBORDERGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_UA_PitchDesignGeometryType": "UA_PITCHDESIGNGEOMETRYTYPE",
    "Parameters_Pattern_PitchExpansion_UA_UpperPitchNumber": "UA_UPPERPITCHNUMBER",
    "Parameters_Pattern_PitchDesignGeometryClass_Both": "BOTH",
    "Parameters_Pattern_PitchDesignGeometryClass_Lower": "LOWER",
    "Parameters_Pattern_PitchDesignGeometryClass_Upper": "UPPER",
    "Parameters_PitchAssy_NumberOfPitchSizes": "NUMBEROFPITCHSIZES",
    "Parameters_PitchAssy_NumberOfPitches": "NUMBEROFPITCHES",
    "Parameters_PitchAssy_PitchSequence": "PITCHSEQUENCE",
    "Parameters_PitchAssy_NumberOfUpperPitchSizes": "NUMBEROFUPPERPITCHSIZES",
    "Parameters_PitchAssy_NumberOfUpperPitches": "NUMBEROFUPPERPITCHES",
    "Parameters_PitchAssy_UpperPitchSequence": "UPPERPITCHSEQUENCE",
    "Parameters_Etc_NumberOfUpperTreadSubBlocks": "NUMBEROFUPPERTREADSUBBLOCKS",
    "Parameters_Etc_NumberOfLowerTreadSubBlocks": "NUMBEROFLOWERTREADSUBBLOCKS",
    "Parameters_Etc_PitchDesignUpperLower": "PITCHDESIGNUPPERLOWER",
    "Spec_DrawingNo": "DRAWING_NO",
    "Spec_ProductCode": "PRODUCT_CODE",
    "Spec_ECNNo": "ECN_NO",
    "Spec_ProductLine": "PRODUCT_LINE",
    "Spec_Segment": "SEGMENT",
    "Spec_SegmentETC": "SEGMENT_ETC",
    "Spec_TireGroup": "TIRE_GROUP",
    "Spec_OriginalCustomer": "ORIGINAL_CUSTOMER",
    "Spec_CurrentCustomer": "CURRENT_CUSTOMER",
    "Spec_Season": "SEASON",
    "Spec_Pattern": "PATTERN",
    "Spec_SizeCommercial": "SIZE_COMMERCIAL",
    "Spec_SizeMetric": "SIZE_METRIC",
    "Spec_SizeNSW": "SIZE_NSW",
    "Spec_SizeSeries": "SIZE_SERIES",
    "Spec_SizeInch": "SIZE_INCH",
    "Spec_SizeSS": "SIZE_SS",
    "Spec_SizeXL": "SIZE_XL",
    "Spec_MoldOD": "MOLD_OD",
    "Spec_MoldSH": "MOLD_SH",
    "Spec_MoldSW": "MOLD_SW",
    "Spec_MoldRW": "MOLD_RW",
    "Spec_MoldSD": "MOLD_SD",
    "Spec_MoldTW": "MOLD_TW",
    "Performance_Info_PitchLength_Pixel": "INFO_PITCHLENGTH_PIXEL",
    "Performance_Info_PitchLength_mm": "INFO_PITCHLENGTH_MM",
    "Performance_Info_PitchWidth_Pixel": "INFO_PITCHWIDTH_PIXEL",
    "Performance_Info_PitchWidth_mm": "INFO_PITCHWIDTH_MM",
    "Performance_Info_TWWidth_Pixel": "INFO_TWWIDTH_PIXEL",
    "Performance_Info_TWWidth_mm": "INFO_TWWIDTH_MM",
    "Performance_Info_DecoWidth_Pixel": "INFO_DECOWIDTH_PIXEL",
    "Performance_Info_DecoWidth_mm": "INFO_DECOWIDTH_MM",
    "Performance_Info_Sea_Depth": "INFO_SEA_DEPTH",
    "Performance_Info_Sea_Pixel": "INFO_SEA_PIXEL",
    "Performance_Info_Sea_Ratio": "INFO_SEA_RATIO",
    "Performance_Info_Land_Depth": "INFO_LAND_DEPTH",
    "Performance_Info_Land_Pixel": "INFO_LAND_PIXEL",
    "Performance_Info_Land_Ratio": "INFO_LAND_RATIO",
    "Performance_Info_Main_Depth": "INFO_MAIN_DEPTH",
    "Performance_Info_Main_Pixel": "INFO_MAIN_PIXEL",
    "Performance_Info_Main_Ratio": "INFO_MAIN_RATIO",
    "Performance_Info_Groove_Depth": "INFO_GROOVE_DEPTH",
    "Performance_Info_Groove_Pixel": "INFO_GROOVE_PIXEL",
    "Performance_Info_Groove_Ratio": "INFO_GROOVE_RATIO",
    "Performance_Info_Semi_Depth": "INFO_SEMI_DEPTH",
    "Performance_Info_Semi_Pixel": "INFO_SEMI_PIXEL",
    "Performance_Info_Semi_Ratio": "INFO_SEMI_RATIO",
    "Performance_Info_Kerf_Depth": "INFO_KERF_DEPTH",
    "Performance_Info_Kerf_Pixel": "INFO_KERF_PIXEL",
    "Performance_Info_Kerf_Ratio": "INFO_KERF_RATIO",
    "Performance_Info_Chamfer_Depth": "INFO_CHAMFER_DEPTH",
    "Performance_Info_Chamfer_Pixel": "INFO_CHAMFER_PIXEL",
    "Performance_Info_Chamfer_Ratio": "INFO_CHAMFER_RATIO",
    "Performance_Info_Line_Depth": "INFO_LINE_DEPTH",
    "Performance_Info_Line_Pixel": "INFO_LINE_PIXEL",
    "Performance_Info_Line_Ratio": "INFO_LINE_RATIO",
    "Performance_Info_Etc_Depth": "INFO_ETC_DEPTH",
    "Performance_Info_Etc_Pixel": "INFO_ETC_PIXEL",
    "Performance_Info_Etc_Ratio": "INFO_ETC_RATIO",
    "Performance_BlockRatio_MainGroovePosition": "MAIN_GROOVE_POSITION",
    "Performance_BlockRatio_NoOfMainGroove": "NO_OF_MAIN_GROOVE",
    "Performance_BlockRatio_IsCenterGroove": "IS_CENTER_GROOVE",
    "Performance_BlockRatio_IsGroove": "IS_GROOVE",
    "Performance_BlockRatio_WidthPixel": "GROOVE_WIDTH_PIXEL",
    "Performance_BlockRatio_WidthMm": "GROOVE_WIDTH_MM",
    "Performance_BlockRatio_WidthRatio": "GROOVE_WIDTH_RATIO",
    "Performance_BlockRatio_BlockSumMm": "BLOCK_SUM_MM",
    "Performance_BlockRatio_BlockSumRatio": "BLOCK_SUM_RATIO",
    "Performance_BlockRatio_GrooveSumMm": "GROOVE_SUM_MM",
    "Performance_BlockRatio_GrooveSumRatio": "GROOVE_SUM_RATIO",
    "Performance_Profile_Depth_Full": "PROFILE_DEPTH_FULL",
    "Performance_Profile_Depth_Max": "PROFILE_DEPTH_MAX",
    "Performance_Profile_Depth_Mean": "PROFILE_DEPTH_MEAN",
    "Performance_GrooveWandering_A_RoadPitch": "GROOVEWANDERING_A_PITCH",
    "Performance_GrooveWandering_A_RoadWidth": "GROOVEWANDERING_A_WIDTH",
    "Performance_GrooveWandering_A_MaxRange": "GROOVEWANDERING_A_MAX_RANGE",
    "Performance_GrooveWandering_A_GMUTS": "GROOVEWANDERING_A_GMUTS",
    "Performance_GrooveWandering_A_Check": "GROOVEWANDERING_A_CHECK",
    "Performance_GrooveWandering_B_RoadPitch": "GROOVEWANDERING_B_PITCH",
    "Performance_GrooveWandering_B_RoadWidth": "GROOVEWANDERING_B_WIDTH",
    "Performance_GrooveWandering_B_MaxRange": "GROOVEWANDERING_B_MAX_RANGE",
    "Performance_GrooveWandering_B_GMUTS": "GROOVEWANDERING_B_GMUTS",
    "Performance_GrooveWandering_B_Check": "GROOVEWANDERING_B_CHECK",
    "Performance_GrooveWandering_C_RoadPitch": "GROOVEWANDERING_C_PITCH",
    "Performance_GrooveWandering_C_RoadWidth": "GROOVEWANDERING_C_WIDTH",
    "Performance_GrooveWandering_C_MaxRange": "GROOVEWANDERING_C_MAX_RANGE",
    "Performance_GrooveWandering_C_GMUTS": "GROOVEWANDERING_C_GMUTS",
    "Performance_GrooveWandering_C_Check": "GROOVEWANDERING_C_CHECK",
    "Performance_GrooveWandering_D_RoadPitch": "GROOVEWANDERING_D_PITCH",
    "Performance_GrooveWandering_D_RoadWidth": "GROOVEWANDERING_D_WIDTH",
    "Performance_GrooveWandering_D_MaxRange": "GROOVEWANDERING_D_MAX_RANGE",
    "Performance_GrooveWandering_D_GMUTS": "GROOVEWANDERING_D_GMUTS",
    "Performance_GrooveWandering_D_Check": "GROOVEWANDERING_D_CHECK",
    "Performance_GrooveWandering_E_RoadPitch": "GROOVEWANDERING_E_PITCH",
    "Performance_GrooveWandering_E_RoadWidth": "GROOVEWANDERING_E_WIDTH",
    "Performance_GrooveWandering_E_MaxRange": "GROOVEWANDERING_E_MAX_RANGE",
    "Performance_GrooveWandering_E_GMUTS": "GROOVEWANDERING_E_GMUTS",
    "Performance_GrooveWandering_E_Check": "GROOVEWANDERING_E_CHECK",
    "Performance_GrooveWandering_F_RoadPitch": "GROOVEWANDERING_F_PITCH",
    "Performance_GrooveWandering_F_RoadWidth": "GROOVEWANDERING_F_WIDTH",
    "Performance_GrooveWandering_F_MaxRange": "GROOVEWANDERING_F_MAX_RANGE",
    "Performance_GrooveWandering_F_GMUTS": "GROOVEWANDERING_F_GMUTS",
    "Performance_GrooveWandering_F_Check": "GROOVEWANDERING_F_CHECK",
    "Performance_GrooveWandering_LA_RoadPitch": "GROOVEWANDERING_LA_PITCH",
    "Performance_GrooveWandering_LA_RoadWidth": "GROOVEWANDERING_LA_WIDTH",
    "Performance_GrooveWandering_LA_MaxRange": "GROOVEWANDERING_LA_MAX_RANGE",
    "Performance_GrooveWandering_LA_GMUTS": "GROOVEWANDERING_LA_GMUTS",
    "Performance_GrooveWandering_LA_Check": "GROOVEWANDERING_LA_CHECK",
    "Performance_GrooveWandering_Inje_RoadPitch": "GROOVEWANDERING_INJE_PITCH",
    "Performance_GrooveWandering_Inje_RoadWidth": "GROOVEWANDERING_INJE_WIDTH",
    "Performance_GrooveWandering_Inje_MaxRange": "GROOVEWANDERING_INJE_MAX_RANGE",
    "Performance_GrooveWandering_Inje_GMUTS": "GROOVEWANDERING_INJE_GMUTS",
    "Performance_GrooveWandering_Inje_Check": "GROOVEWANDERING_INJE_CHECK",
    "Performance_Hydroplaning_Vcr": "HYDROPLANING_VCR",
    "Performance_SnowTraction_TPI": "SNOWTRACTION_TPI",
    "Performance_SnowTraction_TPINoKerf": "SNOWTRACTION_TPI_NO_KERF",
    "Performance_PatternStiffness_ResizeFactor": "PATTERNSTIFFNESS_RESIZE_FACTOR",
    "Performance_PatternStiffness_IsChamfer": "PATTERNSTIFFNESS_IS_CHAMFER",
    "Performance_PatternStiffness_PKX": "PATTERNSTIFFNESS_PKX",
    "Performance_PatternStiffness_PKY": "PATTERNSTIFFNESS_PKY",
    "Performance_PatternStiffness_PKXn": "PATTERNSTIFFNESS_PKX_N",
    "Performance_PatternStiffness_PKYn": "PATTERNSTIFFNESS_PKY_N",
    "Performance_PatternStiffness_PKXstd": "PATTERNSTIFFNESS_PKX_STD",
    "Performance_PatternStiffness_PKYstd": "PATTERNSTIFFNESS_PKY_STD",
    "Performance_PatternStiffness_PKXstep": "PATTERNSTIFFNESS_PKX_STEP",
    "Performance_PatternStiffness_PKYstep": "PATTERNSTIFFNESS_PKY_STEP",
    "Performance_PatternStiffness_BKXIs": "PATTERNSTIFFNESS_BKX_IS",
    "Performance_PatternStiffness_BKXPos": "PATTERNSTIFFNESS_BKX_POS",
    "Performance_PatternStiffness_BKXStiffness": "PATTERNSTIFFNESS_BKX_STIFFNESS",
    "Performance_PatternStiffness_BKXRatio": "PATTERNSTIFFNESS_BKX_R",
    "Performance_PatternStiffness_BKXStep": "PATTERNSTIFFNESS_BKX_STEP",
    "Performance_PatternStiffness_BKXStepIs": "PATTERNSTIFFNESS_BKX_STEP_IS",
    "Performance_PatternStiffness_BKXStepRatio": "PATTERNSTIFFNESS_BKX_STEP_R",
    "Performance_PatternStiffness_PSF": "PATTERNSTIFFNESS_PSF",
    "Performance_PatternStiffness_CSR": "PATTERNSTIFFNESS_CSR",
    "Performance_PatternStiffness_CSR2": "PATTERNSTIFFNESS_CSR2",
    "Performance_ATScore_Area101to400": "ATSCORE_AREA101TO400",
    "Performance_ATScore_Area201to500": "ATSCORE_AREA201TO500",
    "Performance_ATScore_PatternContribution": "ATSCORE_PATTERNCONTRIBUTION",
    # 'Performance_IrregularWear':'IRREGULARWEAR_MAP'
}

def Data_Cleaning(df0,air,test_load,rim_width,use_cols,scale_cols):
    # 스펙 변환
    # use_cols_ = _use_cols.cols4
    # use_cols = _use_cols.cols4_1
    df1 = df0.copy()
    temp = []
    for name12 in df1['CTB_STEP_GAUGE']:
        if name12:  # 비어 있는지 여부를 확인
            name12 = df1['CTB_STEP_GAUGE'].str.split('-').apply(lambda x: [i for i in x if i])
            name12 = name12.apply(lambda x: [float(i) for i in x])
            name12 = [float(i) for i in name12[0]]
            temp = max(name12)
        else:
            temp.append('')
    df1['CTB_STEP_GAUGE'] = temp

    ###################################################
    # 문자열을 단순화 하기
    temp = []
    for i in range(df1.shape[0]):
        if df1['JLC_MATERIAL'][i] is not None:
            jlc_mat = df1['JLC_MATERIAL'][i]
            temp.append(jlc_mat[0])
        elif df1['RBT_MATERIAL'][i] is not None:
            jlc_mat = df1['RBT_MATERIAL'][i]
            temp.append(jlc_mat[0])
    df1['JLC_MATERIAL'] = temp
    #
    # ['SIZE_NSW_MM', 'SIZE_SERIES_MM', 'SIZE_INCH', 'PATTERN', 'CTB_STEP_GAUGE', 'JLC_MATERIAL', 'PRE_BELT_TYPE',
    #  'BT1_MATERIAL', 'BT1_WIDTH', 'BT1_ANGLE', 'BT1_EPI', 'PLY_LOCK', 'C01_MATERIAL', 'C01_EPI', 'FIL_TYPE',
    #  'FIL_COMPOUND', 'FIL_WIDTH',
    #  'AIR', 'TEST_LOAD', 'RIM_WIDTH', 'PATT_PAR1', 'PATT_PAR2', 'PKX', 'PKY', 'HS', 'M10']

    df4 = df1
    for column in df4.columns:
        if column not in use_cols + ['CTB_COMPOUND']:
            df4.drop(columns=[column], inplace=True)

    df4["SPEC_NO"] = df0["SPEC_NO"]
    df4['PRODUCT_CODE'] = [x[4:11] for x in df4.copy()['SPEC_NO']]
    product_cd = df4['PRODUCT_CODE'][0]
    spec_no = df4['SPEC_NO'][0]

    #############################################################################
    #  패턴  특성  추출값  불러오기  (  CNN  )
    df = pd.read_parquet("http://10.82.79.72:9998/gzip/PATTERN")
    # df = df.dropna(subset=['SPEC_NO'])
    # df.to_csv('./df.csv')
    # PRODUCT_CODE 포함여부 체크 : df[df['PRODUCT_CODE']==1014801]

    #  컬럼  정보
    #  EMB_PERFORMANCE_0001  : LAND RATIO
    #  EMB_PERFORMANCE_0002  : TPI (Snow index, 높을수록 Snow 성능에 유리, Lateral/Kerf void에 따라 결정)
    #  EMB_PERFORMANCE_0003  : PKX_N
    #  EMB_PERFORMANCE_0004  : PKY_N
    #  EMB_PERFORMANCE_0005  : PSF (PKY/PKX)
    #  EMB_PERFORMANCE_0006  : CSR2 (Center 대비 Shoulder 강성 비)
    #
    #  EMB_VIT_MAE_    >      VIT-MAE  모델로  패턴  이미지를  숫자로  임베딩  한  것
    #  EMB_SELF_PATCH_    >  Self  Patch  모델로  패턴  이미지를  숫자로  임베딩  한  것

    # SEASON, MOLD_SD 가 없음

    # Spec_No 와 정확하게 일치하는 CNN DB 정보를 별도로 저장
    target_value = spec_no
    matching_df = df[df['SPEC_NO'] == target_value]
    matching_df = matching_df.reset_index(drop=True)

    # 만약 매칭되는 SPEC_NO 가 없을 경우, PRODUCT_CODE 로 CNN DB 정보 중 가장 최신 정보를 별도로 저장
    target_value2 = int(product_cd)
    if matching_df.empty:
        print("SPEC_NO와 매칭되는 CNN data 없기 때문에, 가장 최신 정보로 저장시도")
        matching_df = df[df['PRODUCT_CODE'] == target_value2]
        matching_df = matching_df[:1]
        matching_df = matching_df.reset_index(drop=True)
    else:
        pass

    if matching_df.empty:
        target_value3 = Drawing_no
        print("PRODUCT_CODE 기준으로도 CNN data 정보가 없음. Drawing_no로 저장시도.")
        matching_df = df[df['DRW_NO'] == target_value3]
        matching_df = matching_df[:1]
        matching_df = matching_df.reset_index(drop=True)
    else:
        pass

    if matching_df.empty:
        print("Drawing_no 로도 CNN data 정보가 없음. 종료.")
    else:
        pass

    matching_df_EMB_VIT_MAE = matching_df.iloc[:, matching_df.columns.get_loc('EMB_VIT_MAE_0001'):matching_df.columns.get_loc('EMB_VIT_MAE_0768') + 1]
    import joblib
    # PCA 모델 불러오기
    loaded_pca = joblib.load('pca_model.pkl')

    # 불러온 모델로 변환 적용
    transformed_data = loaded_pca.transform(matching_df_EMB_VIT_MAE)
    print(f"Pattern CNN PCA data : {transformed_data}")
    df_transformed_data = pd.DataFrame(transformed_data)
    new_column_names = [f'EMB_VIT_MAE_PCA_{i + 1}' for i in range(loaded_pca.n_components)]
    df_transformed_data.columns = new_column_names

    # CNN data에 PCA된 결과를 행방향으로 합치기
    combined_df = pd.concat([matching_df, df_transformed_data], axis=1)
    columns_to_remove1 = combined_df.columns[combined_df.columns.get_loc('PRODUCT_CODE'):combined_df.columns.get_loc('SIZE_INCH') + 1]
    combined_df = combined_df.drop(columns=columns_to_remove1)
    # columns_to_remove2 = combined_df.columns[combined_df.columns.get_loc('EMB_SELF_PATCH_0001'):combined_df.columns.get_loc('EMB_SELF_PATCH_0384')+1]
    # combined_df = combined_df.drop(columns=columns_to_remove2)
    columns_to_remove3 = combined_df.columns[combined_df.columns.get_loc('EMB_VIT_MAE_0001'):combined_df.columns.get_loc('EMB_VIT_MAE_0768') + 1]
    combined_df = combined_df.drop(columns=columns_to_remove3)
    columns_to_remove4 = combined_df.columns[combined_df.columns.get_loc('EMB_PERFORMANCE_0004'):combined_df.columns.get_loc('EMB_PERFORMANCE_0006') + 1]
    combined_df = combined_df.drop(columns=columns_to_remove4)
    CNN_indices = list(range(combined_df.columns.get_loc('EMB_PERFORMANCE_0001'),combined_df.columns.get_loc('EMB_VIT_MAE_PCA_14') + 1))
    CNN_columns = [combined_df.columns[i] for i in CNN_indices]
    combined_df.rename(columns = {'SPEC_NO':'SPEC_NO_drawing'}, inplace=True)
    df70 = pd.concat([df0, combined_df], axis=1)

    #############################################################################
    #  패턴  이미지  불러오기
    # img = Image.open(requests.get(f"http://10.82.79.72:9998/pattern/product_code/2.5d/{product_cd}", stream=True).raw)
    # info = eval(img.info['Description'])
    list_of_info = json.loads(requests.get(f"http://10.82.79.72:9998/info/product_code/2.5d/{product_cd}").text)
    if list_of_info is None:
        list_of_info = json.loads(requests.get(f"http://10.82.79.72:9998/info/product_code/2d/{product_cd}").text)
    if list_of_info is None:
        list_of_info = json.loads(requests.get(f"http://10.82.79.72:9998/info/drw_no/2d/{Drawing_no}").text)
    table_of_info = pd.DataFrame(list_of_info)

    use_cols_pattern = ["FILENAME", "PITCHDESIGNTYPE",
                        "B1_PITCHLENGTH", "B2_PITCHLENGTH", 'B3_PITCHLENGTH', "B4_PITCHLENGTH", 'B5_PITCHLENGTH',
                        "B6_PITCHLENGTH", "B7_PITCHLENGTH", "B8_PITCHLENGTH", "B9_PITCHLENGTH", "BA_PITCHLENGTH",
                        "L1_PITCHLENGTH", "L2_PITCHLENGTH", "L3_PITCHLENGTH", "L4_PITCHLENGTH", "L5_PITCHLENGTH",
                        "L6_PITCHLENGTH", "L7_PITCHLENGTH", "L8_PITCHLENGTH", "L9_PITCHLENGTH", "LA_PITCHLENGTH",
                        "U1_PITCHLENGTH", "U2_PITCHLENGTH", "U3_PITCHLENGTH", "U4_PITCHLENGTH", "U5_PITCHLENGTH",
                        "U6_PITCHLENGTH", "U7_PITCHLENGTH", "U8_PITCHLENGTH", "U9_PITCHLENGTH", "UA_PITCHLENGTH",
                        "NUMBEROFPITCHSIZES", "NUMBEROFPITCHES", "PITCHSEQUENCE", "NUMBEROFUPPERPITCHSIZES",
                        "NUMBEROFUPPERPITCHES", "UPPERPITCHSEQUENCE"]
    df_mold_pattern = table_of_info[use_cols_pattern]

    #############################################################################
    #  패턴  성능  정보  불러오기
    list_of_performance = json.loads(
        requests.get(f"http://10.82.79.72:9998/performance/product_code/2.5d/{product_cd}").text)
    if list_of_performance is None:
        list_of_performance = json.loads(requests.get(f"http://10.82.79.72:9998/performance/product_code/2d/{product_cd}").text)
    if list_of_performance is None:
        list_of_performance = json.loads(requests.get(f"http://10.82.79.72:9998/performance/drw_no/2d/{Drawing_no}").text)
    table_of_performance = pd.DataFrame(list_of_performance)

    # ITEM, VALUE 열 추출
    selected_df_perf = table_of_performance[['ITEM', 'VALUE']]

    # ITEM 열을 인덱스로 설정하고, VALUE 열을 값으로 설정
    selected_df_perf = selected_df_perf.set_index('ITEM').T
    selected_df_perf = selected_df_perf[['VCR', 'PKX_N', 'PKY_N']].astype(float)
    selected_df_perf.reset_index(drop=True, inplace=True)

    selected_df0 = df0[['PRODUCT_CODE', 'SEASON', 'MOLD_SD']]

    selected_df_info = table_of_info[['FILENAME']]

    df61 = pd.concat([selected_df0, selected_df_perf, selected_df_info], axis=1, ignore_index=False)

    df62 = df61.rename(
        columns={'VCR': 'HYDROPLANING_VCR', 'PKX_N': 'PATTERNSTIFFNESS_PKX_N', 'PKY_N': 'PATTERNSTIFFNESS_PKY_N'})

    resize_f = 5  # 5인 경우, 이미지 해상도를 0.1mm --> 0.1 * 5 = 0.5mm 로 변경

    # 모아진 스펙 넘버에서, 몰드 정보를 연결하여, 패턴 특성 파라메터 만들기
    df63 = df62[df62['PRODUCT_CODE'] == int(product_cd)]

    if df63.shape[0] == 0:
        print("도면 데이터가 없습니다.")

    else:
        drawing_no = df63["FILENAME"][0].split('_')[0]
        if manual_pattern_key == 0:
            [img_fullpattern, img_stiffness, img_stiffness_normal] = get_image_from_DB(drawing_no, resize_f)
        elif manual_pattern_key == 1:
            [img_fullpattern, img_stiffness, img_stiffness_normal] = get_image_from_manual_upload(uploaded_pattern,
                                                                                                  resize_factor)
            # 새로 불러들인 도면정보를 df62에서 정리 후 df63에 넣어줌
            df62["FILENAME"] = t_temp["FILENAME"]
            df62["MOLD_SD"] = t_temp["INFO_MAIN_DEPTH"]
            df62["HYDROPLANING_VCR"] = t_temp["HYDROPLANING_VCR"]
            df62["PATTERNSTIFFNESS_PKX_N"] = t_temp["PATTERNSTIFFNESS_PKX_N"]
            df62["PATTERNSTIFFNESS_PKY_N"] = t_temp["PATTERNSTIFFNESS_PKY_N"]
            df63 = df62[df62['PRODUCT_CODE'] == int(product_cd)]

            for i in range(df_mold_pattern.shape[1]):
                df_mold_pattern[use_cols_pattern[i]] = None
            for i in range(len(use_cols_pattern)):
                try:
                    df_mold_pattern[use_cols_pattern[i]] = t_temp[use_cols_pattern[i]]
                except:
                    pass

        array1 = np.array(img_fullpattern)[:, :, :3]
        img_fullpattern_r = array1[:, :, 0]  # plt.imshow(img_fullpattern_r)
        img_fullpattern_g = array1[:, :, 1]
        img_fullpattern_b = array1[:, :, 2]
        a0_0 = array1.shape[0]
        a0_1 = array1.shape[1]

        # LATERAL_GROOVE
        lateral_groove_depth1 = df63['MOLD_SD'] * 0.9
        LATERAL_GROOVE = img_fullpattern_b.copy()
        LATERAL_GROOVE[~((img_fullpattern_r == 0) & (img_fullpattern_g == 0) \
                         & (img_fullpattern_b != 0))] = 0
        LATERAL_GROOVE = np.where(LATERAL_GROOVE > 1, lateral_groove_depth1, LATERAL_GROOVE)
        # plt.imshow(LATERAL_GROOVE)
        para_patt1 = np.round(np.count_nonzero(LATERAL_GROOVE) / a0_1, 2)

        # MAIN_GROOVE
        main_groove_depth1 = df63['MOLD_SD']
        MAIN_GROOVE = img_fullpattern_r.copy()
        MAIN_GROOVE[~((img_fullpattern_r != 0) & (img_fullpattern_r != 240) & (img_fullpattern_r != 250) \
                      & (img_fullpattern_r == img_fullpattern_g) & (img_fullpattern_g == img_fullpattern_b))] = 0
        MAIN_GROOVE = np.where(MAIN_GROOVE > 1, main_groove_depth1, MAIN_GROOVE)
        # plt.imshow(MAIN_GROOVE)
        para_patt2 = np.round(np.count_nonzero(MAIN_GROOVE) / a0_1, 2)

        pkx = df63['PATTERNSTIFFNESS_PKX_N'].iloc[-1]
        pky = df63['PATTERNSTIFFNESS_PKY_N'].iloc[-1]
        df_temp = pd.DataFrame([[product_cd, para_patt1, para_patt2, pkx, pky, air, rim_width, test_load]],
                               columns=['PRODUCT_CODE', 'PATT_PAR1', 'PATT_PAR2', 'PKX', 'PKY', 'AIR',
                                        'RIM_WIDTH', 'TEST_LOAD'])

    df70 = pd.merge(df4, df_temp, on='PRODUCT_CODE')
    df70.replace({'SEASON': 'All Weather'}, 'All Season', inplace=True)
    df70.replace({'SEASON': 'Winter Studless'}, 'Winter', inplace=True)
    df70.replace({'SEASON': 'Winter Stud'}, 'Winter', inplace=True)

    # 결측치를 채우기
    df70.fillna({'JLC_MATERIAL': 'N66 1260D/2 28EPI (28T)'}, inplace=True)

    if df70.shape[0] == 0:
        print("컴파운드 데이터가 없습니다.")
        # one hot encording을 적용하기 전에, 숫자 데이타를 지정하기
        df7 = df70.astype('object')  # Data Type을 모두 object로 통일화 하기(데이터를 판다스 object로 변환)
        df7["HS"] = 0
        df7["M10"] = 0
        for col in scale_cols:
            # print(col)
            df7 = df7.astype({col: 'float32'})
        return df7, array1, df_mold_pattern, df63
    else:
        df7 = df70.copy().dropna().reset_index(drop=True)
        ############################################################################
        # COMPD 데이터 연결
        df8 = df7.copy()
        df8['CTB_COMPOUND'] = [x[2:] for x in list(df7['CTB_COMPOUND'])]

        query = f"SELECT * \
                    FROM RCXUSER.CTB_PROPERTY_BY_COMPD_ALL"
        df_temp = pd.read_sql(query, connection)

        query1 = f"SELECT * \
                    FROM MATUSER.CTB_PROPERTY_BY_COMPD"
        df_temp1 = pd.read_sql(query1, connection)

        df_comp = pd.DataFrame()
        df_comp['COMPD'] = sorted(list(set(df8['CTB_COMPOUND'])))
        temp_lst1 = []
        temp_lst2 = []
        for comp in df_comp['COMPD']:
            if (df_temp1['COMPD2'] == comp).any():  # CTB_PROPERTY_BY_COMPD 테이블의 COMPD2 에서 먼저 찾아 넣고
                temp_lst1.append(df_temp1.loc[df_temp1[df_temp1['COMPD2'] == comp].index[0]]['HS'])
                temp_lst2.append(df_temp1.loc[df_temp1[df_temp1['COMPD2'] == comp].index[0]]['M10'])
            elif (df_temp['COMP'] == comp).any():  # 다음 CTB_PROPERTY_BY_COMPD_ALL 테이블의 COMP 에서 찾아 넣고
                temp_lst1.append(df_temp.loc[df_temp[df_temp['COMP'] == comp].index[0]]['HS'])
                temp_lst2.append(df_temp.loc[df_temp[df_temp['COMP'] == comp].index[0]]['M10'])
            elif (df_temp1['COMPD'] == comp).any():  # 다음 CTB_PROPERTY_BY_COMPD 테이블의 COMPD 에서 찾아 넣음. (X 스펙일 수 있음)
                temp_lst1.append(df_temp1.loc[df_temp1[df_temp1['COMPD'] == comp].index[0]]['HS'])
                temp_lst2.append(df_temp1.loc[df_temp1[df_temp1['COMPD'] == comp].index[0]]['M10'])
            else:
                temp_lst1.append(0)
                temp_lst2.append(0)

        df_comp['HS'] = temp_lst1
        df_comp['M10'] = temp_lst2

        # df3 = df_comp
        df9 = pd.merge(df8, df_comp, left_on='CTB_COMPOUND', right_on='COMPD', how='left')
        df10 = df9.drop('CTB_COMPOUND', axis=1)
        # df11 = combined_df.iloc[:, combined_df.columns.get_loc('EMB_PERFORMANCE_0001'):combined_df.columns.get_loc('EMB_VIT_MAE_PCA_14') + 1]
        df11 = pd.concat([df10, combined_df], axis=1)

        return df11, array1, df_mold_pattern, df63

def Data_Cleaning_Manual(df0,air,test_load,rim_width,use_cols,scale_cols,t_temp, input_info):
    # 스펙 변환
    # use_cols_ = _use_cols.cols4
    # use_cols = _use_cols.cols4_1
    df1 = df0.copy()
    temp = []
    for name12 in df1['CTB_STEP_GAUGE']:
        if name12:  # 비어 있는지 여부를 확인
            name12 = [v for v in name12.split('-') if v]
            temp.append(max([float(x) for x in name12]))
        else:
            temp.append('')
    df1['CTB_STEP_GAUGE'] = temp

    ###################################################
    # 문자열을 단순화 하기
    temp = []
    for i in range(df1.shape[0]):
        if df1['JLC_MATERIAL'][i] is not None:
            jlc_mat = df1['JLC_MATERIAL'][i]
            temp.append(jlc_mat[0])
        elif df1['RBT_MATERIAL'][i] is not None:
            jlc_mat = df1['RBT_MATERIAL'][i]
            temp.append(jlc_mat[0])
    df1['JLC_MATERIAL'] = temp
    #
    # ['SIZE_NSW_MM', 'SIZE_SERIES_MM', 'SIZE_INCH', 'PATTERN', 'CTB_STEP_GAUGE', 'JLC_MATERIAL', 'PRE_BELT_TYPE',
    #  'BT1_MATERIAL', 'BT1_WIDTH', 'BT1_ANGLE', 'BT1_EPI', 'PLY_LOCK', 'C01_MATERIAL', 'C01_EPI', 'FIL_TYPE',
    #  'FIL_COMPOUND', 'FIL_WIDTH',
    #  'AIR', 'TEST_LOAD', 'RIM_WIDTH', 'PATT_PAR1', 'PATT_PAR2', 'PKX', 'PKY', 'HS', 'M10']

    df4 = df1[use_cols[:-8]+['CTB_COMPOUND']]
    df4["SPEC_NO"] = df1["SPEC_NO"]
    df4['PRODUCT_CODE'] = [x[4:11] for x in df4.copy()['SPEC_NO']]
    product_cd = df4["SPEC_NO"][0][4:11]
    #
    # connection = cx_Oracle.connect(credentials, encoding="UTF-8", nencoding="UTF-8")
    # query1 = f"SELECT * \
    #             FROM RCXUSER.pdat_catia_fullassy \
    #             WHERE pdat_catia_fullassy.imagetype = '2D_FullAssy' AND pdat_catia_fullassy.PRODUCT_CODE = '{product_cd}'"
    # df61 = pd.read_sql(query1, connection)

    if manual_pattern_key == 1: # 이미지에 빠져 있는 변수를 input_info에서 가져와서 만들어야 함.
        df61 = t_temp
        if "PRODUCT_CODE" not in list(df61.columns):
            df61['PRODUCT_CODE'] = product_cd
        if "SEASON" not in list(df61.columns):
            df61['SEASON'] = 'All Season'
        if "MOLD_SD" not in list(df61.columns):
            df61['MOLD_SD'] = input_info['Performance']['Info']['Groove']['Depth']

        if "B4_PITCHLENGTH" not in list(df61.columns):
            df61["B4_PITCHLENGTH"] = None
        if "B5_PITCHLENGTH" not in list(df61.columns):
            df61["B5_PITCHLENGTH"] = None
        if "B6_PITCHLENGTH" not in list(df61.columns):
            df61["B6_PITCHLENGTH"] = None
        if "B7_PITCHLENGTH" not in list(df61.columns):
            df61["B7_PITCHLENGTH"] = None
        if "B8_PITCHLENGTH" not in list(df61.columns):
            df61["B8_PITCHLENGTH"] = None
        if "B9_PITCHLENGTH" not in list(df61.columns):
            df61["B9_PITCHLENGTH"] = None
        if "BA_PITCHLENGTH" not in list(df61.columns):
            df61["BA_PITCHLENGTH"] = None

        if "L1_PITCHLENGTH" not in list(df61.columns):
            df61[["L1_PITCHLENGTH", "L2_PITCHLENGTH", "L3_PITCHLENGTH", "L4_PITCHLENGTH", "L5_PITCHLENGTH",
                "L6_PITCHLENGTH", "L7_PITCHLENGTH", "L8_PITCHLENGTH", "L9_PITCHLENGTH", "LA_PITCHLENGTH"]] = (
            df61)[["B1_PITCHLENGTH", "B2_PITCHLENGTH", 'B3_PITCHLENGTH', "B4_PITCHLENGTH", 'B5_PITCHLENGTH',
                "B6_PITCHLENGTH", "B7_PITCHLENGTH", "B8_PITCHLENGTH", "B9_PITCHLENGTH", "BA_PITCHLENGTH"]]
        if "U1_PITCHLENGTH" not in list(df61.columns):
            df61[["U1_PITCHLENGTH", "U2_PITCHLENGTH", "U3_PITCHLENGTH", "U4_PITCHLENGTH", "U5_PITCHLENGTH",
                            "U6_PITCHLENGTH", "U7_PITCHLENGTH", "U8_PITCHLENGTH", "U9_PITCHLENGTH", "UA_PITCHLENGTH"]] = (
            df61)[["B1_PITCHLENGTH", "B2_PITCHLENGTH", 'B3_PITCHLENGTH', "B4_PITCHLENGTH", 'B5_PITCHLENGTH',
                   "B6_PITCHLENGTH", "B7_PITCHLENGTH", "B8_PITCHLENGTH", "B9_PITCHLENGTH", "BA_PITCHLENGTH"]]
            df61[["NUMBEROFUPPERPITCHSIZES","NUMBEROFUPPERPITCHES", "UPPERPITCHSEQUENCE"]] = df61[["NUMBEROFPITCHSIZES", "NUMBEROFPITCHES", "PITCHSEQUENCE"]]

    use_cols_pattern = ["FILENAME", "PITCHDESIGNTYPE",
                        "B1_PITCHLENGTH", "B2_PITCHLENGTH", 'B3_PITCHLENGTH', "B4_PITCHLENGTH", 'B5_PITCHLENGTH',
                        "B6_PITCHLENGTH", "B7_PITCHLENGTH", "B8_PITCHLENGTH", "B9_PITCHLENGTH", "BA_PITCHLENGTH",
                        "L1_PITCHLENGTH", "L2_PITCHLENGTH", "L3_PITCHLENGTH", "L4_PITCHLENGTH", "L5_PITCHLENGTH",
                        "L6_PITCHLENGTH", "L7_PITCHLENGTH", "L8_PITCHLENGTH", "L9_PITCHLENGTH", "LA_PITCHLENGTH",
                        "U1_PITCHLENGTH", "U2_PITCHLENGTH", "U3_PITCHLENGTH", "U4_PITCHLENGTH", "U5_PITCHLENGTH",
                        "U6_PITCHLENGTH", "U7_PITCHLENGTH", "U8_PITCHLENGTH", "U9_PITCHLENGTH", "UA_PITCHLENGTH",
                        "NUMBEROFPITCHSIZES", "NUMBEROFPITCHES", "PITCHSEQUENCE", "NUMBEROFUPPERPITCHSIZES",
                        "NUMBEROFUPPERPITCHES", "UPPERPITCHSEQUENCE"]
    df_mold_pattern = df61.copy()[use_cols_pattern]

    resize_f = 5  # 5인 경우, 이미지 해상도를 0.1mm --> 0.1 * 5 = 0.5mm 로 변경

    df63 = df61.copy()[['PRODUCT_CODE', 'FILENAME', 'SEASON', 'MOLD_SD', 'HYDROPLANING_VCR', 'PATTERNSTIFFNESS_PKX_N', 'PATTERNSTIFFNESS_PKY_N']]

    if df63.shape[0] == 0:
        print("도면 데이터가 없습니다.")

    else:
        filename = df63["FILENAME"][0]
        [img_fullpattern, img_stiffness, img_stiffness_normal] = get_image_from_manual_upload(uploaded_pattern,
                                                                                              resize_factor)

        array1 = np.array(img_fullpattern)[:, :, :3]
        img_fullpattern_r = array1[:, :, 0]  # plt.imshow(img_fullpattern_r)
        img_fullpattern_g = array1[:, :, 1]
        img_fullpattern_b = array1[:, :, 2]
        a0_0 = array1.shape[0]
        a0_1 = array1.shape[1]

        # LATERAL_GROOVE
        lateral_groove_depth1 = df63['MOLD_SD'] * 0.9
        LATERAL_GROOVE = img_fullpattern_b.copy()
        LATERAL_GROOVE[~((img_fullpattern_r == 0) & (img_fullpattern_g == 0) \
                         & (img_fullpattern_b != 0))] = 0
        LATERAL_GROOVE = np.where(LATERAL_GROOVE > 1, lateral_groove_depth1, LATERAL_GROOVE)
        # plt.imshow(LATERAL_GROOVE)
        para_patt1 = np.round(np.count_nonzero(LATERAL_GROOVE) / a0_1, 2)

        # MAIN_GROOVE
        main_groove_depth1 = df63['MOLD_SD']
        MAIN_GROOVE = img_fullpattern_r.copy()
        MAIN_GROOVE[~((img_fullpattern_r != 0) & (img_fullpattern_r != 240) & (img_fullpattern_r != 250) \
                      & (img_fullpattern_r == img_fullpattern_g) & (img_fullpattern_g == img_fullpattern_b))] = 0
        MAIN_GROOVE = np.where(MAIN_GROOVE > 1, main_groove_depth1, MAIN_GROOVE)
        # plt.imshow(MAIN_GROOVE)
        para_patt2 = np.round(np.count_nonzero(MAIN_GROOVE) / a0_1, 2)

        season = df63['SEASON'].iloc[-1]
        pkx = df63['PATTERNSTIFFNESS_PKX_N'].iloc[-1]
        pky = df63['PATTERNSTIFFNESS_PKY_N'].iloc[-1]
        df_temp = pd.DataFrame([[product_cd, para_patt1, para_patt2, pkx, pky, air, rim_width, test_load]],
                               columns=['PRODUCT_CODE', 'PATT_PAR1', 'PATT_PAR2', 'PKX', 'PKY', 'AIR',
                                        'RIM_WIDTH', 'TEST_LOAD'])

    df70 = pd.merge(df4, df_temp, on='PRODUCT_CODE')
    df70.replace({'SEASON': 'All Weather'}, 'All Season', inplace=True)
    df70.replace({'SEASON': 'Winter Studless'}, 'Winter', inplace=True)
    df70.replace({'SEASON': 'Winter Stud'}, 'Winter', inplace=True)

    # 결측치를 채우기
    df70.fillna({'JLC_MATERIAL': 'N66 1260D/2 28EPI (28T)'}, inplace=True)

    if df70.shape[0] == 0:
        print("컴파운드 데이터가 없습니다.")
        # one hot encording을 적용하기 전에, 숫자 데이타를 지정하기
        df7 = df70.astype('object')  # Data Type을 모두 object로 통일화 하기(데이터를 판다스 object로 변환)
        df7["HS"] = 0
        df7["M10"] = 0
        for col in scale_cols:
            # print(col)
            df7 = df7.astype({col: 'float32'})
        return df7, array1, df_mold_pattern, df63
    else:
        df7 = df70.copy().dropna().reset_index(drop=True)
        ############################################################################
        # COMPD 데이터 연결
        df8 = df7.copy()
        df8['CTB_COMPOUND'] = [x[2:] for x in list(df7['CTB_COMPOUND'])]

        query = f"SELECT * \
                    FROM RCXUSER.CTB_PROPERTY_BY_COMPD_ALL"
        df_temp = pd.read_sql(query, connection)

        query1 = f"SELECT * \
                    FROM MATUSER.CTB_PROPERTY_BY_COMPD"
        df_temp1 = pd.read_sql(query1, connection)

        df_comp = pd.DataFrame()
        df_comp['COMPD'] = sorted(list(set(df8['CTB_COMPOUND'])))
        temp_lst1 = []
        temp_lst2 = []
        for comp in df_comp['COMPD']:
            if (df_temp1['COMPD2'] == comp).any():  # CTB_PROPERTY_BY_COMPD 테이블의 COMPD2 에서 먼저 찾아 넣고
                temp_lst1.append(df_temp1.loc[df_temp1[df_temp1['COMPD2'] == comp].index[0]]['HS'])
                temp_lst2.append(df_temp1.loc[df_temp1[df_temp1['COMPD2'] == comp].index[0]]['M10'])
            elif (df_temp['COMP'] == comp).any():  # 다음 CTB_PROPERTY_BY_COMPD_ALL 테이블의 COMP 에서 찾아 넣고
                temp_lst1.append(df_temp.loc[df_temp[df_temp['COMP'] == comp].index[0]]['HS'])
                temp_lst2.append(df_temp.loc[df_temp[df_temp['COMP'] == comp].index[0]]['M10'])
            elif (df_temp1['COMPD'] == comp).any():  # 다음 CTB_PROPERTY_BY_COMPD 테이블의 COMPD 에서 찾아 넣음. (X 스펙일 수 있음)
                temp_lst1.append(df_temp1.loc[df_temp1[df_temp1['COMPD'] == comp].index[0]]['HS'])
                temp_lst2.append(df_temp1.loc[df_temp1[df_temp1['COMPD'] == comp].index[0]]['M10'])
            else:
                temp_lst1.append(0)
                temp_lst2.append(0)

        df_comp['HS'] = temp_lst1
        df_comp['M10'] = temp_lst2

        # df3 = df_comp
        df9 = pd.merge(df8, df_comp, left_on='CTB_COMPOUND', right_on='COMPD', how='left')
        df10 = df9.drop('CTB_COMPOUND', axis=1)

        return df10, array1, df_mold_pattern, df63

def Prediction(df1, BestModel):
    targets = list(BestModel.keys())

    df_result = pd.DataFrame()
    for target in targets:
        y_pred = BestModel[target].predict(df1)
        df_result[target] = y_pred
    df = pd.concat([df1,df_result],axis=1)
    return df

# def Prediction(df_samples, df1, use_cols, scale_cols, scaler_all, targets, BestModel):
#     # One Hot Encording
#     X_final = pd.get_dummies(df1[use_cols])
#     X_final.replace((True, False), ("1", "0"), inplace=True)
#     # Input Sample과 동일한 열 이름으로 만들어 주기
#     temp_input2 = pd.concat([df_samples, X_final]).reset_index(drop=True)
#     temp_input3 = temp_input2.iloc[2:, 0:df_samples.shape[1]]  # sample input에 들어 있지 않은 문자열을 삭제하기
#     drop_value = temp_input2.iloc[:, df_samples.shape[1]:].columns
#     if drop_value.size >= 1:
#         print("다음 데이터가 학습 모델에 누락되어 있습니다.")
#         for drop_value_i in drop_value:
#             st.warning(f"{drop_value_i} 데이터가 학습 모델에 누락되어 있습니다")
#     X_final1 = temp_input3.fillna(0)
#     X_final1_num = X_final1[scale_cols]  # GAM 모델용
#     X_final1_dummy = X_final1.loc[:, ~X_final1.columns.isin(scale_cols)]  # GAM 모델용
#
#     # 숫자열의 데이터를 Normalize하기
#     x_final1 = X_final1.copy()
#     x_final1[scale_cols] = scaler_all.transform(X_final1[scale_cols])
#
#     ### 2023.12.28 수정부분 (yhat_XGB 생성 시, 벨트/카카스류 '1'이 아닌 'True'로 표기 변경)
#     X_final2 = X_final1.copy()
#     X_final2.replace(("1", "0"), (True, False), inplace=True)  # XGB 모델용
#
#     result = np.empty(0)
#     for target in targets:
#         yhat_NN = BestModel[f"{target}_(NN)"].predict(x_final1)
#         yhat_XGB = BestModel[f"{target}_(XGB)"].predict(X_final2)
#         yhat_GAM = BestModel[f"{target}_(GAM)"].predict(X_final1_num, X_final1_dummy)
#         # print(f"{target} : {yhat_NN}, {yhat_XGB}, {yhat_GAM}")
#         temp = np.concatenate([yhat_NN, yhat_XGB, yhat_GAM], 0)
#         result = np.append(result, temp)
#
#     result = np.append(result, (result[1] + result[2]) / 2)  # GM_SP_80 XBG & GAM average
#     result = np.append(result, (result[4] + result[5]) / 2)  # GM_TNAI_80 XBG & GAM average
#     result = np.append(result, (result[7] + 6.8))  # Converted from HK to GM, 2024 Correlation factor >> GM = HK(Jangdong) + 6.8
#
#     df_result = round(pd.DataFrame(result), 1).applymap("{0:.1f}".format).transpose()
#     df_result.index = ['Target_Tire']
#     df_result.columns = ['GMSP80_NN', 'GMSP80_XGB', 'GMSP80_GAM',
#                          'GM_TNAI80_NN', 'GM_TNAI80_XGB', 'GM_TNAI80_GAM',
#                          'GMSP80_Final', 'GM_TNAI80_Final', 'GM_TNAI80_Final(GM)']
#     return df_result

def Prediction_Oct(df1,Oct_BestModel,Oct_scaler_all,Oct_use_cols,Oct_scale_cols,Oct_targets,Oct_categoricals,Oct_information_minmax,Oct_df_samples,df_predicted_Oct):
    # One Hot Encording
    X_final = pd.get_dummies(df1[Oct_use_cols])
    X_final.replace((True, False), ("1", "0"), inplace=True)
    X_final = X_final.astype('float')
    # Ys_final = df1[targets]

    # df_samples = df3.loc[:, ~df3.columns.isin(targets+['SPEC_NO'])].iloc[:2,:] # OHE이 적용된 Data Sample 형식
    # df_samples = Xall.copy().iloc[:2,:]
    # Input Sample과 동일한 열 이름으로 만들어 주기
    temp_input2 = pd.concat([Oct_df_samples, X_final]).reset_index(drop=True)
    temp_input3 = temp_input2.iloc[2:, 0:Oct_df_samples.size]  # sample input에 들어 있지 않은 문자열을 삭제하기
    drop_value = temp_input2.iloc[:, Oct_df_samples.size:].columns
    if drop_value.size >= 1:
        print("다음 데이터가 학습 모델에 누락되어 있습니다.")
        for drop_value_i in drop_value:
            print(f"{drop_value_i}")
    X_final1 = temp_input3.fillna(0)
    X_final1_num = X_final1[Oct_scale_cols]  # GAM 모델용
    X_final1_dummy = X_final1.loc[:, ~X_final1.columns.isin(Oct_scale_cols)]  # GAM 모델용

    # 숫자열의 데이터를 Normalize하기
    x_final1 = X_final1.copy()
    x_final1[Oct_scale_cols] = Oct_scaler_all.transform(X_final1[Oct_scale_cols])

    result_Oct = np.empty(0)
    for Oct_target in Oct_targets:
        # y_final = Ys_final[target]
        # yhat_NN = Oct_BestModel[f"{target}_(NN)"].predict(x_final1)
        yhat_XGB = Oct_BestModel[f"{Oct_target}_(XGB)"].predict(X_final1)
        yhat_GAM = Oct_BestModel[f"{Oct_target}_(GAM)"].predict(X_final1_num, X_final1_dummy)
        # print(f"{target} : {yhat_XGB}, {yhat_GAM}")
        temp_Oct = np.concatenate([yhat_XGB], 0)
        result_Oct = np.append(result_Oct, temp_Oct)
    result_Oct = pd.DataFrame(result_Oct)

    return result_Oct, df_predicted_Oct

# FFT 변환
def fft_cal(v):
    fft_vals = fft(v)  # FFT 계산
    fft_norm = fft_vals / len(v)  # FFT 계산된 결과를 정규화
    fft_abs = 2.0 * abs(fft_norm)  # 푸리에 계수 계산
    return fft_vals, fft_norm, fft_abs

# 파라메터 분석
def analysis_parameter(fft_abs1,fft_abs2,total_n):

    fft_abs1_n = fft_abs1 / fft_abs1[0]
    fft_abs2_n = fft_abs2 / fft_abs2[0]
    # plt.plot(fft_abs1_n[1:201])
    # plt.plot(fft_abs2_n[1:201])

    # The Peak Amplitude
    peak_Pyy=[]
    # peak_Pyy.append(np.max(fft_abs1_n[total_n-20:total_n+20]))
    peak_Pyy.append(np.max(fft_abs1_n[1:201]))

    # Half Power Bandwidth
    fft_abs1_n_sm = (fft_abs1_n[1:201] + fft_abs1_n[2:202] + fft_abs1_n[3:203])/3
    p1 = np.max(fft_abs1_n_sm[total_n - 20:total_n + 20])
    t1 = np.where(fft_abs1_n_sm == p1)[0][0]
    arr1 = np.where(fft_abs1_n_sm[1:201] < p1/2)[0]
    low = arr1[np.where(arr1 - t1 < 0)][-1]
    high = arr1[np.where(arr1 - t1 > 0)][0]
    bandwidth = high - low

    # The Second Power of Variation
    spvp = fft_abs1_n[1:201]
    spvm = fft_abs1_n[2:202]
    spv = np.mean((spvp-spvm)*(spvp-spvm))

    return peak_Pyy, spv, bandwidth, fft_abs1_n[1:201], fft_abs2_n[1:21]

# PNI calculator
def PNI_Cal(n_peak_Pyy, n_bandwidth):
    PNI = (math.sqrt(n_bandwidth/22.5))/(math.pow(n_peak_Pyy/0.3,2))
    return PNI

# 입력된 Sequence를 각 Class에 맞게 조정
def seq_check(seq):
    seq = seq.strip()
    seq = seq.replace(' ', '')
    if (pit_kind_n == 3) & (max(seq) == '6'):
        seq = seq.replace('2', '1').replace('3', '2').replace('4', '2').replace('5', '3').replace('6', '3')
    elif (pit_kind_n == 3) & (max(seq) == '9'):
        seq = seq.replace('2', '1').replace('3', '1').replace('4', '2').replace('5', '2').replace('6', '2').replace('7', '3').replace('8', '3').replace('9', '3')
    elif (pit_kind_n == 5) & (max(seq) == 'A'):
        seq = seq.replace('2', '1').replace('3', '2').replace('4', '2').replace('5', '3').replace('6', '3').replace('7', '4').replace('8', '4').replace('9', '5').replace('A', '5')
    else:
        seq = seq
    return seq

def fetch_image(url):
    response = requests.get(url, stream=True)
    if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
        try:
            return Image.open(io.BytesIO(response.content))
        except Exception as e:
            print(f"이미지 열기 실패: {e}")
            return None
    else:
        print(f"이미지 응답 아님. status_code: {response.status_code}, content-type: {response.headers.get('Content-Type')}")
        return None

#############################################################################################
# 오라클 DB 접속 아이디 및 접속 주소 등 정보 입력
credentials = f"{auth.username}/{auth.password}@{auth.host}:{auth.port}/{auth.servicename}/"
connection = cx_Oracle.connect(credentials, encoding="UTF-8", nencoding="UTF-8")

st.set_page_config(layout="wide")
logo_co1, logo_co2 = st.columns([3,1])
with logo_co2:
    img_logo = Image.open('./Pages/Hankooktire KAIST collaboration_v2.png')
    st.image(img_logo)
st.title("GM TNAI - AI PREDICTION")
st.text("Version: GM_TNAI_Version2.2_250710")

##############################################################################################
##############################################################################################
st.subheader("1. TIRE ID")
spec_no = st.text_input(label="Key in Tire ID and Press Enter (Ex: KPKT1028478X00027)")
spec_no = spec_no.strip()
# spec_no = 'KPKT1028478X00027' sample
# spec_no = 'DSKT1024187X00000'  에러 나는 것
# spec_no = 'DPKT2020120S00000'
# spec_no = 'HSKT1033741V00000' Pattern image 없는 것 sample
# spec_no = 'KPKT1031524M00004' Structure 없는 것 sample
# spec_no = 'KPKT1030266S00002' Dual pitch sample

manual_pattern_key = -1
DRW_NO_keyin = "None"

M_code =spec_no[4:11]
if spec_no:
    if len(spec_no) != 17:
        st.error("!!Check again!!")

    url_2_5d = f"http://10.82.79.72:9998/pattern/product_code/2.5d/{M_code}"
    url_2d = f"http://10.82.79.72:9998/pattern/product_code/2d/{M_code}"

    img_pattern = fetch_image(url_2_5d)
    url = url_2_5d
    if img_pattern is None:
        img_pattern = fetch_image(url_2d)
        url = url_2d
        if img_pattern is None:
            st.subheader("※ Drawing_No")
            DRW_NO_keyin = st.text_input(
                label="There is no connection with M-code. So, Key in Drawing_No and Press Enter (Ex: RLR-181407)")
            DRW_NO_keyin = DRW_NO_keyin.strip()
            # DRW_NO_keyin = 'RLR-181407'
            url_drw = f"http://10.82.79.72:9998/pattern/drw_no/2d/{DRW_NO_keyin}"
            img_pattern = fetch_image(url_drw)
            url = url_drw
            if DRW_NO_keyin == "None":
                if img_pattern is None:
                    # 이미지 사용 코드
                    st.warning(
                        'Warning : Sorry, there is NO pattern image. You need to Make pattern image. Please see Guide Tab.')
                    pass
        else:
            pass
    else:
        pass

    if M_code and img_pattern:
        try:
            img_pattern = Image.open(requests.get(url, stream=True).raw)
            manual_pattern_key = 0
            landsea_ori = get_image_info(M_code, DRW_NO_keyin, img_pattern)
            Drawing_no = landsea_ori[0][0][:10]
            # croppedImage = img_pattern.crop((0, 0, img_pattern.size[0] / 3, img_pattern.size[1]))
            Tread_width = float(img_pattern.info['Description'][
                                img_pattern.info['Description'].find('TreadWidth') + 13:img_pattern.info[
                                                                                            'Description'].find(
                                    'TreadWidth') + 18])
            croppedImage = img_pattern.crop((0, (img_pattern.size[1] / 2) - (Tread_width * 10 / 2),
                                             img_pattern.size[0] / 4,
                                             (img_pattern.size[1] / 2) + (Tread_width * 10 / 2)))
            st.image(croppedImage)
            st.write(
                f"■ Drawing file = {landsea_ori[0][0]} , ■ Land ratio [%] = {landsea_ori[0][1]} , ■ Groove(Main) ratio_Gray color [%] = {landsea_ori[0][2]} , Only show within TW")
            Img_col1, Img_col2 = st.columns([5, 1])
            with Img_col1:
                st.warning('If the Pattern is different, add the correct pattern image in the below menu.')
            with Img_col2:
                st.warning('Image scale 1:3')
        except:
            print('파일이 없음')
            st.warning('Warning : Sorry, there is NO pattern image. You need to Make pattern image. Please see Guide Tab.')
            # uploaded_pattern = Image.open('./APR-180571_RF11_225-60R17H_FullAssy_2D_analysis.png')
            # uploaded_pattern = Image.open('./RPR-230570_IK01_225-50R18T_FullAssy_2D.png') "Full pitch만 만든 샘플"
            # uploaded_pattern = Image.open('./CPR-230096_IK01A_235-60R18V_FullAssy_2D_analysis.png') "Full pitch image 분석된 샘플"

    with st.expander("▣ Manual Pattern Image Input", expanded=False):
        Manual_image_tab1, Manual_image_tab2 = st.tabs(["Image input", "Guide"])
        with Manual_image_tab1:
            uploaded_file = st.file_uploader("JPG or PNG pattern image")
            if uploaded_file is not None:
                # To read file as bytes:
                bytes_data = uploaded_file.getvalue()
                # st.write(bytes_data)
                uploaded_pattern = Image.open(uploaded_file)
                manual_pattern_key = 1
                landsea_manual = get_image_info(M_code, uploaded_pattern)
                # uploaded_croppedImage = uploaded_pattern.crop(
                #     (0, 0, uploaded_pattern.size[0] / 3, uploaded_pattern.size[1]))
                uploaded_pattern_Tread_width = float(uploaded_pattern.info['Description'][
                                    uploaded_pattern.info['Description'].find('TreadWidth') + 13:uploaded_pattern.info[
                                                                                                'Description'].find(
                                        'TreadWidth') + 18])
                uploaded_croppedImage = uploaded_pattern.crop((0, (uploaded_pattern.size[1] / 2) - (uploaded_pattern_Tread_width * 10 / 2),
                                                 uploaded_pattern.size[0] / 4,
                                                 (uploaded_pattern.size[1] / 2) + (uploaded_pattern_Tread_width * 10 / 2)))
                st.image(uploaded_croppedImage)
                st.write(
                    f"■ Drawing file = {landsea_manual[0][0]} , ■ Land ratio [%] = {landsea_manual[0][1]} , ■ Groove(Main) ratio_Gray color [%] = {landsea_manual[0][2]}")
                Img2_col1, Img2_col2 = st.columns([5, 1])
                with Img2_col1:
                    st.warning('This is finally selected pattern image.')
                with Img2_col2:
                    st.warning('Image scale 1:3')
                img_pattern = uploaded_pattern
                input_info = eval(img_pattern.info["Description"])
                # st.write(input_info)
                d_info = fn_dict_to_flat(input_info)
                # st.write(d_info)
                t_temp = pd.DataFrame.from_dict([d_info]).rename(columns=d_table, inplace=False)
                # t_temp.to_csv("t_temp.csv", index=False)
                # st.write(t_temp)
                # t_temp = pd.DataFrame(d_info, columns=d_table.keys(), index=[0]).rename(columns=d_table, inplace=False)
                resize_factor = 5
        with Manual_image_tab2:
            manual_img1 = Image.open('./Pages/Manual image/Manual_image1.jpg')
            manual_img2 = Image.open('./Pages/Manual image/Manual_image2.jpg')
            manual_img3 = Image.open('./Pages/Manual image/Manual_image3.jpg')
            manual_img4 = Image.open('./Pages/Manual image/Manual_image4.jpg')
            manual_img5 = Image.open('./Pages/Manual image/Manual_image5.jpg')
            st.image(manual_img1)
            st.image(manual_img2)
            st.image(manual_img3)
            st.image(manual_img4)
            st.image(manual_img5)

    try:
        if t_temp.shape[1] == 120:
            manual_pattern_key = -1
            st.error(
                'Warning : Sorry, there is NO pattern image analysis information. You need to check guide.')
    except:
        pass

if manual_pattern_key > -1:
    with st.form(key='DOE_STUDY'):
        st.subheader("3. DOE STUDY")
        ### T:HINT 결과
        query1 = f"SELECT * \
                        FROM RCXUSER.da_spec\
                        WHERE da_spec.spec_no = '{spec_no}'"
        df0 = pd.read_sql(query1, connection)

        # 초기값
        # air = 0
        # test_load = 0
        # rim_width = 0

        ###############################################################################
        # _PN_data_path = 'Z:\\_DATA_Training\\GM_TNAI\\_GM_TNAI_data_241008_SJH.pkl'
        _PN_data_path = 'Z:\\_DATA_Training\\GM_TNAI\\_GM_TNAI_data_250528_SJH.pkl'
        _PN_data = pickle.load(open(_PN_data_path, 'rb'))
        BestModel = _PN_data['2.MLP_Best_Model']
        # scaler_all = _PBN_data['2.Scaler (with all data)']
        use_cols = _PN_data['1.Input Properties list']
        scale_cols = _PN_data['1.Input Properties list (number)']
        targets = _PN_data['1.Target list']
        categoricals = _PN_data["2.Category"]
        information_minmax = _PN_data["2.Min_Max information (all data)"]
        # df_samples = _PBN_data['2.Input_Sample for New Prediction']  # OHE이 적용된 Data Sample 형식
        all_data = _PN_data['1.Input Data']

        # Oct_PN_data_path = 'Z:\\_DATA_Training\\PATTERN_NOISE\\_PN_Oct_data_231122_r.pkl'
        # Oct_PN_data = pickle.load(open(Oct_PN_data_path, 'rb'))
        # Oct_BestModel = Oct_PN_data['2.MLP_Best_Model']
        # Oct_scaler_all = Oct_PN_data['2.Scaler (with all data)']
        # Oct_use_cols = Oct_PN_data['1.Input Properties list']
        # Oct_scale_cols = Oct_PN_data['1.Input Properties list (number)']
        # Oct_targets = Oct_PN_data['1.Target list']
        # Oct_categoricals = Oct_PN_data["2.Category"]
        # Oct_information_minmax = Oct_PN_data["2.Min_Max information (all data)"]
        # Oct_df_samples = Oct_PN_data['2.Input_Sample for New Prediction']  # OHE이 적용된 Data Sample 형식
        # df_predicted_Oct = Oct_PN_data['2.MLP_Data and Result']

        ###############################################################################
        ################################# Input GUI #################################
        #############################################################################
        st.subheader("1) Test Condition")
        Cond_col1, Cond_col2, Cond_col3 = st.columns(3)
        with Cond_col1:
            test_load = st.number_input('Key in Test Load [kgf]', value=540.0, format="%.0f")
        with Cond_col2:
            # air = st.number_input('Key in Air Pressure [kgf/cm2]', min_value=1, max_value=10, value=2.0, format="%.2f")
            air = st.number_input('Key in Air Pressure [kgf/cm2]', value=2.40, format="%.2f")
        with Cond_col3:
            rim_width = st.number_input('Key in Rim Width [inch]', value=8.0, format="%.1f")
        st.markdown("""---""")
        # air = 2.40
        # test_load = 600
        # rim_width = 8.0

        # GM TNAI 2024 P-V validation (총 37개)
        # DSKT1024187X00000, DSKT1024187X00001, DSKT1024187X00002, DSKT1024187X00003 : 동일스펙X , X05~13
        # JSKT1023860X00003 : 동일스펙X , X25~X30
        # DPKT2020615X00017, DPKT2020615X00019 : M-code X
        # KPKT1021931X00020 : OK
        # TPKT1024217X00021, TPKT1024217X00020, TPKT1024217X00019, TPKT1024217X00018 : OK
        # DPKT2021223X00001, DPKT2021223X00009 : M-code X
        # DSKT1025783X00003, TPKT1024217X00027, TPKT1024217X00029, TPKT1024217X00028, CPKT1024007X00049 : OK
        # JSKT1023860X00021 : 동일스펙X , X25~30
        # KPKT1014191X00013, TPKT1024217X00030, KPKT1026754X00020, KPKT1026754X00026, KPKT1026754X00045 : OK
        # KPKT1034601X00005, KPKT1034601X00006, MPKT1025134M00007 : OK
        # MPKT1025135M00006 : Test data 없음
        # MPKT1029026M00006 : 동일스펙X , X00~06 V00~01
        # MPKT1027404M00005 : OK
        # IPKT2021485M00002, JMKT1014801S00000 : M-code X
        # KPKT1013373S00000, KPKT1015220S00000 : OK
        # DPKT2020120S00000, DPKT2001863S00000 : M-code X

        # CNN DB 에서 SPEC_NO 기준으로 동일한 스펙이나 유사 스펙이 있는지 찾아보기
        # substring = 'DPKT2001863'
        # matching_values = df[df['SPEC_NO'].str.contains(substring, na=False)]
        # print(matching_values)

        # 데이터를 크리닝하기
        if manual_pattern_key == 0:
            [df1, Pattern_Image, df_mold_pattern, df_product] = Data_Cleaning(df0, air, test_load, rim_width, use_cols, scale_cols)
        elif manual_pattern_key == 1:
            [df1, Pattern_Image, df_mold_pattern, df_product] = Data_Cleaning_Manual(df0, air, test_load, rim_width, use_cols, scale_cols, t_temp, input_info)

        if df1.shape[0] == 0:
            st.error('ERROR : There is no compound information, please check the spec no.')
        else:
            st.success('Spec data check complete')

        df1['AIR'][0] = air
        df1['TEST_LOAD'][0] = test_load
        df1['RIM_WIDTH'][0] = rim_width

        df5 = df1
        st.subheader("2) Specs")
        col2_1, col2_2, col2_3, col2_4, col2_5 = st.columns(5)

        # SIZE : NSW
        unit = 10
        Min = df5['SIZE_NSW_MM'][0] - unit
        Max = df5['SIZE_NSW_MM'][0] + unit * 2
        SIZE_NSW_MM = col2_1.multiselect('SIZE_NSW',
                                      list(np.arange(Min, Max, unit)), default=df5['SIZE_NSW_MM'])

        # SIZE : SERISE
        unit = 5
        Min = df5['SIZE_SERIES_MM'][0] - unit
        Max = df5['SIZE_SERIES_MM'][0] + unit * 2
        SIZE_SERIES_MM = col2_2.multiselect('SIZE_SERIES',
                                         list(np.arange(Min, Max, unit)), default=df5['SIZE_SERIES_MM'])

        # SIZE : INCH
        unit = 1
        Min = df5['SIZE_INCH'][0] - unit
        Max = df5['SIZE_INCH'][0] + unit * 2
        SIZE_INCH = col2_3.multiselect('SIZE_INCH',
                                       list(np.arange(Min, Max, unit)), default=df5['SIZE_INCH'])

        # PLY_LOCK
        # if "PLY_LOCK" in use_cols:
        #     if df5['PLY_LOCK'][0] in categoricals['PLY_LOCK']:
        #         PLY_LOCK = col2_4.multiselect('PLY_LOCK',
        #                                     list(np.sort(categoricals['PLY_LOCK'])),
        #                                     default=df5['PLY_LOCK'])
        #     else:
        #         PLY_LOCK = col2_4.multiselect('PLY_LOCK',
        #                                     list(np.sort(categoricals['PLY_LOCK'])),
        #                                     default=categoricals['PLY_LOCK'][0])
        #         st.write(f"PLY_LOCK: {df5['PLY_LOCK'][0]} isn't used in this AI/ML model")
        # else:
        #     st.write("PLY_LOCK isn't used in this AI/ML model")

        # # PRE_BELT_TYPE
        # if "PRE_BELT_TYPE" in use_cols:
        #     if df5['PRE_BELT_TYPE'][0] in categoricals['PRE_BELT_TYPE']:
        #         PRE_BELT_TYPE = col2_5.multiselect('PRE_BELT_TYPE',
        #                                            list(np.sort(categoricals['PRE_BELT_TYPE'])),
        #                                            default=df5['PRE_BELT_TYPE'])
        #     else:
        #         PRE_BELT_TYPE = col2_5.multiselect('PRE_BELT_TYPE',
        #                                            list(np.sort(categoricals['PRE_BELT_TYPE'])),
        #                                            default=categoricals['PRE_BELT_TYPE'][0])
        #         st.write(f"PRE_BELT_TYPE: {df5['PRE_BELT_TYPE'][0]} isn't used in this AI/ML model")
        # else:
        #     st.write("PRE_BELT_TYPE isn't used in this AI/ML model")

        st.markdown("""---""")
        col3_1, col3_2, col3_3, col3_4, col3_5 = st.columns(5)

        # JLC_MATERIAL
        if "JLC_MATERIAL" in use_cols:
            if df5['JLC_MATERIAL'][0] in categoricals['JLC_MATERIAL']:
                JLC_MATERIAL = col3_1.multiselect('JLC_MATERIAL',
                                                  list(np.sort(categoricals['JLC_MATERIAL'])),
                                                  default=df5['JLC_MATERIAL'])
            else:
                JLC_MATERIAL = col3_1.multiselect('JLC_MATERIAL',
                                                  list(np.sort(categoricals['JLC_MATERIAL'])),
                                                  default=categoricals['JLC_MATERIAL'][0])
                st.write(f"JLC_MATERIAL: {df5['JLC_MATERIAL'][0]} isn't used in this AI/ML model")
        else:
            st.write("JLC_MATERIAL isn't used in this AI/ML model")

        # BT1_ANGLE
        unit = 3
        # Min = df5['BT1_ANGLE'][0] - unit
        # Max = df5['BT1_ANGLE'][0] + unit * 2
        BT1_ANGLE = col3_2.multiselect('BT1_ANGLE',
                                       list(np.arange(24, 39, unit)), default=df5['BT1_ANGLE'])
        # BT1_WIDTH
        unit = 5
        Min = df5['BT1_WIDTH'][0] - unit
        Max = df5['BT1_WIDTH'][0] + unit * 2
        BT1_WIDTH = col3_3.multiselect('BT1_WIDTH',
                                       list(np.arange(Min, Max, unit)), default=df5['BT1_WIDTH'])

        # BT1_MATERIAL
        # if "BT1_MATERIAL" in use_cols:
        #     if df5['BT1_MATERIAL'][0] in categoricals['BT1_MATERIAL']:
        #         BT1_MATERIAL = col3_4.multiselect('BT1_MATERIAL',
        #                                           list(np.sort(categoricals['BT1_MATERIAL'])),
        #                                           default=df5['BT1_MATERIAL'])
        #     else:
        #         BT1_MATERIAL = col3_4.multiselect('BT1_MATERIAL',
        #                                           list(np.sort(categoricals['BT1_MATERIAL'])),
        #                                           default=categoricals['BT1_MATERIAL'][0])
        #         st.write(f"BT1_MATERIAL: {df5['BT1_MATERIAL'][0]} isn't used in this AI/ML model")
        # else:
        #     st.write("BT1_MATERIAL isn't used in this AI/ML model")

        # BT1_EPI
        # unit = 3
        # Min = df5['BT1_EPI'][0] - unit
        # Max = df5['BT1_EPI'][0] + unit * 2
        # BT1_EPI = col3_5.multiselect('BT1_EPI',
        #                              list(np.arange(Min, Max, unit)), default=df5['BT1_EPI'])


        st.markdown("""---""")
        col4_1, col4_2, col4_3, col4_4, col4_5 = st.columns(5)

        # HS
        unit = 1
        # Min = df5['HS'][0] - unit * 2
        # Max = df5['HS'][0] + unit * 3
        HS = col4_1.multiselect('HS',
                                     list(np.arange(55, 81, unit)), default=df5['HS'])

        # M10
        unit = 0.5
        Min = df5['M10'][0] - unit * 5
        Max = df5['M10'][0] + unit * 5
        M10 = col4_2.multiselect('M10',
                                    list(np.arange(Min, Max, unit)), default=df5['M10'])

        # CTB_STEP_GAUGE
        unit = 0.5
        Min = df5['CTB_STEP_GAUGE'][0] - unit * 2
        Max = df5['CTB_STEP_GAUGE'][0] + unit * 2
        CTB_STEP_GAUGE = col4_3.multiselect('CTB_STEP_GAUGE',
                                 list(np.arange(Min, Max, unit)), default=df5['CTB_STEP_GAUGE'])

        # C01_EPI
        # unit = 3
        # Min = df5['C01_EPI'][0] - unit
        # Max = df5['C01_EPI'][0] + unit * 2
        # C01_EPI = col4_4.multiselect('C01_EPI',
        #                                  list(np.arange(Min, Max, unit)), default=df5['C01_EPI'])

        # SEASON
        # if "SEASON" in use_cols:
        #     if df5['SEASON'][0] in categoricals['SEASON']:
        #         SEASON = col4_5.multiselect('SEASON',
        #                                           list(np.sort(categoricals['SEASON'])),
        #                                           default=df5['SEASON'])
        #     else:
        #         SEASON = col4_5.multiselect('SEASON',
        #                                           list(np.sort(categoricals['SEASON'])),
        #                                           default=categoricals['SEASON'][0])
        #         st.write(f"SEASON: {df5['SEASON'][0]} isn't used in this AI/ML model")
        # else:
        #     st.write("SEASON isn't used in this AI/ML model")

        submitted2 = st.form_submit_button('Execute')
        ######################################################################################
    if submitted2:
        # st.write(df1)
        try:
            df_result = Prediction(df1, BestModel)
            df_result_all = Prediction(all_data[use_cols], BestModel)
            df_result_all['GM_TNAI_80'] = all_data['GM_TNAI_80']
        except:
            st.warning("There is not enough 'Value(level) training model'.")

            ######################################################################################
            # DOE STUDY
            # ['AIR', 'TEST_LOAD', 'RIM_WIDTH', 'SIZE_NSW_MM', 'SIZE_SERIES_MM',
            # 'PATT_PAR1', 'PATT_PAR2', 'PKX', 'CTB_STEP_GAUGE', 'HS', 'M10',
            # 'JLC_MATERIAL', 'BT1_WIDTH', 'BT1_ANGLE', 'PLY_LOCK']
        GM_PV = 0
        if GM_PV == 1:
            # GM P-V CHART VALIDATION ----------------------------------------------------------------
            PVchart_temp0 = pd.read_csv('./GM TNAI Model/2024_GM_PVchart_DrawingCheck.txt', sep='\t')
            PVchart_temp0['Air_kgf_cm2'] = PVchart_temp0['Air_kgf_cm2'].astype(float)
            PVchart_temp0['Test_load_kgf'] = PVchart_temp0['Test_load_kgf'].astype(float)
            PVchart_spec_list = PVchart_temp0['SPEC_NO']
            PVchart_spec_list = PVchart_spec_list.values.tolist()

            df_result_PV = pd.DataFrame()
            for i in range(len(PVchart_spec_list)):
                spec_no = PVchart_spec_list[i]
                query1 = f"SELECT * \
                                        FROM RCXUSER.da_spec\
                                        WHERE da_spec.spec_no = '{spec_no}'"
                temp0 = pd.read_sql(query1, connection)

                air = PVchart_temp0['Air_kgf_cm2'][i]
                test_load = PVchart_temp0['Test_load_kgf'][i]
                rim_width = PVchart_temp0['Rim_width'][i]
                Drawing_no = PVchart_temp0['Drawing_no'][i]

                [temp1, Pattern_Image, df_mold_pattern, df_product] = Data_Cleaning(temp0, air, test_load, rim_width, use_cols, scale_cols)

                df_result = Prediction(temp1, BestModel)
                df_result_PV = pd.concat([df_result_PV, df_result], ignore_index=True)

                print(f"{round((i+1)/(len(PVchart_spec_list)) * 100)}% 진행됨.")

            df_result_PV_2 = pd.merge(df_result_PV, PVchart_temp0, on='SPEC_NO')
            df_result_PV_2 = df_result_PV_2.dropna(subset = 'Physical(GM)').reset_index(drop=True)
            df_result_PV_2['Drawing_check'] = df_result_PV_2.apply(lambda row: 'OK' if row['SPEC_NO'] == row['SPEC_NO_drawing'] else 'Similar', axis=1)
            TNAI_correction_factor = 6.3
            df_result_PV_2['Prediction(GM)'] = df_result_PV_2['GM_TNAI_80_(XGB)'] + TNAI_correction_factor
            df_result_PV_2['TNAI_ERROR'] = df_result_PV_2['Physical(GM)'] - df_result_PV_2['Prediction(GM)']
            count_within_range = df_result_PV_2[(df_result_PV_2['TNAI_ERROR'] >= -4) & (df_result_PV_2['TNAI_ERROR'] <= 4)].shape[0]
            total_count = df_result_PV_2.shape[0]
            probability_within_range = (count_within_range / total_count) * 100
            print(f"TNAI 실측대비 예측값 차이가 4 이내일 확률 : {probability_within_range:.0f} %")

            # 선형 회귀 모델 학습
            model = LinearRegression()
            model.fit(df_result_PV_2[['Physical(GM)']], df_result_PV_2['Prediction(GM)'])
            y_pred = model.predict(df_result_PV_2[['Physical(GM)']])

            # R² 값 계산
            r_squared = model.score(df_result_PV_2[['Physical(GM)']], df_result_PV_2['Prediction(GM)'])

            # Figure와 하위 플롯 생성
            fig = plt.figure(figsize=(15, 7))
            gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1])

            # 첫 번째 하위 플롯: 상관성을 보여주는 플롯
            ax1 = fig.add_subplot(gs[:,0])
            ax1.plot(df_result_PV_2['Physical(GM)'], y_pred, color='green', label=f'Linear fit: $R^2$ = {r_squared:.2f}')
            sns.scatterplot(x='Physical(GM)', y='Prediction(GM)', data=df_result_PV_2, ax=ax1)
            ax1.set_title('Scatter Plot of TNAI Prediction data')
            ax1.set_xlabel('Physical(GM)')
            ax1.set_ylabel('Prediction(GM)')
            ax1.set_xlim(round(min(df_result_PV_2[['Physical(GM)', 'Prediction(GM)']].min()) - 2),
                         round(max(df_result_PV_2[['Physical(GM)', 'Prediction(GM)']].max()) + 2))
            ax1.set_ylim(round(min(df_result_PV_2[['Physical(GM)', 'Prediction(GM)']].min()) - 2),
                         round(max(df_result_PV_2[['Physical(GM)', 'Prediction(GM)']].max()) + 2))
            ax1.plot([60, 100], [60, 100], color='yellow', linestyle='--', label='y=x')
            ax1.plot([60, 100], [64, 104], color='orange', linestyle='--', label='TNAI error 4')
            ax1.plot([60, 100], [56, 96], color='orange', linestyle='--', label='TNAI error 4')

            ax2 = fig.add_subplot(gs[0,1])
            bar_plot = sns.barplot(x=df_result_PV_2['SPEC_NO']+'_'+df_result_PV_2['Drawing_check'], y=df_result_PV_2['TNAI_ERROR'], ax=ax2)
            ax2.set_title('Bar Plot of TNAI ERROR')
            ax2.set_ylabel('TNAI_ERROR')
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
            # 세로축 값 기준으로 직선 형태의 선 추가
            threshold = 4  # 예시로 세로축 값 5에 직선 추가
            ax2.axhline(threshold, color='orange', linestyle='--')
            threshold = -4  # 예시로 세로축 값 5에 직선 추가
            ax2.axhline(threshold, color='orange', linestyle='--')

            # 데이터 레이블 추가
            for p in bar_plot.patches:
                bar_plot.annotate(format(p.get_height(), '.1f'),
                                  (p.get_x() + p.get_width() / 2., p.get_height()),
                                  ha='center', va='center',
                                  xytext=(0, 10),
                                  textcoords='offset points')

            ax3 = fig.add_subplot(gs[1,1])
            # 두 번째 하위 플롯: 확률을 텍스트로 표현
            ax3.text(0.5, 0.5, f'Probability within TNAI ERROR 4  : {probability_within_range:.0f} %',
                     horizontalalignment='center', verticalalignment='center', fontsize=12, transform=ax3.transAxes)
            ax3.axis('off')  # 축 숨기기

            # Figure 제목 설정
            fig.suptitle('GM TNAI P-V Validation_2024', fontsize=16)

            # csv 파일로 예측된 결과포함 테이블 저장
            today = date.today()
            df_result_PV_2.to_csv(f"df_result_PV_{today}.csv")

            # Figure 표시
            plt.tight_layout()
            plt.show()

        # # DOE 부분 --------------------------------------------------------------------------------
        TEST_LOAD = [test_load]
        RIM_WIDTH = [rim_width]
        AIR = [air]
        EMB_PERFORMANCE_0001 = [df5["EMB_PERFORMANCE_0001"][0]]
        EMB_PERFORMANCE_0002 = [df5["EMB_PERFORMANCE_0002"][0]]
        EMB_PERFORMANCE_0003 = [df5["EMB_PERFORMANCE_0003"][0]]
        EMB_VIT_MAE_PCA_1 = [df5["EMB_VIT_MAE_PCA_1"][0]]
        EMB_VIT_MAE_PCA_2 = [df5["EMB_VIT_MAE_PCA_2"][0]]
        EMB_VIT_MAE_PCA_3 = [df5["EMB_VIT_MAE_PCA_3"][0]]
        EMB_VIT_MAE_PCA_4 = [df5["EMB_VIT_MAE_PCA_4"][0]]
        EMB_VIT_MAE_PCA_5 = [df5["EMB_VIT_MAE_PCA_5"][0]]
        EMB_VIT_MAE_PCA_6 = [df5["EMB_VIT_MAE_PCA_6"][0]]
        EMB_VIT_MAE_PCA_7 = [df5["EMB_VIT_MAE_PCA_7"][0]]
        EMB_VIT_MAE_PCA_8 = [df5["EMB_VIT_MAE_PCA_8"][0]]
        EMB_VIT_MAE_PCA_9 = [df5["EMB_VIT_MAE_PCA_9"][0]]
        EMB_VIT_MAE_PCA_10 = [df5["EMB_VIT_MAE_PCA_10"][0]]
        EMB_VIT_MAE_PCA_11 = [df5["EMB_VIT_MAE_PCA_11"][0]]
        EMB_VIT_MAE_PCA_12 = [df5["EMB_VIT_MAE_PCA_12"][0]]
        EMB_VIT_MAE_PCA_13 = [df5["EMB_VIT_MAE_PCA_13"][0]]
        EMB_VIT_MAE_PCA_14 = [df5["EMB_VIT_MAE_PCA_14"][0]]
        doe_items = []
        for item in use_cols:
            # st.write(item)
            exec(f"a = len({item})")
            if a > 1:
                doe_items.append(item)
        # st.write(doe_items)
        if len(doe_items) > 0:
            col_names = []
            levels_list = []
            for column in doe_items:
                exec(f"levels_list.append(len({column}))")
                col_names.append(column)
            levels = np.array(levels_list)

            n = len(levels)  # number of factors
            nb_lines = np.prod(levels)  # number of trial conditions
            H = np.zeros((nb_lines, n))

            level_repeat = 1
            range_repeat = np.prod(levels)
            for i in range(n):
                range_repeat = range_repeat // levels[i]
                lvl = []
                for j in range(levels[i]):
                    lvl += [j] * level_repeat
                rng = lvl * range_repeat
                level_repeat = level_repeat * levels[i]
                H[:, i] = rng

            df_doe = H
            df_doe = pd.DataFrame(H, columns=col_names)

            i = 0
            for colmn in col_names:
                # print(i)
                for j in range(int(levels[i])):
                    # print(j)
                    # df_doe[colmn] = df_doe.copy()[colmn].replace(j, doe_items[i][1][j])
                    exec(f"df_doe[colmn] = df_doe.copy()[colmn].replace(j, {doe_items[i]}[j])")
                i = i + 1

            for col in use_cols:
                if col not in col_names:
                    df_doe[col] = df5[col][0]
            # st.write(df_doe)
            df_doe_1 = df_doe.astype('object')  # Data Type을 모두 object로 통일화 하기
            for col in scale_cols:
                df_doe_1 = df_doe_1.astype({col: 'float16'})
            df_doe_1.columns = df_doe_1.columns.astype(str)

            df_doe2 = Prediction(df_doe_1, BestModel)

            # GM TNAI 80 예측값 열을 가장 왼쪽으로 옮긴 뒤 내림차순 정렬
            first_column = 'GM_TNAI_80_(XGB)'
            cols = [first_column] + [col for col in df_doe2.columns if col != first_column]
            df_doe2 = df_doe2[cols]
            df_doe2 = df_doe2.sort_values(by='GM_TNAI_80_(XGB)', ascending=False)

        st.markdown("""---""")
        st.subheader("4. PREDICTED RESULT")
        result_tab1, result_tab2 = st.tabs(["Final result", "Pitch analysis"])
        with result_tab1:
            result1_col1, result1_col2 = st.columns(2)
            with result1_col1:
                st.write('')
                st.write('GM_TNAI 80 (HK)')
                np.set_printoptions(precision=1)
                st.subheader(round(df_result['GM_TNAI_80_(XGB)'].values[0],1))
            with result1_col2:
                st.write('')
                st.write('GM_TNAI 80 (GM) ★')
                st.subheader(round(df_result['GM_TNAI_80_(XGB)'].values[0]+6.3,1))
            st.write('')
            st.write('※ This is original specs result. If you execute DOE study, check it at the bottom.')
            st.write('※ GMSP80 = GM Sound Power 80km/h, TNAI = Tire Noise Articulation Index (Higher is Better)')
            st.write('※ TNAI Correlation Factor(2025, AI+MAE) >> GM = HK + 6.3')
            st.markdown("""---""")

            result2_col1, result2_col2 = st.columns(2)
            with result2_col1:
                st.write('▼ Used pattern image (Scale 1:3)')
                try:
                    st.image(uploaded_croppedImage)
                except:
                    st.image(croppedImage)
                st.write(f"Mold Skid Depth : {df_product['MOLD_SD'][0]} mm")
                st.write(f"Drawing name : {df_product['FILENAME'][0]} ")
                st.write('')
            with result2_col2:
                fig = plt.figure(figsize=(6, 6))
                fig.suptitle(f"Target tire position ")
                plot1 = fig.add_subplot()
                plot1.title.set_text(f"GM_TNAI_80 , (Total data : {len(all_data['GM_TNAI_80'])} EA)")
                plot1.plot(all_data['GM_TNAI_80'], df_result_all['GM_TNAI_80_(XGB)'], 's',
                           color='Grey',
                           markersize=2, alpha=0.5)
                plot1.plot(float(df_result['GM_TNAI_80_(XGB)']),
                           float(df_result['GM_TNAI_80_(XGB)']), 's', color='Blue',
                           markersize=7, alpha=1)
                plot1.text(float(df_result['GM_TNAI_80_(XGB)']),
                           float(df_result['GM_TNAI_80_(XGB)']) + 0.5,
                           '%.1f' % float(df_result['GM_TNAI_80_(XGB)']), ha='center', va='bottom',
                           size=12)
                if len(doe_items) > 0:
                    plot1.plot(df_doe2['GM_TNAI_80_(XGB)'].astype('float'),
                               df_doe2['GM_TNAI_80_(XGB)'].astype('float'), 's', color='Red',
                               markersize=4, alpha=1)
                plt.xlabel('Tested (GM_TNAI)')
                plt.ylabel('Predicted (GM_TNAI)')

                fig.tight_layout(pad=2.0)
                st.pyplot(fig)

            st.success('Prediction complete')

            if len(doe_items) > 0:
                max_value = df_doe2['GM_TNAI_80_(XGB)'].max()
                # Row highlight
                def highlight_max(row):
                    return ['background-color: yellow' if row['GM_TNAI_80_(XGB)'] == max_value else '' for _ in row]
                styled_df_doe2 = df_doe2.style.apply(highlight_max, axis=1).format(precision=1)
                st.write(styled_df_doe2)
                st.write('If you download this table into csv file, please push down load button')


                def convert_df(df):
                    return df.to_csv(index=False).encode('utf-8')


                csv_file = convert_df(df_doe2)
                st.download_button(
                    "Press to Download",
                    csv_file,
                    f"DOE_Result_PBN_data_{spec_no}_load{test_load}_air{air}.csv",
                    "text/csv",
                    key='download-csv'
                )

            # Model 에서 쓰인 Pickle 데이터의 각 Season별 Data 수를 표로 보여줌
            with st.expander("▣ Model Data Status", expanded=False):
                Used_data = pd.DataFrame()
                Value_list = [len(all_data['SEASON']), all_data['SEASON'].value_counts()[0], all_data['SEASON'].value_counts()[1],
                              all_data['SEASON'].value_counts()[2], _PN_data_path.split('\\',10)[-1]]
                # Oct_list = [len(df_predicted_Oct['SEASON']), df_predicted_Oct['SEASON'].value_counts()[0], df_predicted_Oct['SEASON'].value_counts()[1],
                #             df_predicted_Oct['SEASON'].value_counts()[2], Oct_PN_data_path.split('\\',10)[-1]]
                Used_data["GM TNAI Value Data"] = Value_list
                # Used_data["GM TNAI Oct. Data"] = Oct_list
                Used_data.index = ["Total", "Summer", "All season", "Winter", "Pickle data version"]
                st.write(Used_data)

            with result_tab2:
                st.subheader('')
                # Pitch 배열 분석
                p_len1 = list(df_mold_pattern.iloc[0, 2:12].unique())
                p_len2 = list(df_mold_pattern.iloc[0, 12:22].unique())  # Lower
                p_len3 = list(df_mold_pattern.iloc[0, 22:32].unique())  # Upper

                if len(p_len3) == 1:
                    print("Single Pitch Sequence")
                    total_n = int(df_mold_pattern['NUMBEROFPITCHES'][0])

                    if len(p_len3) == 1:
                        p_len = df_mold_pattern.iloc[0, 2:12]
                    else:
                        p_len = df_mold_pattern.iloc[0, 12:22]

                    # define delta / uniform function
                    pit_kind_n = p_len.value_counts().sum()  # p_len[0]
                    pitch_delta = {}
                    pitch_uniform = {}
                    for i in range(pit_kind_n):
                        exec(f"pitch_delta[{i}] = np.zeros((round(p_len[{i}]*100)))")
                        exec(f"pitch_delta[{i}][0] = 1")
                        exec(f"pitch_uniform[{i}] = np.ones((round(p_len[{i}] * 100))) * p_len[{i}]")
                    # temp_seq = df_pattern['PITCHSEQUENCE'][0]
                    pitch_signal_delta = np.array([])
                    pitch_signal_uniform = np.array([])
                    count = 1
                    for s in df_mold_pattern['PITCHSEQUENCE'][0]:  # s='1'
                        if s == 'A':
                            s = '10'
                        p = int(s) - 1
                        if count == 1:
                            exec(f"pitch_signal_delta = pitch_delta[{p}]")
                            exec(f"pitch_signal_uniform = pitch_uniform[{p}]")
                        else:
                            exec(f"pitch_signal_delta = np.concatenate([pitch_signal_delta, pitch_delta[{p}]])")
                            exec(f"pitch_signal_uniform = np.concatenate([pitch_signal_uniform, pitch_uniform[{p}]])")
                        count = count + 1

                    # FFT 분석
                    [fft_vals1, fft_norm1, fft_abs1] = fft_cal(pitch_signal_delta)
                    [fft_vals2, fft_norm2, fft_abs2] = fft_cal(pitch_signal_uniform)
                    [peak_Pyy, spv, bandwidth, fft_abs1_n, fft_abs2_n] = analysis_parameter(fft_abs1, fft_abs2, total_n)
                    n_peak_Pyy = peak_Pyy[0]
                    n_spv = spv
                    n_bandwidth = bandwidth
                    PNI = PNI_Cal(n_peak_Pyy, n_bandwidth)

                    limit1 = list(
                        [0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04])
                    limit2 = list(
                        [0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.03,0.03,0.03,0.03,0.02,0.02,0.02,0.02,0.02,0.025,0.025,0.025,0.025])

                    st.subheader('Pitch analysis result (Single)')
                    col1_1, col1_2 = st.columns(2)
                    with col1_1:
                        st.write('')
                        st.write('PNI')
                        st.subheader(f"{round(PNI, 3)}")
                        st.write('')
                        st.write('※ Max peak')
                        st.write(f"{round(n_peak_Pyy, 2)}")
                        st.write('※ Bandwidth')
                        st.write(f"{n_bandwidth}")
                        st.write('※ Total pitch no. [EA]')
                        st.write(f"{total_n}")
                        st.write('※ L/S Pitch ratio')
                        st.write(f"{round(p_len[pit_kind_n-1] / p_len[0], 3)}")
                    with col1_2:
                        aa = np.arange(0, 2 * np.pi, 2 * np.pi / pitch_signal_uniform.shape[0])
                        fig = plt.figure()
                        fig.suptitle('Pitch sequence in polar graph')
                        ax = fig.add_subplot(projection='polar')
                        ax.plot(aa, pitch_signal_uniform)
                        plt.grid(True, linestyle='--')
                        fig.tight_layout(pad=1.0)
                        st.pyplot(fig)
                    st.write('')
                    col2_1, col2_2 = st.columns(2)
                    with col2_1:
                        fig = plt.figure()
                        fig.suptitle('Delta pitch FFT')
                        ax = fig.add_subplot()
                        x_labels = list(np.arange(1, 201, 1))
                        values = fft_abs1_n[:200]
                        plt.plot(x_labels, values, label='Target tire')
                        plt.legend()
                        plt.xlabel('Harmonic')
                        plt.ylabel('Amplitude')
                        plt.ylim([0, np.max(values)*1.5])
                        plt.grid(True, linestyle='--')
                        st.pyplot(fig)
                    with col2_2:
                        fig = plt.figure()
                        fig.suptitle('Pitch uniformity analysis')
                        ax = fig.add_subplot()
                        x_labels = list(np.arange(1, 21, 1))
                        values = fft_abs1_n[:20]
                        ax.plot(x_labels, values, '-s', markersize=4, linewidth=4, alpha=0.8, label='Target tire')
                        ax.plot(x_labels, limit1, '-s', color='green', markersize=3, linewidth=2, alpha=0.5, label='DB standard')
                        ax.plot(x_labels, limit2, '-s', color='orange', markersize=3, linewidth=2, alpha=0.5, label='AW & WI')
                        plt.xticks(x_labels)
                        plt.legend()
                        plt.xlabel('Harmonic')
                        plt.ylabel('Amplitude')
                        plt.ylim([0, 0.05])
                        plt.grid(visible=True, axis='y', linewidth=0.5, linestyle='--')
                        st.pyplot(fig)
                    # st.write(f"Peak Level: {peak_Pyy[0]} / Second Power Variation: {spv} / Band Width: {bandwidth}")

                else:
                    print("Dual Pitch Sequence")
                    total_n_L = int(df_mold_pattern['NUMBEROFPITCHES'][0])
                    total_n_U = int(df_mold_pattern['NUMBEROFUPPERPITCHES'][0])

                    p_len_L = df_mold_pattern.iloc[0, 12:22]
                    p_len_U = df_mold_pattern.iloc[0, 22:32]

                    seq_L = df_mold_pattern['PITCHSEQUENCE'][0]
                    seq_U = df_mold_pattern['UPPERPITCHSEQUENCE'][0]

                    # Lower pitch
                    p_len = p_len_L
                    total_n = total_n_L
                    seq = seq_L

                    pit_kind_n = p_len.value_counts().sum()  # p_len[0]
                    pit_kind_n_L = pit_kind_n
                    pitch_delta = {}
                    pitch_uniform = {}
                    for i in range(pit_kind_n):
                        exec(f"pitch_delta[{i}] = np.zeros((round(p_len[{i}]*100)))")
                        exec(f"pitch_delta[{i}][0] = 1")
                        exec(f"pitch_uniform[{i}] = np.ones((round(p_len[{i}] * 100))) * p_len[{i}]")
                    # temp_seq = df_pattern['PITCHSEQUENCE'][0]
                    pitch_signal_delta = np.array([])
                    pitch_signal_uniform = np.array([])
                    count = 1
                    for s in seq:  # s='1'
                        if s == 'A':
                            s = '10'
                        p = int(s) - 1
                        if count == 1:
                            exec(f"pitch_signal_delta = pitch_delta[{p}]")
                            exec(f"pitch_signal_uniform = pitch_uniform[{p}]")
                        else:
                            exec(f"pitch_signal_delta = np.concatenate([pitch_signal_delta, pitch_delta[{p}]])")
                            exec(f"pitch_signal_uniform = np.concatenate([pitch_signal_uniform, pitch_uniform[{p}]])")
                        count = count + 1

                    # FFT 분석
                    [fft_vals1, fft_norm1, fft_abs1] = fft_cal(pitch_signal_delta)
                    [fft_vals2, fft_norm2, fft_abs2] = fft_cal(pitch_signal_uniform)
                    [peak_Pyy, spv, bandwidth, fft_abs1_n, fft_abs2_n] = analysis_parameter(fft_abs1, fft_abs2, total_n)
                    n_peak_Pyy = peak_Pyy[0]
                    n_spv = spv
                    n_bandwidth = bandwidth
                    PNI = PNI_Cal(n_peak_Pyy, n_bandwidth)
                    [PNI_L, n_peak_Pyy_L, n_spv_L, n_bandwidth_L, fft_abs1_n_L, fft_abs2_n_L] = [PNI, n_peak_Pyy, n_spv, n_bandwidth, fft_abs1_n, fft_abs2_n]
                    pitch_signal_uniform_L = pitch_signal_uniform

                    # Upper pitch
                    p_len = p_len_U
                    total_n = total_n_U
                    seq = seq_U

                    pit_kind_n = p_len.value_counts().sum()  # p_len[0]
                    pit_kind_n_U = pit_kind_n
                    pitch_delta = {}
                    pitch_uniform = {}
                    for i in range(pit_kind_n):
                        exec(f"pitch_delta[{i}] = np.zeros((round(p_len[{i}]*100)))")
                        exec(f"pitch_delta[{i}][0] = 1")
                        exec(f"pitch_uniform[{i}] = np.ones((round(p_len[{i}] * 100))) * p_len[{i}]")
                    # temp_seq = df_pattern['PITCHSEQUENCE'][0]
                    pitch_signal_delta = np.array([])
                    pitch_signal_uniform = np.array([])
                    count = 1
                    for s in seq:  # s='1'
                        if s == 'A':
                            s = '10'
                        p = int(s) - 1
                        if count == 1:
                            exec(f"pitch_signal_delta = pitch_delta[{p}]")
                            exec(f"pitch_signal_uniform = pitch_uniform[{p}]")
                        else:
                            exec(f"pitch_signal_delta = np.concatenate([pitch_signal_delta, pitch_delta[{p}]])")
                            exec(f"pitch_signal_uniform = np.concatenate([pitch_signal_uniform, pitch_uniform[{p}]])")
                        count = count + 1

                    # FFT 분석
                    [fft_vals1, fft_norm1, fft_abs1] = fft_cal(pitch_signal_delta)
                    [fft_vals2, fft_norm2, fft_abs2] = fft_cal(pitch_signal_uniform)
                    [peak_Pyy, spv, bandwidth, fft_abs1_n, fft_abs2_n] = analysis_parameter(fft_abs1, fft_abs2, total_n)
                    n_peak_Pyy = peak_Pyy[0]
                    n_spv = spv
                    n_bandwidth = bandwidth
                    PNI = PNI_Cal(n_peak_Pyy, n_bandwidth)
                    [PNI_U, n_peak_Pyy_U, n_spv_U, n_bandwidth_U, fft_abs1_n_U, fft_abs2_n_U] = [PNI, n_peak_Pyy, n_spv, n_bandwidth, fft_abs1_n, fft_abs2_n]
                    pitch_signal_uniform_U = pitch_signal_uniform

                    # Dual parameter calculation
                    fft_abs1_n_Dual = (fft_abs1_n_L[1:201] + fft_abs1_n_U[1:201]) / 2
                    total_n_Daul = int(round((total_n_L+total_n_U)/2))
                    peak_Pyy_Dual = []
                    peak_Pyy_Dual.append(np.max(fft_abs1_n_Dual))
                    fft_abs1_n_Dual_sm = (fft_abs1_n_Dual[1:197] + fft_abs1_n_Dual[2:198] + fft_abs1_n_Dual[3:199]) / 3
                    p1 = np.max(fft_abs1_n_Dual_sm[total_n_Daul - 20:total_n_Daul + 20])
                    t1 = np.where(fft_abs1_n_Dual_sm == p1)[0][0]
                    arr1 = np.where(fft_abs1_n_Dual_sm[1:201] < p1 / 2)[0]
                    low = arr1[np.where(arr1 - t1 < 0)][-1]
                    high = arr1[np.where(arr1 - t1 > 0)][0]
                    bandwidth_Dual = high - low
                    PNI_Dual = PNI_Cal(peak_Pyy_Dual[0], bandwidth_Dual)


                    limit1 = list(
                        [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04,
                         0.04, 0.04, 0.04, 0.04])
                    limit2 = list(
                        [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02,
                         0.025, 0.025, 0.025, 0.025])

                    st.subheader('Pitch analysis result (Dual)')
                    col1_1, col1_2, col1_3 = st.columns([1,2,2])
                    with col1_1:
                        st.write('')
                        st.write('PNI (Dual)')
                        st.subheader(f"{round(PNI_Dual, 3)}")
                        st.write('')
                        st.write('※ Max peak (Dual)')
                        st.write(f"{round(peak_Pyy_Dual[0], 2)}")
                        st.write('※ Bandwidth (Dual)')
                        st.write(f"{bandwidth_Dual}")
                        st.write('※ Total pitch no. [EA]')
                        st.write(f"Lower : {total_n_L} / Upper : {total_n_U} ")
                        st.write('※ L/S Pitch ratio')
                        st.write(f"Lower : {round(p_len_L[pit_kind_n_L - 1] / p_len_L[0], 3)} / Upper : {round(p_len_U[pit_kind_n_U - 1] / p_len_U[0], 3)}")
                    with col1_2:
                        aa = np.arange(0, 2 * np.pi, 2 * np.pi / pitch_signal_uniform_L.shape[0])
                        fig = plt.figure()
                        fig.suptitle('Lower pitch sequence in polar graph')
                        ax = fig.add_subplot(projection='polar')
                        ax.plot(aa, pitch_signal_uniform_L, color='green', markersize=1.5)
                        plt.grid(True, linestyle='--')
                        fig.tight_layout(pad=1.0)
                        st.pyplot(fig)
                    with col1_3:
                        aa = np.arange(0, 2 * np.pi, 2 * np.pi / pitch_signal_uniform_U.shape[0])
                        fig = plt.figure()
                        fig.suptitle('Upper pitch sequence in polar graph')
                        ax = fig.add_subplot(projection='polar')
                        ax.plot(aa, pitch_signal_uniform_U, color='skyblue', markersize=1.5)
                        plt.grid(True, linestyle='--')
                        fig.tight_layout(pad=1.0)
                        st.pyplot(fig)

                    st.write('')
                    col2_1, col2_2 = st.columns(2)
                    with col2_1:
                        fig = plt.figure()
                        fig.suptitle('Delta pitch FFT (Dual)')
                        ax = fig.add_subplot()
                        x_labels = list(np.arange(1, 201, 1))
                        plt.plot(x_labels, fft_abs1_n_L[:200], color='green', linewidth=1.5, alpha=0.5,
                                 label='Target tire (Lower)')
                        plt.plot(x_labels, fft_abs1_n_U[:200], color='skyblue', linewidth=1.5, alpha=0.8,
                                 label='Target tire (Upper)')
                        plt.plot(x_labels, (fft_abs1_n_L[:200] + fft_abs1_n_U[:200]) / 2, linewidth=3, alpha=0.8,
                                 label='Target tire (Dual)')
                        plt.legend()
                        plt.xlabel('Harmonic')
                        plt.ylabel('Amplitude')
                        plt.ylim([0, np.max(fft_abs1_n_U[:200]) * 1.5])
                        plt.grid(True, linestyle='--')
                        st.pyplot(fig)
                    with col2_2:
                        fig = plt.figure()
                        fig.suptitle('Pitch uniformity analysis (Dual)')
                        ax = fig.add_subplot()
                        x_labels = list(np.arange(1, 21, 1))
                        ax.plot(x_labels, fft_abs1_n_L[:20], '-s', color='green', markersize=1.5, linewidth=1.5, alpha=0.5,
                                label='Target tire (Lower)')
                        ax.plot(x_labels, fft_abs1_n_U[:20], '-s', color='skyblue', markersize=1.5, linewidth=1.5,
                                alpha=0.8,
                                label='Target tire (Upper)')
                        ax.plot(x_labels, (fft_abs1_n_L[:20] + fft_abs1_n_U[:20]) / 2, '-s', markersize=4, linewidth=4,
                                alpha=0.8,
                                label='Target tire (Dual)')
                        ax.plot(x_labels, limit1, '-s', color='black', markersize=3, linewidth=2, alpha=0.5,
                                label='DB standard')
                        ax.plot(x_labels, limit2, '-s', color='orange', markersize=3, linewidth=2, alpha=0.5,
                                label='AW & WI')
                        plt.xticks(x_labels)
                        plt.legend()
                        plt.xlabel('Harmonic')
                        plt.ylabel('Amplitude')
                        plt.ylim([0, 0.05])
                        plt.grid(visible=True, axis='y', linewidth=0.5, linestyle='--')
                        st.pyplot(fig)
