# -*- coding: utf-8 -*-
"""
Created on 06/May/21
@author: GuiThalita 
"""

# %% 
import os
import sys
import numpy as np
import pandas as pd

pd.set_option('display.max_columns',100)
pd.set_option('precision', 3)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

from feature_engine.imputation import MeanMedianImputer, ArbitraryNumberImputer
from feature_engine.outliers import Winsorizer
from feature_engine.wrappers import SklearnTransformerWrapper
from feature_engine.discretisation import EqualFrequencyDiscretiser
from feature_engine.discretisation.arbitrary import ArbitraryDiscretiser
from feature_engine.selection import DropFeatures
from feature_engine.selection import SmartCorrelatedSelection
from feature_engine.selection import RecursiveFeatureAddition
from feature_engine.encoding import MeanEncoder

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import RobustScaler , MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline 
from sklearn.feature_selection import SequentialFeatureSelector

s_path = os.path.join("..", "explore")
sys.path.append(s_path)

import tm_teoriaMvto_base_prep as base_prep
import tm_teoriaMvto_label as tm_label
import tm_teoriaMvto_train as tm_train
import tm_teoriaMvto_ft_eng as ft_eng
# import tm_teoriaMvto_ft_sel as ft_sel

# models to test
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# optimizing ensemble
from skopt.space import Integer
from skopt.space import Real
from skopt.space import Categorical
from skopt.utils import use_named_args
from skopt import forest_minimize
from skopt.callbacks import DeltaYStopper


# %% configuration for labeling and loading
path_files = os.path.join("..", "explore", "historical_files")

# base prep control
EXPORT_X = False
EXPORT_Y = False

# params labeling
s_prefix = 'ft_'
s_lbl_type = 'c_binary'
f_th = 0.2
b_dist = True
b_percent = True
b_custom = True

# params loading
s_regime = 'mi_up'


#%% Get data - run ONLY ONCE (if EXPORT_X = True) 
if EXPORT_X:
  df_data = base_prep.import_sampling(path_files= os.path.join(path_files, 'FeaturesFiles'), prefix_files= s_prefix)

  # create new features before extracting samples - using full data needed by lag features
  df_data = ft_eng.BasicFeatures().transform(X=df_data)

  df_X = base_prep.ft_export(df_data, path_files= os.path.join(path_files, 'TrainFiles'), prefix= s_prefix)
  y_sc = base_prep.y_export(df_data, path_files= os.path.join(path_files, 'TrainFiles'), prefix= s_prefix)


# %% running labeling configs 
y_config = 'prefix_'+ s_prefix +'_type_'+ s_lbl_type +'_th_'+ str(f_th) +'_dist_'+ str(b_dist) +'_percent_'+str(b_percent)+'_custom_'+str(b_custom)

if EXPORT_Y:
  df_label = None 
  if EXPORT_X: df_label = y_sc

  # specify df_data = None (default) to load pickle from s_path + 'y_prep_data.pkl'
  label = tm_label.Labeling(df_data = df_label, label_type= s_lbl_type, b_dist_to_high= b_dist, 
                            s_path = os.path.join(path_files, 'TrainFiles'), prefix_files = s_prefix, th_label_y1=f_th, 
                            b_percent_freq=b_percent, b_custom_dir=b_custom)

  df_y = label.apply_label(s_model_return = s_regime)  # it will export 4 pickle files to label_obj.s_path
  # y_config = label.s_name

y_outfile = os.path.join(os.path.join(path_files, 'TrainFiles'), 'y_' + y_config +'.pkl')

if os.path.exists(y_outfile):
  print('y outfile ready for configuration: {}'.format(y_config))
else:
  print('WARNING: y outfile not found for configuration. Define EXPORT_Y = True and try again {}'.format(y_config))


#%% feature engineering - parameters for fit/transform classes
l_prop_08 = ['escora_bid_2.5_0.8', 'escora_ask_2.5_0.8', 'escora_bid_3.5_0.8', 
            'escora_ask_3.5_0.8', 'escora_bid_4.5_0.8', 'escora_ask_4.5_0.8', ]

l_prop_12 = ['escora_bid_2.5_1.2', 'escora_ask_2.5_1.2', 'escora_bid_3.5_1.2',
            'escora_ask_3.5_1.2', 'escora_ask_4.5_1.2', 'escora_bid_4.5_1.2', ]

l_prop_2 = [ 'escora_bid_2.5_2', 'escora_ask_2.5_2', 'escora_bid_3.5_2',
            'escora_ask_3.5_2', 'escora_bid_4.5_2', 'escora_ask_4.5_2', ]

l_col_log = ['agg_net_d', 'aggbig_net_d', 'vol_trd', 'vol_big', 'big_v', 'vol_trd_aux', 
   'vol_big_aux', 'big_v_aux', 'loc_agg_net_d', 'big_c', 'big_c_aux',
   'loc_aggbig_net_d', 'agg_net_m', 'agg_net_m_aux', 'abagg', 'abagg_aux',
   'aggbig_net_m', 'aggbig_net_m_aux', 'loc_agg_net_m', 'loc_aggbig_net_m',
   'loc_agg_net_m_aux', 'loc_aggbig_net_m_aux', 'loc_aggbig_c_m', 'loc_aggbig_v_m', 
   'loc_aggbig_c_m_aux', 'loc_aggbig_v_m_aux', 'abs_v', 'abs_c', 'aggpior_v', 'aggpior_v_aux', 
   'aggpior_c', 'aggpior_c_aux', 'agg_net_10', 'agg_net_40', 'agg_net_80', 'loc_agg_net_10',
   'aggbig_net_10', 'aggpior_DIF', 'aggpior_DIF_30', 'abs_DIF', 'abs_DIF_30',
   'abagg_10', 'aggpior_aux_DIF', ]

l_side_drop = ['big_c','big_v','aggpior_c','aggpior_v','loc_aggbig_c_m','loc_aggbig_v_m','pagg_c_best',
  'pagg_c_best_0.5','pagg_c_best_0.7','pagg_c_best_0.9','pagg_v_best','pagg_v_best_0.5','pagg_v_best_0.7',
  'pagg_v_best_0.9','abs_c','abs_v','int_c','int_c_0.6','int_c_0.7','int_c_0.8','int_c_0.9','int_dif_c',
  'int_v','int_v_0.6','int_v_0.7','int_v_0.8','int_v_0.9','int_dif_v','imp_c','imp_c_0.6','imp_c_0.7',
  'imp_c_0.8','imp_c_0.9','imp_v','imp_v_0.6','imp_v_0.7','imp_v_0.8','imp_v_0.9','escora_bid_2.5_1.2',
  'escora_bid_2.5_2','escora_ask_2.5_1.2','escora_ask_2.5_2','escora_bid_3.5_1.2','escora_bid_3.5_2',
  'escora_ask_3.5_1.2','escora_ask_3.5_2','escora_bid_4.5_0.8','escora_bid_4.5_2','escora_ask_4.5_0.8',
  'escora_ask_4.5_2','movesc_bid_2.5','movesc_ask_2.5','movesc_bid_2.5_0.5','movesc_ask_2.5_0.5',
  'movesc_bid_2.5_0.7','movesc_ask_2.5_0.7','movesc_bid_3.5','movesc_ask_3.5','movesc_bid_3.5_0.7',
  'movesc_ask_3.5_0.7','movesc_bid_3.5_0.9','movesc_ask_3.5_0.9','movesc_bid_4.5','movesc_ask_4.5',
  'movesc_bid_4.5_0.5','movesc_ask_4.5_0.5','movesc_bid_4.5_0.7','movesc_ask_4.5_0.7','depth_bid7','depth_ask7'
  ]

l_side_drop_aux = ['big_c_aux','big_v_aux','aggpior_c_aux','aggpior_v_aux','loc_aggbig_c_m_aux',
  'loc_aggbig_v_m_aux','pagg_c_best_aux','pagg_c_best_0.5_aux','pagg_c_best_0.7_aux',
  'pagg_c_best_0.9_aux','pagg_v_best_aux','pagg_v_best_0.5_aux', 'pagg_v_best_0.7_aux',
  'pagg_v_best_0.9_aux'
  ]

l_ft_aux = [
  'vol_trd_aux', 'n_trd_aux','vol_big_aux','n_big_aux','vol_big_ratio_aux','big_c_aux','big_v_aux',
  'aggpior_c_aux','aggpior_v_aux','aggimb_aux','aggimb_big_aux','n_aggimb_aux','agg_net_m_aux',
  'aggbig_net_m_aux','loc_aggbig_c_m_aux','loc_aggbig_v_m_aux','loc_agg_net_m_aux','loc_aggbig_net_m_aux',
  'loc_agg_imb_m_aux','loc_aggbig_imb_m_aux','pagg_c_best_aux','pagg_c_best_0.5_aux','pagg_c_best_0.7_aux',
  'pagg_c_best_0.9_aux','pagg_v_best_aux','pagg_v_best_0.5_aux','pagg_v_best_0.7_aux','pagg_v_best_0.9_aux',
  'abagg_aux','n_p_aux','aggpior_aux_DIF','pagg_aux_DIF'
  ]


#%% load & split train test
test_size = 0.2

if not EXPORT_X:
  try:
    df_X = tm_train.load_models('X_samples_'+s_prefix, os.path.join(path_files, 'TrainFiles'))  # search for path_files/s_regime.pkl
  except FileNotFoundError:
    raise FileNotFoundError('X data not ready! Check EXPORT_X.')

X = df_X.loc[df_X['model'] == s_regime].drop(columns='model')

if not EXPORT_Y:
  try:
    df_y = tm_train.load_models('y_' + y_config, os.path.join(path_files, 'TrainFiles'))
    y = df_y.loc[df_y['model'] == s_regime].drop(columns='model')
  except FileNotFoundError:
    raise FileNotFoundError('Y data not ready! Check EXPORT_Y.')
else:
  y = df_y.drop(columns='model')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, shuffle = False)

print('load_split: X, y data load and split complete!')


#%% fit transformers on train data
# DEBUG: prop_features contaning np.NaN = (pd.isna(X_train).sum()>0).sort_values(ascending=False).head(50)
X_train = ft_eng.PropImputer(0.8, l_prop_08).transform(X_train)
X_train = ft_eng.PropImputer(1.2, l_prop_12).transform(X_train)
X_train = ft_eng.PropImputer(2, l_prop_2).transform(X_train)

median_inputer = MeanMedianImputer(variables=['PA_down',])
X_train = median_inputer.fit_transform(X_train)

nan_imputer = ArbitraryNumberImputer(0.0, variables=['ohlc_10','ohlc_50'])
X_train = nan_imputer.fit_transform(X_train)

X_train = ft_eng.DifAll().transform(X_train)
X_train = ft_eng.LogVolume(l_col_log).transform(X_train)

if s_regime[:2] == 'mw':
  # for now, removing all side columns
  l_cols_drop = l_side_drop + l_ft_aux + ['smart_price', 'sspread']
else:
  l_cols_drop = l_side_drop + l_side_drop_aux + ['s_run', 'n_p_aux', 'smart_price', 'sspread']

# TODO: INCREMENTAR L COLS DROP TAMBEM QUANDO LABEL NAO FOR DIST-TO-HIGH, RETIRANDO AS COLUNAS DE VARIACAO DE PRECO!

X_train.drop(columns=l_cols_drop, inplace=True)

#%% preprocessing test data 
X_test = ft_eng.PropImputer(0.8, l_prop_08).transform(X_test)
X_test = ft_eng.PropImputer(1.2, l_prop_12).transform(X_test)
X_test = ft_eng.PropImputer(2, l_prop_2).transform(X_test)

X_test = median_inputer.transform(X_test)

X_test = nan_imputer.transform(X_test)

X_test = ft_eng.DifAll().transform(X_test)
X_test = ft_eng.LogVolume(l_col_log).transform(X_test)

X_test.drop(columns=l_cols_drop, inplace=True)


#%% features preparation 

# TODO: move function to ft_selection.py
def list_sub(lst1, lst2):
  final_list = list(set(lst1) - set(lst2))
  return final_list


def list_union(*l_lists):
  l_union = l_lists[0]
  if len(l_lists) > 1:
    for i in range(len(l_lists)-1):
      l_union = set(l_union) | set(l_lists[i+1])
  return list(l_union)

l_duplicate = ['loc_agg_net_m',]  # _2 ft comes from ft_eng.duplicate()

l_cap_1 = ['vewma_c_v', 'vewmag_dif', 'book_imb', 'book_imb_dif', ]

l_cap_5 = ['PA_up', 'PA_down', ]

l_cap_10 = ['n_big_aux', ]

l_cap_default = list_sub(X_train.columns.to_list(), list_union(l_cap_1, l_cap_5, l_cap_10))

l_bins_q2 = ['loc_agg_net_m_2',]  # _2 ft comes from ft_eng.duplicate()

l_bins_q4 = ['ohlc_10','ohlc_50',]

l_bins_q5 = ['loc_aggbig_net_m','vol_trd_aux','aggbig_net_m_aux','book_imb','rng_ewma_dif',
            'rng_ewma_dif_40','rng_ewma_dif_80','vewma_10','vewma_g_p_10','aggbig_net_10',
]

l_bins_q6 = ['agg_net_d', 'aggbig_net_d', 'loc_agg_imb_m_aux',]

l_bins_q8 = ['rng_ewma','vewma_g_p','vewmag_dif','n_trd_aux','abagg_aux','escora_bid_2.5_0.8','escora_ask_2.5_0.8',
            'escora_bid_3.5_0.8','escora_ask_3.5_0.8','escora_bid_4.5_1.2','escora_ask_4.5_1.2','movesc_bid_2.5_0.9',
            'movesc_ask_2.5_0.9','movesc_bid_3.5_0.5','movesc_ask_3.5_0.5','movesc_bid_4.5_0.9','movesc_ask_4.5_0.9',
            'msg_imb','rng_smart_10','imp_DIF_10','imp_DIF_50','agg_net_80','imp_FCAST_40','aggpior_DIF',
            'book_imb_mean_dif_cp','msg_imb_mean_10','escora_4.5_0.8_DIF',
]

l_bins_q10 = ['vol_big_ratio','aggimb','n_aggimb','agg_net_m','loc_aggbig_imb_m','abagg','n_p','vewma',
            'vewma_c_v', 'aggimb_aux','aggimb_big_aux','agg_net_m_aux','loc_aggbig_net_m_aux','smart_price_dif',
            'smart_price_50','rng_smart_50','agg_net_10','agg_net_40','loc_agg_net_10','int_DIF_10','abagg_10',
            'book_imb_mean_10','book_imb_mean_dif_lp','msg_imb_mean_40','msg_imb_mean_dif_lp','msg_imb_mean_dif_cp',
            'sspread_mean','movesc_2.5_0.7_DIF','msg_imb_mean_40_ABS','loc_agg_net_m',
] 

d_bins_arbitrary = {
            'n_trd':  [-0.01, 70, 180, 280, 380, 5000000],
            'vol_trd':  [0, 7.237, 7.55, 10000],
            'aggimb_big':		[-1.1, -0.4, 0.4, 1.1],
            'aggbig_net_m':		[-10000, -5, +10000],
            'chgfreq':		[-0.01, 0.167, 0.280, 1.01],
            'last_d_s':		[-1.01, -0.5, 0.5, 1.01],
            'loc_agg_net_d':		[-10000, -6.4, 5.2, 5.7, 10000],
            'loc_aggbig_net_d':		[-10000, -5.4, -3.8, 10000],
            'n_big_aux':	[-0.01, 3.1, 1000],
            'vol_big_ratio_aux':		[-0.01, 0.275, 0.520, 1.01],
            'loc_agg_net_m_aux':		[-10000, -4.451, -3.592, -2.435, 3.607, 10000] ,
            'loc_aggbig_imb_m_aux':		[-1.1, -0.99, -0.391, -0.0118, 0.4, 0.99, 1.1],
            'aggpior_DIF_30':		[-10000, -4.454, -4.111, -3.829, -3.26, -3.05, -3.04, -0.01, 0.01, 3.714, 10000],
            'abs_DIF':		[-10000, -4.796, -4.19, -0.1, 0.1, 2.398, 4.564, 10000],
            'pagg_DIF':	[-1.1, -0.95, -0.6, -0.4, -0.001, 0.001, 0.4, 0.6, 0.95, 1.1],
            'book_imb_dif':	[-1000, -27, -14, -6.8, -2.5, 0.1, 4.29, 11.2, 22, 1000],
            'book_imb_mean_40': [-10000, -30, -20, -16, 10000],
            'aggpior_aux_DIF':		[-10000, -4.331, -3.584, -3.045, -0.01, 0.01, 3.045, 10000] ,
            'pagg_aux_DIF':		[-1.1, -0.95, -0.5, -0.001, 0.001, 0.5, 0.95, 1.1],
            'escora_3.5_1.2_DIF':		[-1.1, -0.171, -0.0482, 0, 0.0482, 0.0927, 1.1],
            'movesc_3.5_0.7_DIF':		[-10000, -4.451, -1.391, -0.146, 0.0, 0.312, 10000],
            'book_imb_mean_dif_cp_ABS':		[-0.01, 0.119, 10000],
            'msg_imb_mean_dif_lp_ABS':		[-0.01, 0.281, 1.1],
            'loc_agg_imb_m':  [-1.01, -0.491, -0.366, -0.288, -0.223, -0.167, 1.01],
            'imp_FCAST_10': [-10000, -3.012, -1.674, -0.653, 10000],
}

l_bins_nulls = ['vol_big','n_big','PA_up','PA_down','vol_big_aux','n_aggimb_aux','imp_DIF','int_DIF',
                'int_DIF_50','abs_DIF_30','msg_imb_dif','depth_DIF',
                'depth_DIF_10','book_imb_mean_us_5','book_imb_mean_us_20','sspread_mean_us_5',
                'escora_2.5_2_DIF','movesc_4.5_0.7_DIF', 
                # TODO: 's_run', 'n_p_aux' treat for MW
]

# ft classification based on mean encoding 
l_alta = ['pagg_DIF','loc_agg_net_m_aux','aggpior_aux_DIF','loc_agg_imb_m','vol_big_ratio','aggimb',
             'n_aggimb','agg_net_m','loc_aggbig_imb_m','n_p','vewma','agg_net_m_aux','smart_price_dif',
             'smart_price_50','agg_net_10','msg_imb_mean_40_ABS','ohlc_10','ohlc_50','loc_aggbig_net_m',
             'vol_trd_aux','aggbig_net_m_aux','rng_ewma_dif_80','vewma_10','rng_ewma','vewma_g_p','vewmag_dif',
             'n_trd_aux','escora_bid_2.5_0.8','escora_ask_2.5_0.8','escora_bid_3.5_0.8','escora_ask_3.5_0.8'
             ,'escora_bid_4.5_1.2','escora_ask_4.5_1.2','movesc_bid_3.5_0.5','movesc_ask_3.5_0.5','msg_imb','rng_smart_10',
]

l_media = ['last_d_s','aggimb_big','loc_aggbig_imb_m_aux','pagg_aux_DIF','movesc_3.5_0.7_DIF','abs_DIF',
              'book_imb_mean_dif_cp_ABS','book_imb_dif','abagg','vewma_c_v','aggimb_aux','loc_aggbig_net_m_aux',
              'rng_smart_50','agg_net_40','loc_agg_net_10','int_DIF_10','abagg_10','msg_imb_mean_dif_cp',
              'sspread_mean','movesc_2.5_0.7_DIF','loc_agg_net_m','book_imb','rng_ewma_dif','rng_ewma_dif_40',
              'vewma_g_p_10','aggbig_net_10','vol_trd','loc_agg_imb_m_aux','movesc_bid_2.5_0.9','movesc_ask_2.5_0.9',
              'movesc_bid_4.5_0.9','movesc_ask_4.5_0.9','imp_DIF_10','imp_FCAST_40','aggpior_DIF','book_imb_mean_dif_cp',
              'msg_imb_mean_10',
]

# best features from RecursiveFeatureAdd/RandomForest, threshold= 0.002
l_best = ['ohlc_50', 'vewma', 'vewmag_dif', 'rng_smart_10', 'ohlc_10',
          'loc_agg_imb_m', 'aggpior_DIF', 'agg_net_10', 'smart_price_dif',
          'escora_bid_3.5_0.8', 'msg_imb_mean_40_ABS'
]

l_ambas = ['movesc_ask_2.5_0.9', 'smart_price_dif', 'ohlc_50', 'ohlc_10',
           'movesc_bid_2.5_0.9', 'loc_agg_imb_m', 'loc_agg_net_m_aux', 'rng_smart_10',
]

l_tm = ['chgfreq','msg_imb_mean_dif_cp','pagg_aux_DIF','msg_imb','imp_FCAST_10','imp_FCAST_40',]

l_linear = ['pagg_DIF','vol_trd','vewmag_dif','msg_imb_mean_40_ABS','n_p','rng_ewma_dif',]

l_tree=['agg_net_10', 'vewma', 'vewma_c_v', 'agg_net_40', 'imp_DIF_10', 'book_imb_mean_dif_cp',]

l_best_svm = ['smart_price_dif', 'ohlc_50', 'chgfreq','movesc_ask_2.5_0.9']

l_second_svm = ['vewmag_dif', 'msg_imb', 'vol_trd', 'n_p', 'ohlc_10',
                'msg_imb_mean_40_ABS', 'loc_agg_net_m_aux_me', 'loc_agg_imb_m_me',
                'pagg_DIF_me', 'pagg_aux_DIF_me']

#%% pipelines preprocessing
pipe_cap_outliers = Pipeline([
                ('cap1', Winsorizer(variables=l_cap_1, capping_method='quantiles', fold = 0.01, tail = 'both')),
                ('cap5', Winsorizer(variables=l_cap_5, capping_method='quantiles', fold = 0.05, tail = 'both')),
                ('cap10', Winsorizer(variables=l_cap_10, capping_method='quantiles', fold = 0.10, tail = 'both')),
                ('cap001', Winsorizer(variables=l_cap_default, capping_method='quantiles', fold = 0.001, tail = 'both')),
])

pipe_norm_scale = Pipeline([
                ('scaler', SklearnTransformerWrapper(transformer=RobustScaler(quantile_range=(0.10, 0.90)))),
                ('minmax', SklearnTransformerWrapper(transformer=MinMaxScaler())),
                # ('pca', PCA(n_components=30, svd_solver='auto')), 
                # ('isomap', Isomap(n_components=13, n_neighbors=50, n_jobs=-2)),   #  expensive
])

pipe_discrete = Pipeline([('drop', DropFeatures(l_bins_nulls)),
                        ('duplicate', ft_eng.Duplicate(l_duplicate)),
                        ('outliers_all', Winsorizer(capping_method='quantiles', fold = 0.001, tail = 'both')),
                        ('bins_manual', ArbitraryDiscretiser(d_bins_arbitrary)),
                        ('bins_q10', EqualFrequencyDiscretiser(return_object=False, q=10, variables=l_bins_q10)),
                        ('bins_q8', EqualFrequencyDiscretiser(return_object=False, q=8, variables=l_bins_q8)),
                        ('bins_q6', EqualFrequencyDiscretiser(return_object=False, q=6, variables=l_bins_q6)),
                        ('bins_q5', EqualFrequencyDiscretiser(return_object=False, q=5, variables=l_bins_q5)),
                        ('bins_q4', EqualFrequencyDiscretiser(return_object=False, q=4, variables=l_bins_q4)),
                        ('bins_q2', EqualFrequencyDiscretiser(return_object=False, q=2, variables=l_bins_q2)),
])

pipe_mean_encoding = Pipeline([('drop', DropFeatures(l_bins_nulls)),
                        ('duplicate', ft_eng.Duplicate(l_duplicate)),
                        ('outliers_all', Winsorizer(capping_method='quantiles', fold = 0.001, tail = 'both')),
                        ('bins_manual', ArbitraryDiscretiser(d_bins_arbitrary)),
                        ('bins_q10', EqualFrequencyDiscretiser(return_object=True, q=10, variables=l_bins_q10)),
                        ('bins_q8', EqualFrequencyDiscretiser(return_object=True, q=8, variables=l_bins_q8)),
                        ('bins_q6', EqualFrequencyDiscretiser(return_object=True, q=6, variables=l_bins_q6)),
                        ('bins_q5', EqualFrequencyDiscretiser(return_object=True, q=5, variables=l_bins_q5)),
                        ('bins_q4', EqualFrequencyDiscretiser(return_object=True, q=4, variables=l_bins_q4)),
                        ('bins_q2', EqualFrequencyDiscretiser(return_object=True, q=2, variables=l_bins_q2)),
                        ('mean_enc', MeanEncoder())
])

# X_train_transf = pipe_mean_encoding.fit_transform(X_train, y_train)
# X_test_pre = pipe_cap_outliers.transform(X_test)

# DEBUG: must return all zeroes 
# pd.isna(X_train_transf).sum().sort_values(ascending=False)


#%% fuctions definitions for optimization process
import time 
def epoch2str(epoch):
  # from Fredy
  mlsec = "000"
  if str(epoch).find(".") >= 0:
        mlsec = repr(epoch).split('.')[1][:3]
  return time.strftime(
      '[%Y-%m-%d %H:%M:%S.{}]'.format(mlsec), time.localtime(epoch))


#%% creating configuration DICTS for ENSEMBLE

pipe_s = Pipeline([('cap_outliers', pipe_cap_outliers),
                  ('norm_scale', pipe_norm_scale)
                  ])
pipe_s.fit(X_train, y_train) 
X_scale = pipe_s.transform(X_train)
X_test_scale = pipe_s.transform(X_test)

pipe_mean_encoding.fit(X_train, y_train) 
X_me = pipe_mean_encoding.transform(X_train)
X_test_me = pipe_mean_encoding.transform(X_test)

pipe_discrete.fit(X_train, y_train) 
X_d = pipe_discrete.transform(X_train)
X_test_d = pipe_discrete.transform(X_test)

X_ens_opt = X_scale.join(X_me, how='outer', rsuffix='_me', sort=False).join(X_d, how='outer', rsuffix='_d', sort=False)
X_ens_opt_test = X_test_scale.join(X_test_me, how='outer', rsuffix='_me', sort=False).join(X_test_d, how='outer', rsuffix='_d', sort=False)

# feature selection for ensemble models
d_run_models = {'xgb1': 'scale_discrete',    # based on this, specify _d or _me for discrete or mean encoding
                'svm1': 'scale_me',           # on d_ft_bins below
                'knn57': 'scale',
                'knn75': 'scale',
                'knn100': 'scale',
                'logit1': 'me',
                'logit2': 'me',
                'xgb2': 'scale_discrete',
                'xgb3': 'discrete',
                'xgb4': 'discrete',

}

d_ft_bins = {'xgb1': ['smart_price_dif_d','chgfreq_d','agg_net_10_d','book_imb_mean_dif_cp_ABS_d',],
             'svm1': ['ohlc_50_me', 'loc_agg_imb_m_me', 'chgfreq_me', ],
             'knn57': 'NA',
             'knn75': 'NA',
             'knn100': 'NA',
             'logit1': ['agg_net_m_me', 'n_p_me', 'chgfreq_me', 'vewma_c_v_me', 'aggimb_big_aux_me', 'smart_price_dif_me', 
                        'smart_price_50_me', 'agg_net_10_me', 'int_DIF_10_me', 'msg_imb_mean_40_ABS_me'],
             'logit2': ['agg_net_m_me', 'chgfreq_me', 'smart_price_dif_me', 'smart_price_50_me', 'agg_net_10_me'],
             'xgb2': ['smart_price_dif_d', 'smart_price_50_d', 'pagg_DIF_d', 'loc_agg_imb_m_d', 
                      'loc_agg_net_m_aux_d', 'int_DIF_10_d', 'vewmag_dif_d', 'n_aggimb_d',],
             'xgb3': ['smart_price_dif_d', 'chgfreq_d', 'agg_net_10_d', 'book_imb_mean_dif_cp_ABS_d', 'ohlc_50_d'],
             'xgb4': ['book_imb_mean_dif_cp_d', 'vewma_g_p_d', 'loc_agg_imb_m_d', 'agg_net_m_d', 'loc_agg_net_m_aux_d',
                      'escora_ask_3.5_0.8_d', 'n_trd_aux_d', 'movesc_bid_2.5_0.9_d']
}

d_ft_scale = {'xgb1': ['smart_price_dif','ohlc_10','chgfreq','agg_net_10',],
              'svm1': ['imp_FCAST_10', 'rng_smart_10', 'smart_price_dif', 'movesc_ask_2.5_0.9', 'msg_imb_mean_dif_cp', 'pagg_aux_DIF'],
              'knn57': ['chgfreq', 'smart_price_dif', 'rng_smart_10', ],  # 'ohlc_50', 'loc_agg_net_m_aux' 
              'knn75': ['book_imb_mean_dif_cp', 'imp_FCAST_10', 'imp_FCAST_40', 'pagg_aux_DIF', 'loc_agg_net_m_aux', 'n_trd_aux', 'msg_imb',],
              'knn100': ['n_trd',  'vol_big',  'aggimb_big',  'loc_agg_imb_m',  'chgfreq',  'PA_down',  
                        'vewma',  'movesc_ask_4.5_0.9',  'smart_price_dif',  'ohlc_10',  'ohlc_50',  
                        'vewma_g_p_10', 'int_DIF_50', 'book_imb_dif', 'book_imb_mean_dif_cp', 'msg_imb_mean_40_ABS'],
              'logit1': 'NA',
              'logit2': 'NA',
              'xgb2': ['rng_smart_10_d', 'chgfreq_d', 'agg_net_10_d', 'movesc_ask_2.5_0.9_d', 'n_trd_d',
                      'msg_imb_mean_dif_cp_d', 'msg_imb_mean_dif_lp_ABS_d',],
              'xgb3': 'NA',
              'xgb4': 'NA',

}


from sklearn.metrics import fbeta_score, make_scorer
f_beta_scorer = make_scorer(fbeta_score, beta=0.5)

#%% define models for ensemble
N_ENS = 11
b_optimize = True

print('-------------------------------------------')
print('ENSEMBLE: {}'.format(N_ENS))
print('Initial time: {}'.format(epoch2str(time.time())))

l_estimators = [
              ('svm1',Pipeline([('ft_sel', DropFeatures(list_sub(X_ens_opt.columns.to_list(),d_ft_scale['svm1']+d_ft_bins['svm1']))),
                                ('clf', SVC(probability=True, 
                                          class_weight='balanced', cache_size=1000, 
                                          C=24.84969151880429, 
                                          gamma= 0.0030157367155080337))])
              ), 
              ('xgb1', Pipeline([('ft_sel', DropFeatures(list_sub(X_ens_opt.columns.to_list(),d_ft_scale['xgb1']+d_ft_bins['xgb1']))),
                                ('clf', XGBClassifier( objective='binary:logistic'
                                                      ,eval_metric='auc'
                                                      ,use_label_encoder=False
                                                      ,tree_method='auto'
                                                      ,max_depth=5
                                                      ,subsample=0.7
                                                      ,colsample_bytree=0.99999
                                                      ,min_child_weight=25
                                                      ,learning_rate=0.0015
                                                      ,n_estimators=1100
                                                      ,gamma=3
                                                      ,reg_alpha=2
                                                      ,scale_pos_weight=2.5))])
              ), 
              # ('xgb2', Pipeline([('ft_sel', DropFeatures(list_sub(X_ens_opt.columns.to_list(),d_ft_scale['xgb2']+d_ft_bins['xgb2']))),
              #                   ('clf', XGBClassifier( objective='binary:logistic'
              #                                         ,eval_metric='auc'
              #                                         ,use_label_encoder=False
              #                                         ,tree_method='auto'
              #                                         ,max_depth=3
              #                                         ,subsample=0.5
              #                                         ,colsample_bytree=0.4
              #                                         ,min_child_weight=60
              #                                         ,learning_rate=0.0015
              #                                         ,n_estimators=1500
              #                                         ,gamma=0
              #                                         ,reg_alpha=0
              #                                         ,scale_pos_weight=4))])
              # ), 
              # ('xgb3', Pipeline([('ft_sel', DropFeatures(list_sub(X_ens_opt.columns.to_list(),d_ft_bins['xgb3']))),
              #                   ('clf', XGBClassifier( objective='binary:logistic'
              #                                         ,eval_metric='auc'
              #                                         ,use_label_encoder=False
              #                                         ,tree_method='auto'
              #                                         ,max_depth=3
              #                                         ,subsample=0.5
              #                                         ,colsample_bytree=0.4
              #                                         ,min_child_weight=60
              #                                         ,learning_rate=0.0015
              #                                         ,n_estimators=1500
              #                                         ,gamma=0
              #                                         ,reg_alpha=0
              #                                         ,scale_pos_weight=4))])
              # ), 
              # ('xgb4', Pipeline([('ft_sel', DropFeatures(list_sub(X_ens_opt.columns.to_list(),d_ft_bins['xgb4']))),
              #                   ('clf', XGBClassifier( objective='binary:logistic'
              #                                         ,eval_metric='auc'
              #                                         ,use_label_encoder=False
              #                                         ,tree_method='auto'
              #                                         ,max_depth=4
              #                                         ,subsample=0.5
              #                                         ,colsample_bytree=0.8
              #                                         ,min_child_weight=60
              #                                         ,learning_rate=0.0015
              #                                         ,n_estimators=1500
              #                                         ,gamma=2
              #                                         ,reg_alpha=3
              #                                         ,scale_pos_weight=4))])
              # ), 
              ('knn57', Pipeline([('ft_sel', DropFeatures(list_sub(X_ens_opt.columns.to_list(),d_ft_scale['knn57']))),
                                  ('clf', KNeighborsClassifier(n_neighbors=57))])
              ), 
              ('knn75', Pipeline([('ft_sel', DropFeatures(list_sub(X_ens_opt.columns.to_list(),d_ft_scale['knn75']))),
                                  ('clf', KNeighborsClassifier(n_neighbors=75))])
              ), 
              # ('knn100', Pipeline([('ft_sel', DropFeatures(list_sub(X_ens_opt.columns.to_list(),d_ft_scale['knn100']))),
              #                      ('clf', KNeighborsClassifier(n_neighbors=100))])
              # ), 
              ('logit1', Pipeline([('ft_sel', DropFeatures(list_sub(X_ens_opt.columns.to_list(),d_ft_bins['logit1']))),
                                   ('clf', LogisticRegression(max_iter=800))])
              ), 
              # ('logit2', Pipeline([('ft_sel', DropFeatures(list_sub(X_ens_opt.columns.to_list(),d_ft_bins['logit2']))),
              #                      ('clf', LogisticRegression(max_iter=800))])
              # ), 
]


# search params OR run fix models
if b_optimize:
  search_space = list()
  param_name = []

  ### final estimator
  param_name.append('logit__C') 
  search_space.append(Real(1e-6, 100.0, 'log-uniform', name='logit__C'))
  param_name.append('rf__n_estimators') 
  search_space.append(Integer(100, 1500, name='rf__n_estimators'))
  param_name.append('rf__max_depth') 
  search_space.append(Integer(5, 25, name='rf__max_depth'))
  param_name.append('rf__min_samples_leaf') 
  search_space.append(Integer(5, 50, name='rf__min_samples_leaf'))
  param_name.append('rf__max_features') 
  search_space.append(Categorical(['sqrt', 'log2', 0.5, 0.9], name='rf__max_features'))
  param_name.append('rf__max_samples') 
  search_space.append(Real(0.1, 0.9, name='rf__max_samples'))
  param_name.append('rf__class_weight') 
  search_space.append(Categorical(['balanced_subsample', 'balanced'], name='rf__class_weight'))


  # define the function used to evaluate a given configuration
  @use_named_args(search_space)
  def evaluate_model(**params):

    ens_opt = StackingClassifier(estimators=l_estimators, 
                                final_estimator=LogisticRegression(dual=False, tol=0.0001, fit_intercept=True, intercept_scaling=1, 
                                                                  random_state=None, max_iter=100, multi_class='auto',
                                                                  verbose=0, warm_start=False, n_jobs=-2, l1_ratio=None,
                                                                  penalty='l1', solver='liblinear', class_weight='balanced',
                                                                  C=params['logit__C']),
                                # final_estimator= RandomForestClassifier(n_jobs=-2, class_weight=params['rf__class_weight'],
                                                              # n_estimators = params['rf__n_estimators'], 
                                                              # max_depth = params['rf__max_depth'],
                                                              # min_samples_leaf = params['rf__min_samples_leaf'], 
                                                              # max_features = params['rf__max_features'],
                                                              # max_samples = params['rf__max_samples']),
                                # final_estimator=MajorityVote(),
                                # final_estimator=KNeighborsClassifier(),
                                passthrough=False, cv=3, verbose=1, n_jobs=-2
                                )

    cv_results = cross_validate(ens_opt, X_ens_opt, y_train, cv=4, n_jobs=-2, scoring=f_beta_scorer, return_train_score= True)
    
    overfit_devs = abs(cv_results['train_score'].mean() - cv_results['test_score'].mean())
    estimate = cv_results['test_score'].mean() - cv_results['test_score'].std() / 4 - overfit_devs / 3

    return 1.0 - estimate               


  result_ens_opt = forest_minimize(evaluate_model, search_space, n_calls=100, n_jobs=-2, verbose=1, 
                                        kappa=4.00, callback=DeltaYStopper(0.0002, n_best=8))
      
  # summarizing finding:
  print('\noptimized_ensemble():')
  print('Best Metric: %.3f' % (1.0 - result_ens_opt.fun))
  for i, val in enumerate(result_ens_opt.x):
    print('{}: {}'.format(param_name[i],val))

  from skopt.plots import plot_evaluations
  _ = plot_evaluations(result_ens_opt, bins=10)

  from skopt.plots import plot_objective
  _ = plot_objective(result_ens_opt)

  print('Time at end: {}'.format(epoch2str(time.time())))

else:
  ens_opt = StackingClassifier(estimators=l_estimators, 
                              # final_estimator=LogisticRegression(dual=False, tol=0.0001, fit_intercept=True, intercept_scaling=1, 
                              #                                   random_state=None, max_iter=100, multi_class='auto',
                              #                                   verbose=0, warm_start=False, n_jobs=-2, l1_ratio=None,
                              #                                   penalty='l1', solver='liblinear', class_weight='balanced',),
                              final_estimator=RandomForestClassifier(max_depth=3, n_jobs=-2, max_samples=0.6),
                              passthrough=False, cv=3, verbose=1, n_jobs=-2
                              )

# fit ensemble model and predict 
# ens_opt.set_params(final_estimator__C=0.018202370448969933)
ens_opt.fit(X_ens_opt, np.ravel(y_train))

print('\nEnsemble results for fixed parameters: ')
l_results = tm_train.report_results(X_ens_opt, X_ens_opt_test, y_train, y_test, fitted_model=ens_opt, ready_probs=False, th=0)

print('\nFinal time: {}'.format(epoch2str(time.time())))


# %%
