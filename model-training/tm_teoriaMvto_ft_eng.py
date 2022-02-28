# -*- coding: utf-8 -*-
"""
Created on 25/Mar/21
@author: GuiThalita 
"""

# %% import features files 
import os
import numpy as np
import pandas as pd

EXPORT_ORIGINAL = True
FEATURE_IMPORTANCE_ORIGINALS = False ## INCLUIR MODELO

pd.set_option('display.max_columns',100)
pd.set_option('precision', 3)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

DEBUG = True
d_path = { 'train': '..\\neutrino-lab\\historical_files\\TrainFiles\\',
           'test': '..\\neutrino-lab\\historical_files\\TestFiles\\'}

s_path = d_path['train']


#%% classes to attend pipeline requirements

from sklearn.base import BaseEstimator, TransformerMixin

class BasicFeatures(BaseEstimator, TransformerMixin):
  ''' to be used with training data base'''
  def __init__(self):
    pass

  def fit(self, X, y=None):
    return self 

  def transform(self, X):
    return self.basic_features(X)
  
  def basic_features(self, base):

    ####### FEATURES DE PRECO
    base['f_smart_price_dif'] = (base['f_smart_price'] - base.groupby('date_trd')['f_smart_price'].shift(10))
    base['f_smart_price_50'] = (base['f_smart_price'] - base.groupby('date_trd')['f_smart_price'].shift(50))
    base['f_rng_smart_10'] = (base.groupby('date_trd')['f_smart_price'].rolling(11).max()
                                - base.groupby('date_trd')['f_smart_price'].rolling(11).min()).reset_index(drop=True)
    base['f_rng_smart_50'] = (base.groupby('date_trd')['f_smart_price'].rolling(51).max()
                                - base.groupby('date_trd')['f_smart_price'].rolling(51).min()).reset_index(drop=True)
    base['f_ohlc_10'] = (base['f_smart_price_dif'] / base['f_rng_smart_10']).replace([np.inf, -np.inf], 0.0)
    base['f_ohlc_50'] = (base['f_smart_price_50'] / base['f_rng_smart_50']).replace([np.inf, -np.inf], 0.0)
    base['f_rng_ewma_dif'] = (base['f_rng_ewma'] - base.groupby('date_trd')['f_rng_ewma'].shift(10))
    base['f_rng_ewma_dif_40'] = (base['f_rng_ewma'] - base.groupby('date_trd')['f_rng_ewma'].shift(40))
    base['f_rng_ewma_dif_80'] = (base['f_rng_ewma'] - base.groupby('date_trd')['f_rng_ewma'].shift(80))
    # ultimas acrescentadas
    base['f_vewma_10'] = (base['f_vewma'] - base.groupby('date_trd')['f_vewma'].shift(10))
    base['f_vewma_g_p_10'] = (base['f_vewma_g_p'] - base.groupby('date_trd')['f_vewma_g_p'].shift(10))
    base['f_imp_DIF'] = base['f_imp_c_0.8'] - base['f_imp_v_0.8']
    base['f_imp_DIF_10'] = (base['f_imp_DIF'] - base.groupby('date_trd')['f_imp_DIF'].shift(10))
    base['f_imp_DIF_50'] = (base['f_imp_DIF'] - base.groupby('date_trd')['f_imp_DIF'].shift(50))
    
    # debug: first update_seq is nan because smart_price is the same in all trades. 
    # base.loc[(base['date_trd']=='2020-06-01')&(base['update']>36835), ['date_trd', 'update', 'time', 'trd_id',
    #             'smart_price','smart_price_dif','smart_price_50','rng_smart_10','rng_smart_50','ohlc_10',
    #             'ohlc_50','rng_ewma_dif','rng_ewma_dif_40','rng_ewma_dif_80']].head(40)
    
    ####### FEATURES DE VOLUME
    base['f_agg_net_10'] = (base['f_agg_net_d'] - base.groupby('date_trd')['f_agg_net_d'].shift(10))
    base['f_agg_net_40'] = (base['f_agg_net_d'] - base.groupby('date_trd')['f_agg_net_d'].shift(40))
    base['f_agg_net_80'] = (base['f_agg_net_d'] - base.groupby('date_trd')['f_agg_net_d'].shift(80))
    # ultimas acrescentadas
    base['f_loc_agg_net_10'] = (base['f_loc_agg_net_d'] - base.groupby('date_trd')['f_loc_agg_net_d'].shift(10))
    base['f_aggbig_net_10'] = (base['f_aggbig_net_d'] - base.groupby('date_trd')['f_aggbig_net_d'].shift(10))
    base['f_int_DIF'] = base['f_int_c_0.8'] - base['f_int_v_0.8']
    base['f_int_DIF_10'] = (base['f_int_DIF'] - base.groupby('date_trd')['f_int_DIF'].shift(10))
    base['f_int_DIF_50'] = (base['f_int_DIF'] - base.groupby('date_trd')['f_int_DIF'].shift(50))
    base['f_imp_FCAST_10'] = base['f_imp_DIF'] * base['f_agg_net_10']
    base['f_imp_FCAST_40'] = base['f_imp_DIF'] * base['f_agg_net_40']
    base['f_aggpior_DIF'] = base['f_aggpior_c'] - base['f_aggpior_v']
    base['f_aggpior_DIF_30'] = (base['f_aggpior_DIF'] - base.groupby('date_trd')['f_aggpior_DIF'].shift(30))
    base['f_abs_DIF'] = base['f_abs_c'] - base['f_abs_v']
    base['f_abs_DIF_30'] = (base['f_abs_DIF'] - base.groupby('date_trd')['f_abs_DIF'].shift(30))
    base['f_pagg_DIF'] = base['f_pagg_c_best_0.7'] - base['f_pagg_v_best_0.7']
    base['f_abagg_10'] = (base['f_abagg'] - base.groupby('date_trd')['f_abagg'].shift(10))

    # debug:
    # base.iloc[40:90][['date_trd', 'update', 'time', 'trd_id','agg_net_d','agg_net_10','agg_net_40','agg_net_80']]
    
    ####### FEATURES DE BOOK
    base['f_book_imb_dif'] = (base['f_book_imb'] - base.groupby('date_trd')['f_book_imb'].shift(10))
    base['f_book_imb_mean_10'] = (base.groupby(['date_trd'])['f_book_imb'].rolling(11).mean().reset_index(drop=True))
    base['f_book_imb_mean_40'] = (base.groupby(['date_trd'])['f_book_imb'].rolling(41).mean().reset_index(drop=True))
    base['f_book_imb_mean_dif_lp'] = base['f_book_imb_mean_10'] - base['f_book_imb_mean_40']
    base['f_book_imb_mean_dif_cp'] = base['f_book_imb'] - base['f_book_imb_mean_10']
    # debug: base[['date_trd', 'update', 'time', 'trd_id','book_imb','book_imb_dif','book_imb_mean_10','book_imb_mean_40','book_imb_mean_dif_lp', 'book_imb_mean_dif_cp']].tail(40)
    base['f_msg_imb_dif'] = base['f_msg_imb'] - base.groupby('date_trd')['f_msg_imb'].shift(10)
    base['f_msg_imb_mean_10'] = (base.groupby(['date_trd'])['f_msg_imb'].rolling(11).mean().reset_index(drop=True))
    base['f_msg_imb_mean_40'] = (base.groupby(['date_trd'])['f_msg_imb'].rolling(41).mean().reset_index(drop=True))
    base['f_msg_imb_mean_dif_lp'] = base['f_msg_imb_mean_10'] - base['f_msg_imb_mean_40']
    base['f_msg_imb_mean_dif_cp'] = base['f_msg_imb'] - base['f_msg_imb_mean_10']
    # debug: base[['date_trd', 'update', 'time', 'trd_id','msg_imb','msg_imb_dif','msg_imb_mean_10','msg_imb_mean_40', 'msg_imb_mean_dif_lp', 'msg_imb_mean_dif_cp']].tail(40)
    base['f_sspread_mean'] = (base.groupby(['date_trd'])['f_sspread'].rolling(11).mean().reset_index(drop=True))
    # debug: base[['date_trd', 'update', 'time', 'trd_id','sspread','sspread_mean']].head(40)
    # ultimas acrescentadas
    base['f_depth_DIF'] = base['f_depth_bid7'] - base['f_depth_ask7']
    base['f_depth_DIF_10'] = (base['f_depth_DIF'] - base.groupby('date_trd')['f_depth_DIF'].shift(10))

    # features below need join to work properly
    df_ft_join = base.groupby(['date_trd','update'])[['f_book_imb']].last().rolling(6).mean()
    df_ft_join.rename(columns={'f_book_imb': 'f_book_imb_mean_us_5'}, inplace=True)
    df_ft_join['f_book_imb_mean_us_20'] = base.groupby(['date_trd','update'])['f_book_imb'].last().rolling(21).mean()
    df_ft_join['f_sspread_mean_us_5'] = base.groupby(['date_trd','update'])['f_sspread'].last().rolling(6).mean()
    # debug: base[['date_trd', 'update', 'time', 'trd_id','book_imb','book_imb_mean_us_5','book_imb_mean_us_20','sspread','sspread_mean_us_5']].tail(40)

    base = base.join(df_ft_join, on=['date_trd', 'update'])

    return base 


# fteng = NewFeatures()
# X_newft = fteng.transform(X=X)

#%% LOG FEATURES TRADE_VOLUME 

class LogVolume(BaseEstimator, TransformerMixin):
  def __init__(self, l_columns):
    self.cols = l_columns

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    return self.log_volume_ft(X.copy())

  def log_volume_ft(self, base):
    # using copy() to prevent SettingWithCopyWarning
    for col in self.cols:
      # if it keeps SettingWithCopyWarning, try to run only the one of the below queries
      base.loc[base[col] >= 0, col] = np.log(1+base.loc[base[col] >= 0, col])
      base.loc[base[col] < 0, col] = -np.log(1-base.loc[base[col] < 0, col])

    return base


#%%
class RobustNorm(BaseEstimator, TransformerMixin):
  def __init__(self, clip=0.001, l_columns=[]):
    self.clip = clip
    self.m_perc = pd.DataFrame()
    self.std = 0
    self.mean = 0
    self.cols = l_columns

  def fit(self, X, y=None):
    if self.cols == []: self.cols = X.columns.to_list()
    self.m_perc = X[self.cols].describe(percentiles=[self.clip,(1.0-self.clip)]).iloc[[4,6]]
    self.std = X[self.cols].std()
    self.mean = X[self.cols].mean()
    return self

  def transform(self, X):
    X[self.cols] = X[self.cols].clip(self.m_perc.iloc[0], self.m_perc.iloc[1], axis=1)
    X[self.cols] = (X[self.cols]-self.mean)/self.std
    return X


#%% Escora - imputer NANs
class PropImputer(BaseEstimator, TransformerMixin):
  def __init__(self, exp, l_columns):
    self.exp = exp
    self.cols = l_columns
    self.ft_names = []

  def fit(self, X, y=None):
    self.ft_names = X.columns.to_list()

  def transform(self, X):
    return self.escora_imputer(X.copy(), self.exp)

  def escora_imputer(self, base, exp):
    # using copy() to prevent SettingWithCopyWarning
    base[self.cols] = base[self.cols].fillna(1/(14**exp))
    return base

  def get_feature_names(self):
    return self.ft_names


#%% dif features
class DifAll(BaseEstimator, TransformerMixin): # USAR ANTES DE LOG
  def __init__(self):
    pass

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    return self.diff_cols(X)

  def diff_cols(self, base):
    cols = base.columns.tolist()
    
    if 'aggpior_c_aux' in cols:
      base['aggpior_aux_DIF'] = base['aggpior_c_aux'] - base['aggpior_v_aux']
      base['pagg_aux_DIF'] = base['pagg_c_best_0.7_aux'] - base['pagg_v_best_0.7_aux']
    
    base['escora_2.5_2_DIF'] = base['escora_bid_2.5_2'] - base['escora_ask_2.5_2']
    base['escora_3.5_1.2_DIF'] = base['escora_bid_3.5_1.2'] - base['escora_ask_3.5_1.2']
    base['escora_4.5_0.8_DIF'] = base['escora_bid_4.5_0.8'] - base['escora_ask_4.5_0.8']
    base['movesc_2.5_0.7_DIF'] = base['movesc_bid_2.5_0.7'] - base['movesc_ask_2.5_0.7']
    base['movesc_3.5_0.7_DIF'] = base['movesc_bid_3.5_0.7'] - base['movesc_ask_3.5_0.7']
    base['movesc_4.5_0.7_DIF'] = base['movesc_bid_4.5_0.7'] - base['movesc_ask_4.5_0.7']

    # add during mean encoding
    base['book_imb_mean_dif_cp_ABS'] = base['book_imb_mean_dif_cp'].abs()
    base['msg_imb_mean_40_ABS'] = base['msg_imb_mean_40'].abs()
    base['msg_imb_mean_dif_lp_ABS'] = base['msg_imb_mean_dif_lp'].abs()


    return base


#%% DUP FEATURES for different preprocessing 

class Duplicate(BaseEstimator, TransformerMixin):
  def __init__(self, l_columns):
    self.cols = l_columns

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    
    for s_col in self.cols:
      X[s_col+'_2'] = X[s_col]

    return X

#%% SELECT FT

class SelectFt(BaseEstimator, TransformerMixin):
  def __init__(self, l_columns):
    self.cols = l_columns

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    return X[self.cols]

