# -*- coding: utf-8 -*-
"""
Created on 06/May/21
@author: GuiThalita 
"""

import os
import numpy as np
import pandas as pd
import pickle

def import_sampling(path_files, prefix_files):

  raw_data = read_files(path_files, prefix_files)
  raw_data = sampling(raw_data)

  return raw_data


def read_files(path_files, prefix_files):
  df_train = pd.DataFrame()
  # path_files = os.path.join(s_path, 'historical_files', 'FeaturesFiles')

  l_dir = os.listdir(path_files)
  d_type = prefix_files  # process each month 'ft_2020-06' or all files 'ft_20'
  l_file = [_file for _file in l_dir if d_type in _file]
  for _file in l_file:
    df_aux = pd.read_csv(os.path.join(path_files, _file))
    df_aux = df_aux.reset_index().rename(columns={'index': 'trd_id'})
    df_train = df_train.append(df_aux,
                              ignore_index=True,
                              sort=False)
  
  print('base_prep - read_files: imported from {}'.format(path_files))
  # DEBUG: df_train.loc[('2020-06-01',36914)].head(50)
  # df_train.groupby('date_trd')['trd_id'].max()
  
  print(df_train['date_trd'].value_counts())
  print('total: {}'.format(df_train['date_trd'].value_counts().sum()))

  return df_train


def sampling(df_train):
  '''creating column 'model' following theory from TeoriaMvto.''' 

  # df_mi 
  df_train.loc[:,'model'] = np.nan
  mi_down = ((df_train['flag_d'] == 1) & (df_train['chg_max'] * df_train['s_run'] < 0) 
      & (df_train['id_p'] > df_train['id_p'].shift()) & (df_train['chg_max'] < 0))
  df_train.loc[mi_down, 'model'] = 'mi_down'

  mi_up = ((df_train['flag_d'] == 1) & (df_train['chg_max'] * df_train['s_run'] < 0) 
      & (df_train['id_p'] > df_train['id_p'].shift()) & (df_train['chg_max'] > 0))
  df_train.loc[mi_up, 'model'] = 'mi_up'

  # df_md
  md_down = ((df_train['flag_d'] == 1) & (df_train['chg_max'] * df_train['s_run'] > 0) 
      & (df_train['id_p'] > df_train['id_p'].shift()) & (df_train['chg_max'] < 0))
  df_train.loc[md_down, 'model'] = 'md_down'

  md_up = ((df_train['flag_d'] == 1) & (df_train['chg_max'] * df_train['s_run'] > 0) 
      & (df_train['id_p'] > df_train['id_p'].shift()) & (df_train['chg_max'] > 0))
  df_train.loc[md_up, 'model'] = 'md_up'

  # df_mw
  mw_down= pd.isna(df_train['model']) & (df_train['id_p'] > df_train['id_p'].shift()) & (df_train['chg_max'] < 0) 
  df_train.loc[mw_down, 'model'] = 'mw_down'

  mw_up= pd.isna(df_train['model']) & (df_train['id_p'] > df_train['id_p'].shift()) & (df_train['chg_max'] > 0) 
  df_train.loc[mw_up, 'model'] = 'mw_up'

  # PARA VALIDAR 
  dfg = df_train.loc[(df_train['id_p'] > df_train['id_p'].shift()),['date_trd', 'time', 'first_price', 'chg_cum' ,'chg_max','chg_max_mp',
    'id_p', 'seq', 's_run', 'id_run', 'max_run', 'flag_d', 'model']]

  total = dfg['model'].count()
  n_mi_u = (dfg['model'] == 'mi_up').sum()
  n_mi_d = (dfg['model'] == 'mi_down').sum()
  n_mw_u = (dfg['model'] == 'mw_up').sum()
  n_mw_d = (dfg['model'] == 'mw_down').sum()
  n_md_u = (dfg['model'] == 'md_up').sum() 
  n_md_d = (dfg['model'] == 'md_down').sum()
  print('mi_up: {}  mi_down: {} \nmd_up: {}  md_down: {} \nmw_up: {}  mw_down: {}'.format(n_mi_u, n_mi_d, n_md_u, n_md_d, n_mw_u, n_mw_d))

  print('base_prep - sampling: complete!')
  return df_train


def ft_export(df_train, path_files, prefix):

  cols = dict()
  # ignore first UPDATE - also done in y during labeling 
  base_main = df_train.loc[df_train['update'] != df_train.groupby('date_trd')['update'].transform('first')]
  base_main.set_index(['date_trd','update','time','trd_id'], inplace=True)
  base_main = base_main.rename(columns={'model': 'f_model'})
  
  # remove model = NaN & selecting only the real features
  base_main = base_main.loc[pd.notna(base_main['f_model'])].filter(regex='f_')
  
  # rename removing 'f_'
  for s in base_main.columns: cols[s] = s[2:]
  base_main.rename(columns=cols, inplace=True)
    
  outfile = open(os.path.join(path_files, 'X_samples_' + prefix + '.pkl'),'wb')
  pickle.dump(base_main, outfile)
  outfile.close()
  print('samples from first update ignored.')
  print(df_train['model'].value_counts())
  
  print('base_prep - X exported succesfully to {}'.format(os.path.join(path_files, 'X_samples_' + prefix + '.pkl')))

  return base_main


def y_export(df_train, path_files, prefix):

  df_train.set_index(['date_trd','update','time','trd_id'], inplace=True)
    
  col_export_label = [ # 'date_trd', 'update', 'time',       #  - IDX
                      'side', 'first_price', 'id_run', 's_run', 
                      'max_run', 'flag_d', 'id_p', 'seq', 
                      'chg_max', 'chg_max_mp', 'freq', 'model']

  outfile = open(os.path.join(path_files, 'y_source_' + prefix + '.pkl'),'wb')
  pickle.dump(df_train[col_export_label], outfile)
  outfile.close()

  print('base_prep - y prep exported to {}!'.format(os.path.join(path_files, 'y_source_' + prefix + '.pkl')))

  return df_train[col_export_label]


