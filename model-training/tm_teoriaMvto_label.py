# -*- coding: utf-8 -*-
"""
Created on 25/Mar/21
@author: GuiThalita 
"""

# %% import features files 
import os 
import numpy as np
import pandas as pd
import pickle

pd.set_option('display.max_columns',100)
pd.set_option('precision', 3)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


ft_aux = [
  'vol_trd_aux',
  'n_trd_aux',
  'vol_big_aux',
  'n_big_aux',
  'vol_big_ratio_aux',
  'big_c_aux',
  'big_v_aux',
  'aggpior_c_aux',
  'aggpior_v_aux',
  'aggimb_aux',
  'aggimb_big_aux',
  'n_aggimb_aux',
  'agg_net_m_aux',
  'aggbig_net_m_aux',
  'loc_aggbig_c_m_aux',
  'loc_aggbig_v_m_aux',
  'loc_agg_net_m_aux',
  'loc_aggbig_net_m_aux',
  'loc_agg_imb_m_aux',
  'loc_aggbig_imb_m_aux',
  'pagg_c_best_aux',
  'pagg_c_best_0.5_aux',
  'pagg_c_best_0.7_aux',
  'pagg_c_best_0.9_aux',
  'pagg_v_best_aux',
  'pagg_v_best_0.5_aux',
  'pagg_v_best_0.7_aux',
  'pagg_v_best_0.9_aux',
  'abagg_aux',
  'n_p_aux'
  ]


class Labeling():

  def __init__(self, label_type, b_dist_to_high, s_path, prefix_files, df_data=None, 
              th_label_y1 = 0.6, b_percent_freq = True, b_custom_dir = False):
    '''
      Start label class reading files in "s_path/prefix_files*"
      ::s_path: if not specified, will search on default "explore\\historical_files\\FeaturesFiles\\"
      ::prefix_files: files to label. If each month: 'ft_2020-06' or all files: 'ft_20'
                    (only used if b_run_all = True)       
      ::label_type: 'c_binary' to classification and binary (0,1)
                  'c_multi'  to classification and multi (0,1,2) or (0,1,-1)
                  'regression' use param b_percent_freq = True to label y as percent of frquency
                  (only 'c_binary' working at the current version)
      ::th_label_y1: float, percent of daily frequency considered as relevant price movement
                    if b_percent_freq = True
                    or, int, the number of final size for runs to be considered in labeling
                    Default is used to flag_d = 1 (current = 0.6 * freq)
      ::b_dist_to_high: True label y=1 if the distance to current max to final max of Run > th * freq
                      False label y=1 if just final max of Run > th * freq      
      ::b_percent_freq: define th_label_y1 if flot or int. 
                      if label_type 'regression', specify True to label y as percent of frequency
                      Also changes b_custom_dir
      ::b_custom_dir: True to calculate d_run to high_run > th
                      if False d_run is based on flag_d (default values for x when running the strategy)
    '''     
    self.b_error = False
    self.s_path = s_path
    self.prefix_files = prefix_files

    if isinstance(df_data, pd.core.frame.DataFrame):
      self.df_train = df_data.reset_index()
    else:
      file_to_load = open(os.path.join(s_path,'y_source_'+ prefix_files + '.pkl'), "rb")
      self.df_train = pickle.load(file_to_load)
      self.df_train.reset_index(inplace=True)
      file_to_load.close()

    self.set_params(label_type, b_dist_to_high, th_label_y1, b_percent_freq, b_custom_dir)


  def apply_label(self, s_model_return):

    if self.label_type == 'c_binary': self.labeling_binary()
    elif self.label_type == 'regression': self.labeling_regression()
    else: 
      print('labeling - label_type {} not implemented... '.format(self.label_type))
      self.b_error = True

    self.b_error = False
    self.s_name = 'prefix_'+ self.prefix_files +'_type_'+ self.label_type +'_th_'+ str(self.th_label_y1) +'_dist_'+ str(self.b_dist_to_high) +'_percent_'+str(self.b_percent_freq)+'_custom_'+str(self.b_custom_dir)    
    y_model = self.y_model_export(s_model_return)

    return y_model


  def set_params(self, label_type, b_dist_to_high, th_label_y1, b_percent_freq, b_custom_dir):
    
    self.label_type = label_type 
    self.th_label_y1 = th_label_y1
    self.b_dist_to_high = b_dist_to_high
    self.b_percent_freq = b_percent_freq
    self.b_custom_dir = b_custom_dir

    if (not b_percent_freq) & (th_label_y1 < 1):
      print('labeling.set_params: params not setted properly.')
      print('labeling.set_params: b_percent_freq = False but th_label < 1. Please, pass threshold as RUN size in points.')
      self.b_error = True
    else:
      self._flag_dir()  # calculate run_high e d_run based on params b_percent_freq, th_label_y1 and b_custom_dir
      self.b_error = False
      
    return
      

  def _flag_dir(self):

    th = int(self.th_label_y1) if not self.b_percent_freq else (self.th_label_y1 * self.df_train['freq'])

    # variables needed for label
    self.df_train['run_high'] = self.df_train.groupby(['date_trd','id_run'])['max_run'].transform('last')

    if self.b_custom_dir:
      if self.b_dist_to_high:
        run_sz = th * 2
      else:
        run_sz = th
      self.df_train['d_run'] = self.df_train['run_high'].abs() >= run_sz
    else:
      self.df_train['d_run'] = self.df_train.groupby(['date_trd','id_run'])['flag_d'].transform('last')

    m_first_p = ((self.df_train['d_run'] == 1) & (self.df_train['id_run'] > self.df_train['id_run'].shift()) & (self.df_train['s_run'] != 0)
                & ((self.df_train['chg_max'] == self.df_train['max_run']) | (self.df_train['chg_max_mp'] == - self.df_train['max_run'])))
    self.df_train.loc[m_first_p, 'id_first_p'] = self.df_train['id_p']

    self.df_train.loc[(self.df_train['d_run'] == 1) & (self.df_train['id_run'] > self.df_train['id_run'].shift()) 
                & pd.isna(self.df_train['id_first_p']), 'id_first_p'] = self.df_train['id_p']-2

    df_first_p = self.df_train.loc[pd.notna(self.df_train['id_first_p']), ['date_trd', 'id_first_p', 'run_high']]
    df_first_p.set_index(['date_trd', 'id_first_p'], inplace= True)
    
    if df_first_p.index.duplicated().any():
      df_first_p = df_first_p.loc[~df_first_p.index.duplicated(keep='first')]

    # try to drop to prevent error in case of 
    # running multiple times
    try:
      self.df_train.drop(columns=['run_high_dir'], inplace=True)  
    except KeyError:
      pass

    # add _dir joining from first P of the run
    self.df_train = self.df_train.join(df_first_p, on=['date_trd','id_p'], rsuffix='_dir')

    # ignore first UPDATE
    self.df_train = self.df_train.loc[self.df_train['update'] != self.df_train.groupby('date_trd')['update'].transform('first')]

  
  # LABEL - binary problem
  def labeling_binary(self):

    # criar versão em que label baseia-se em pontos ao invés de percentual da frequencia 
    th = int(self.th_label_y1) if not self.b_percent_freq else (self.th_label_y1 * self.df_train['freq'])
    # print(self.df_train[['date_trd', 'time', 'freq', 'first_price', 'chg_cum' ,'chg_max_mp','chg_max',
    #                     'id_p', 'id_first_p', 'seq', 's_run', 'id_run', 'max_run', 'run_high', 'flag_d', 'd_run',
    #                     'run_high_dir', 'model', 'y']])   # .to_csv(self.s_path + 'self.df_train.csv')      

    self.df_train['y'] = np.nan

    if not self.b_dist_to_high:
      # alternative 1) run_high > threshold
      label_1_inicio_ideal = ( pd.notna(self.df_train['run_high_dir']) & pd.notna(self.df_train['model']) 
                              & (self.df_train['run_high_dir'].abs() >= th)
                              )
    else:
      # alternative 2) run_high - current_max > threshold
      label_1_inicio_ideal = ( pd.notna(self.df_train['run_high_dir']) & pd.notna(self.df_train['model']) 
                              & ((self.df_train['run_high_dir'] - self.df_train['chg_max']).abs() >= th)  
                              )
  
    self.df_train.loc[label_1_inicio_ideal, 'y'] = 1

    label_0_inicio_ideal = (pd.notna(self.df_train['run_high_dir']) & pd.notna(self.df_train['model']) 
                            & pd.isna(self.df_train['y'])
                            )
    self.df_train.loc[label_0_inicio_ideal, 'y'] = 0

    label_1_run_so_far_to_end = (pd.notna(self.df_train['model']) 
                  & pd.isna(self.df_train['y'])
                  & (self.df_train['chg_max'] * self.df_train['s_run'] > 0)
                  & (self.df_train['d_run'] == 1)
                  )
    
    if not self.b_dist_to_high:
      # alternative 1) run_high > threshold
      label_1_run_so_far_to_end = label_1_run_so_far_to_end & (self.df_train['run_high'].abs() >= th) 
    else:
      # alternative 2) run_high - current_max > threshold
      label_1_run_so_far_to_end = label_1_run_so_far_to_end &  (
                                  ( (self.df_train['chg_max'].abs() >= self.df_train['chg_max_mp'].abs()) 
                                    & ((self.df_train['run_high'] - self.df_train['max_run']).abs() >= th)
                                  )
                                  |
                                  ( (self.df_train['chg_max'].abs() < self.df_train['chg_max_mp'].abs()) 
                                    & ((self.df_train['run_high'] - (self.df_train['max_run'] + self.df_train['chg_max_mp'] + self.df_train['chg_max'])).abs() >= th) 
                                  )
                                )

    self.df_train.loc[label_1_run_so_far_to_end, 'y'] = 1

    self.df_train.loc[pd.notna(self.df_train['model']) 
                & pd.isna(self.df_train['y']), 'y'] = 0

  
  # LABEL - regression problem
  def labeling_regression(self):

    self.df_train['y'] = np.nan

    # treating ideal start of the run (formal end of previous movement)
    if not self.b_dist_to_high:
      # alternative 1) run_high > threshold
      if not self.b_percent_freq:
        # 1.1) y = price change 
        self.df_train.loc[pd.notna(self.df_train['run_high_dir']) & pd.notna(self.df_train['model']) , 'y'] = self.df_train['run_high_dir']
      else:
        # 1.2) y= price change / freq  
        self.df_train.loc[pd.notna(self.df_train['run_high_dir']) & pd.notna(self.df_train['model']) , 'y'] = self.df_train['run_high_dir'] / self.df_train['freq'] 
    else:
      # alternative 2) run_high - current_max > threshold
      if not self.b_percent_freq:
        # 1.1) y = price change 
        self.df_train.loc[pd.notna(self.df_train['run_high_dir']) & pd.notna(self.df_train['model']) , 'y'] = self.df_train['run_high_dir'] - self.df_train['chg_max']
      else:
        # 1.2) y= freq / price change  self.df_train['freq'] / self.df_train['run_high_dir']
        self.df_train.loc[pd.notna(self.df_train['run_high_dir']) & pd.notna(self.df_train['model']) , 'y'] = (self.df_train['run_high_dir'] - self.df_train['chg_max']) / self.df_train['freq'] 

    # DEBUG:
    df_debug = self.df_train.loc[pd.notna(self.df_train['model']) & pd.notna(self.df_train['run_high_dir']) & pd.isna(self.df_train['y'])]
    if not df_debug.empty:
      print('DEBUG: labeling_regression() L: 360')
      print(df_debug.iloc[0:5, 0:6])
    
    label_1_run_so_far_to_end = (pd.notna(self.df_train['model']) 
                  & pd.isna(self.df_train['y'])
                  & (self.df_train['chg_max'] * self.df_train['s_run'] > 0)
                  & (self.df_train['d_run'] == 1)
                  )
    
    if not self.b_dist_to_high:
      # alternative 1) run_high > threshold
      if not self.b_percent_freq:
        # 1.1) y = price change 
        self.df_train.loc[label_1_run_so_far_to_end, 'y'] = self.df_train['run_high']
      else:
        # 1.2) y= price change / freq
        self.df_train.loc[label_1_run_so_far_to_end, 'y'] = self.df_train['run_high'] / self.df_train['freq'] 
    else:
      # alternative 2) run_high - current_max > threshold
      if not self.b_percent_freq:
        self.df_train.loc[label_1_run_so_far_to_end & (self.df_train['chg_max'].abs() >= self.df_train['chg_max_mp'].abs()) , 'y'] = self.df_train['run_high'] - self.df_train['max_run']
        self.df_train.loc[label_1_run_so_far_to_end & (self.df_train['chg_max'].abs()  < self.df_train['chg_max_mp'].abs()), 'y'] = self.df_train['run_high'] - (self.df_train['max_run'] + self.df_train['chg_max_mp'] + self.df_train['chg_max'])
      else:
        self.df_train.loc[label_1_run_so_far_to_end & (self.df_train['chg_max'].abs() >= self.df_train['chg_max_mp'].abs()) , 'y'] = (self.df_train['run_high'] - self.df_train['max_run']) / self.df_train['freq'] 
        self.df_train.loc[label_1_run_so_far_to_end & (self.df_train['chg_max'].abs()  < self.df_train['chg_max_mp'].abs()), 'y'] = (self.df_train['run_high'] - (self.df_train['max_run'] + self.df_train['chg_max_mp'] + self.df_train['chg_max'])) / self.df_train['freq'] 
    
    self.df_train.loc[pd.notna(self.df_train['model']) & pd.isna(self.df_train['y']), 'y'] = 0

    # self.df_train[['date_trd', 'time', 'freq', 'first_price', 'chg_cum' ,'chg_max_mp','chg_max',
    #   'id_p', 'id_first_p', 'seq', 's_run', 'id_run', 'max_run', 'run_high', 'flag_d', 'd_run',
    #   'run_high_dir', 'model', 'y']].to_csv(self.s_path + 'self.df_train.csv')                          
  

  def y_model_export(self,s_model_return = 'NONE'):

    #  export only y and index
    base_main = self.df_train.set_index(['date_trd','update','time','trd_id'])
    base_main.rename(columns={'model': 'model'}, inplace=True)
    df_export = base_main.loc[pd.notna(base_main['model']), ['model', 'y']]
    
    outfile = open(os.path.join(self.s_path, 'y_' + self.s_name +'.pkl'),'wb')
    pickle.dump(df_export, outfile)
    outfile.close()

    print('labeling - y config exported to {}!'.format(os.path.join(self.s_path, 'y_'+ self.s_name + '.pkl')))
    print(base_main['model'].value_counts())
    
    if s_model_return != 'NONE':
      return df_export.loc[base_main['model'] == s_model_return]
    else:
      return df_export.loc[pd.notna(base_main['model'])]

  
# %% 
# label_obj = labeling(prefix_files= 'ft_2020-06')  # start DataFrame 

# 1) classification
# label_obj.reset_params(label_type= 'c_binary', b_dist_to_high= True, th_label_y1=4, b_percent_freq=False) 
# label_obj.apply_label()
# label_obj.reset_params(label_type= 'c_binary', b_dist_to_high= False, th_label_y1=0.6) # Default for classification
# label_obj.apply_label()
# label_obj.reset_params(label_type= 'c_binary', b_dist_to_high= False, th_label_y1=0.8)
# label_obj.apply_label()
# label_obj.reset_params(label_type= 'c_binary', b_dist_to_high= True, th_label_y1=0.6) 
# label_obj.apply_label()
# label_obj.reset_params(label_type= 'c_binary', b_dist_to_high= True, th_label_y1=0.4) 
# label_obj.apply_label()

# 2) regression
# label_obj.reset_params(label_type= 'regression', b_dist_to_high= False, th_label_y1=0.4, b_percent_freq= True, b_custom_dir= True)  
# label_obj.apply_label()
# label_obj.reset_params(label_type= 'regression', b_dist_to_high= False, b_percent_freq= False, b_custom_dir= False)  # Default for regression
# label_obj.apply_label()
# label_obj.reset_params(label_type= 'regression', b_dist_to_high= True, b_percent_freq= False) 
# label_obj.apply_label()
# label_obj.reset_params(label_type= 'regression', b_dist_to_high= True, b_percent_freq= True) 
# label_obj.apply_label()


