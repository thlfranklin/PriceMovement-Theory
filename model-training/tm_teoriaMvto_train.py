# -*- coding: utf-8 -*-
"""
Created on 06/May/21
@author: GuiThalita 

"""
# %%

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt 
import seaborn as sns
import time 

from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier, Pool, sum_models
from sklearn.svm import SVC

import keras
from keras import metrics
from keras import regularizers 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_validate
from skopt.space import Integer
from skopt.space import Real
from skopt.space import Categorical
from skopt.utils import use_named_args
from skopt import forest_minimize
from skopt.plots import plot_objective
from skopt.callbacks import DeltaYStopper

pd.set_option('display.max_columns',100)
pd.set_option('precision', 3)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

d_path = { 'train': 'historical_files\\FeaturesFiles\\',
           'test': 'historical_files\\TestFiles\\'}

#%%
class ModelTraining():

  def __init__(self, s_regime='mi_up', run_model='svm', s_path = d_path['train']):
    self.s_path = s_path
    self.s_regime = ''
    self.run_model = run_model
    self.model = None
    self.x_train = pd.DataFrame()
    self.y_train = pd.DataFrame()
    self.x_test  = pd.DataFrame()
    self.y_test  = pd.DataFrame()
    self.search_space = []
    self.param_name= []
    self.x_h_train = pd.DataFrame()
    self.x_h_test = pd.DataFrame()
    self.result_opt = None

    # self.training_init()

    return

  
  def training_init(self):
    '''
      specify:
        ::s_model in ['svm', 'xgb', 'rf']
        ::s_regime in ['mi_up', 'mi_down', 'mw_up', 'mw_down']
    '''
    
    pkl_data = load_models(self.s_regime, self.s_path)  # will search for s_path/s_regime.pkl
    # pkl_data.fillna(0, inplace=True)  # NaN is treated on base_norm()

    X = pkl_data
    y = X.pop('y')

    X = base_norm(X)  # include b_norm = True / False
    self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)

    print('training_init: data load and split complete!')

    return 


  def xgboost(self, l_cols):
    l_cols = [
    #  'chgfreq',
    'movesc_bid_2.5_0.5',
    'smart_price_dif',
    'ohlc_50',
    'agg_net_10',
    'ohlc_10',
    'movesc_ask_2.5_0.7',
    'smart_price_50',
    'loc_agg_net_m_aux',
    'depth_bid7',
    'movesc_ask_3.5_0.5',
    'loc_agg_imb_m',
    'vewma',
    'n_trd_aux',
    'rng_smart_10',
    'big_c_aux',
    'rng_ewma',
    'big_v_aux',
    'movesc_bid_4.5_0.7',
    'int_v_0.8',
    'loc_aggbig_v_m',
    'imp_c_0.8',
    'last_d_s',
    'depth_ask7',
    'rng_ewma_dif_80',
    'escora_bid_4.5_2',
    'msg_imb_mean_10',
    'movesc_ask_3.5_0.7',
    'imp_v_0.7',
    'escora_ask_2.5_1.2',
    'int_c_0.8',
    'aggpior_v',
    ]

    X_train = self.x_train[l_cols]
    # [['chgfreq','movesc_bid_2.5_0.5','smart_price_dif','ohlc_50','movesc_bid_2.5','agg_net_10','ohlc_10','movesc_ask_2.5_0.7','smart_price_50']]

    X_test = self.x_test[l_cols]
    # [['chgfreq','movesc_bid_2.5_0.5','smart_price_dif','ohlc_50','movesc_bid_2.5','agg_net_10','ohlc_10','movesc_ask_2.5_0.7','smart_price_50']]
    y_train = self.y_train
    y_test = self.y_test

    fmodel = xgb.XGBClassifier(n_estimators=1500, 
                              objective='binary:logistic',
                              use_label_encoder=False, 
                              colsample_bytree=0.8, 
                              subsample=.3, 
                              learning_rate=0.0005,
                              max_depth=3, 
                              gamma=0.5,
                              # reg_alpha=0.45,
                              n_jobs=-1,
                              tree_method='hist',  # sempre hist
                              min_child_weight=100, 
                              # scale_pos_weight=1.4,  # mi-up label False_0.5_True
                              verbosity=1,
                              # early_stopping_rounds=10,
                              eval_metric = 'map'
                              )

    fmodel.fit(X_train, y_train)

    (train_auc, test_auc, train_f1, test_f1, train_f1_balanced, test_f1_balanced, 
    cm_train, cm_test, yhat_train, yhat_test, th) = report_results(X_train, X_test, y_train, y_test, fitted_model=fmodel)
    
    f_imp = ft_importance_plot(fmodel, X_train, plot_result=True)

    return fmodel

    
  def create_search_space(self):

    if self.run_model =='svm':
    ### SVM1
      self.param_name.append('svc__C') 
      self.search_space.append(Real(1e-6, 100.0, 'log-uniform', name='svc__C'))
      self.param_name.append('svc__gamma') 
      self.search_space.append(Real(1e-4, 10.0, 'log-uniform', name='svc__gamma'))

    if self.run_model =='rf':
    ### RF1
    # atencao; calibrar rf n_estimators maximo antes na mao devido ao tempo
      # X_train = X_m_train
      # X_test = X_m_test
      
      self.param_name.append('rf__n_estimators') 
      self.search_space.append(Integer(300, 600, name='rf__n_estimators'))
      self.param_name.append('rf__max_depth') 
      self.search_space.append(Integer(5, 25, name='rf__max_depth'))
      self.param_name.append('rf__min_samples_leaf') 
      self.search_space.append(Integer(5, 50, name='rf__min_samples_leaf'))
      self.param_name.append('rf__max_features') 
      self.search_space.append(Categorical(['sqrt', 'log2', 0.5, 0.9], name='rf__max_features'))
      self.param_name.append('rf__max_samples') 
      self.search_space.append(Real(0.1, 0.9, name='rf__max_samples'))
      self.param_name.append('rf__class_weight') 
      self.search_space.append(Categorical(['balanced_subsample', 'balanced'], name='rf__class_weight'))

    ### XGB1
    if self.run_model =='xgb':
      # X_train = X_train_raw
      # X_test = X_test_raw

      self.param_name.append('xgb__max_depth') 
      self.search_space.append(Integer(3, 15, name='xgb__max_depth'))
      self.param_name.append('xgb__subsample') 
      self.search_space.append(Real(.3, 1, name='xgb__subsample'))
      self.param_name.append('xgb__colsample_bytree') 
      self.search_space.append(Real(.1, .8, name='xgb__colsample_bytree'))
      self.param_name.append('xgb__min_child_weight') 
      self.search_space.append(Integer(1, 30, name='xgb__min_child_weight'))
      self.param_name.append('xgb__learning_rate') 
      self.search_space.append(Real(1e-4, 0.5, 'log-uniform', name='xgb__learning_rate'))
      self.param_name.append('xgb__n_estimators') 
      self.search_space.append(Integer(200, 2000, name='xgb__n_estimators'))
      self.param_name.append('xgb__gamma') ## related to trees pruning... cannot be too high
      self.search_space.append(Integer(0, 5, name='xgb__gamma'))
      self.param_name.append('xgb__reg_alpha') 
      self.search_space.append(Real(0., 10., name='xgb__reg_alpha'))
      self.param_name.append('xgb__scale_pos_weight') 
      self.search_space.append(Real(5., 15., name='xgb__scale_pos_weight'))

    # self.search_space.append(Categorical(['linear', 'poly', 'rbf', 'sigmoid'], name='kernel'))
    # self.search_space.append(Integer(1, 5, name='degree'))

    if self.run_model == 'logistic':
      self.param_name.append('logit__C') 
      self.search_space.append(Real(1e-6, 100.0, 'log-uniform', name='logit__C'))
        

    return 
  

  def evaluate_model(self,**params):
    # configure the model with specific hyperparameters
    # model = SVC()
    # model.set_params(C=params['C'], gamma=params['gamma'])
    # ('rf1', RandomForestClassifier(n_jobs=-1, class_weight='balanced_subsample',
    #           n_estimators = params['rf__n_estimators'], max_depth = params['rf__max_depth'],
    #           min_samples_leaf = params['rf__min_samples_leaf'], max_features = params['rf__max_features'],
    #           criterion = params['rf__criterion'], max_samples = params['rf__max_samples'])),

    if self.run_model =='xgb':
      model = xgb.XGBClassifier(objective='binary:logistic',
                      use_label_encoder=False, 
                      eval_metric = 'aucpr',
                      n_jobs=-1,
                      tree_method='hist',
                      n_estimators=params['xgb__n_estimators'],
                      colsample_bytree=params['xgb__colsample_bytree'],
                      subsample=params['xgb__subsample'],
                      learning_rate=params['xgb__learning_rate'],
                      max_depth=params['xgb__max_depth'],
                      gamma=params['xgb__gamma'],
                      reg_alpha=params['xgb__reg_alpha'],
                      min_child_weight=params['xgb__min_child_weight'],
                      scale_pos_weight=params['xgb__scale_pos_weight'],
                      verbosity=0)
    
    elif self.run_model =='rf':
      model = RandomForestClassifier(n_jobs=-1, class_weight=params['rf__class_weight'],
                            n_estimators = params['rf__n_estimators'], 
                            max_depth = params['rf__max_depth'],
                            min_samples_leaf = params['rf__min_samples_leaf'], 
                            max_features = params['rf__max_features'],
                            max_samples = params['rf__max_samples'])
    
    elif self.run_model =='svm':
      self.model.set_params(C=params['svc__C'], gamma=params['svc__gamma'])
    
    elif self.run_model =='logistic':
      self.model.set_params(C=params['logit__C'])
    
    # cv_results = cross_val_score(model, X_TRAIN, Y_TRAIN, cv=4, n_jobs=-1, scoring='f1')
    # scoring: valid options are ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 
    #  'log_loss', 'mean_absolute_error', 'mean_squared_error', 'precision', 'r2', 'recall', 'roc_auc']

    cv_results = cross_validate(self.model, self.x_train, self.y_train, cv=4, n_jobs=-1, scoring='roc_auc', return_train_score= True)

    overfit_devs = abs(cv_results['train_score'].mean() - cv_results['test_score'].mean())
    estimate = cv_results['test_score'].mean() - cv_results['test_score'].std() / 4 - overfit_devs / 3

    # calculate the mean of the scores
    # estimate = np.mean(result)

    # convert from a maximizing score to a minimizing score
    return 1.0 - estimate


  def evaluate_model_pipe(self,**params):
    
    self.model.set_params(**params)
    
    cv_results = cross_validate(self.model, self.x_train, self.y_train, cv=4, n_jobs=-1, scoring='roc_auc', return_train_score= True)

    overfit_devs = abs(cv_results['train_score'].mean() - cv_results['test_score'].mean())
    estimate = cv_results['test_score'].mean() - cv_results['test_score'].std() / 4 - overfit_devs / 3

    # calculate the mean of the scores
    # estimate = np.mean(result)

    # convert from a maximizing score to a minimizing score
    return 1.0 - estimate


  def optimized_training(self, X_train, y_train, s_model, model_obj, pipe=False, n_calls=50):

    start_time = time.time()
    # set class data
    self.x_train = X_train
    self.y_train = y_train
    self.run_model = s_model 
    self.model = model_obj
    
    self.create_search_space()
    
    @use_named_args(dimensions=self.search_space)
    def fitness_wrapper(**kwargs):
      if pipe:
        return self.evaluate_model_pipe(**kwargs)
      else:
        return self.evaluate_model(**kwargs)
      

    self.result_opt = forest_minimize(fitness_wrapper, self.search_space, n_calls=n_calls, n_jobs=-1, verbose=1, 
                                      kappa=4.00, callback=DeltaYStopper(0.0002, n_best=5))
    
    # summarizing finding:
    print('\noptimize_train():')
    print('Best Metric: %.3f' % (1.0 - self.result_opt.fun))
    for i, val in enumerate(self.result_opt.x):
      print('{}: {}'.format(self.param_name[i],val))
    print('execution time: {}'.format((time.time() - start_time)/60.))

    # result.x_iters
    # result.func_vals
    # plot optimized models
    # plot_objective(result, size=4)

    # run_best_params(s_model, result, param_name)
    return 

  
  def run_individual(self):

    l_score = []
    start_time = time.time()
    # ft_p = [['int_c_0.7', 'int_v', 'imp_c_0.9', 'imp_v', 'big_v', 'agg_net_m', 'loc_aggbig_saldo_d', 'PA_up', 'aggpior_c', 'pagg_c_best_0.7', 'escora_bid_3.5_2', 'escora_ask_2.5_1.2', 'movesc_bid_4.5', 'movesc_ask_2.5_0.9', 'escora_ask_4.5_2'],]
    ft_p = [self.x_train.columns.to_list(),]    # may include sets of features combinations here to run for each of them

    for l_ft in ft_p:
      print('\nModels will be evaluated using:')
      print(l_ft)

      # l_score has the following columns: 
      #   'model_type': 'svm', 
      #   'features': l_ft,
      #   'best_params': best,
      #   'score_is': s_is,
      #   'score_oos': s_oos

      print('---- KNN ---')
      knn = KNeighborsClassifier(algorithm='auto', n_jobs=-1)
      params = {'n_neighbors': [1, 2, 5, 10],
                'metric': ['minkowski', 'braycurtis'],
                'weights': ['uniform', 'distance']
                }
      l_return = rscv_clf(knn, params, self.x_train[l_ft], self.y_train, self.x_test[l_ft], self.y_test)
      l_score.append(['knn', l_ft] + l_return)


      print('---- SVM ---')
      svm = SVC(probability=True, class_weight='balanced', cache_size=1000, verbose=0)
      params = {'C': [.1, .5, 1, 10, 100], 
                'gamma': [0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6]
                }
      l_return = rscv_clf(svm, params, self.x_train[l_ft], self.y_train, self.x_test[l_ft], self.y_test)
      l_score.append(['svm', l_ft] + l_return)


      print('---- RFC ---')
      rfc = RandomForestClassifier(verbose=0, random_state=1, oob_score=True, n_jobs=-1, class_weight='balanced_subsample')
      # TODO: learn more about class_weight
      params = {'n_estimators': [50, 100, 300],
                'max_depth': [3, 8, 10, 17],
                'min_samples_split': [2, 10, 20],
                'min_samples_leaf': [1, 10, 50],
                'max_features': ['sqrt', 'log2', 0.1, 0.9],
                'criterion': ['gini', 'entropy'],
                'max_samples': [0.2, 0.5, 0.7, 0.9]
                }
      l_return = rscv_clf(rfc, params, self.x_train[l_ft], self.y_train, self.x_test[l_ft], self.y_test)
      l_score.append(['rfc', l_ft] + l_return)
      

      # print('---- GBM ---')
      # gbm = GradientBoostingClassifier(verbose=0, random_state=1)
      # params = {'learning_rate': [.1, .2, .3], 
      #           'n_estimators': [100, 300, 500],
      #           'subsample': [.5, .75, 1]
      #           }
      # l_return = rscv_clf(gbm, params, self.x_train[l_ft], self.y_train, self.x_test[l_ft], self.y_test)
      # l_score.append(['gbm', l_ft] + l_return)


      print('---- XGB ---')
      # data_dmatrix = xgb.DMatrix(data= self.x_train[l_ft], label=dy)
      params = {'colsample_bytree': [0.5, 0.7, 1],
                'subsample': [0.5, 1],
                'learning_rate': [0.1, 0.2, 0.3],
                'max_depth': [4, 6, 8], 
                'lambda': [1, 2, 3],   # L2
                'alpha': [0, 1, 10],   # L1
                'gamma': [0.5, 1, 1.5, 2, 5],
                'tree_method': ['auto', 'hist']  # TODO: to use gpu-hist needs to install package using pip
                }
      xgb_c = xgb.XGBClassifier(n_estimators=600, objective='binary:logistic', max_delta_step= 1,
                                eval_metric='auc',use_label_encoder=False)
      l_return = rscv_clf(xgb_c, params, self.x_train[l_ft], self.y_train, self.x_test[l_ft], self.y_test)
      l_score.append(['xgb', l_ft] + l_return)


      print('---- DLK ---')
      layers_drops = []
      n_layers_drops = 15
      p98_n_layers = 8
      p98_2n_layer_nodes = 128 # it means layer size <= 2**p98_2n_layer_nodes (98% conf)
      for j in range(n_layers_drops):
        n_layers = abs(int(np.random.randn() * p98_n_layers / 2.36))+1
        layers = []
        drops = []
        for i in range(n_layers):
          layers.append(2**(abs(int(np.random.randn() * p98_2n_layer_nodes / 2.36))+2))
          drops.append(np.random.random() * .6)
        layers_drops.append((tuple(layers), tuple(drops)))
      params = {'learn_rate': [.005, .01, .05], 
                'layers_drops': layers_drops, 
                'regularizer': [0.0001, 0.001, 0.1, 1],
                'epochs':  [20, 10], 
                'batch_size': [32, 64, 256]
                }
      
      global ddim 
      ddim = self.x_train.shape[1] 
      dl = KerasClassifier(build_fn=create_dl, verbose=0) 
      l_return = rscv_clf(dl, params, self.x_train[l_ft], self.y_train, self.x_test[l_ft], self.y_test)
      l_score.append(['dl_k', l_ft] + l_return)


      # exporting features, models and scores
      df_score = pd.DataFrame(l_score, columns=['model_type', 'features', 'score_train_cv', 'score_test_cv', 
                                                'score_is', 'score_oos', 'best_params', 'cm_is', 'th_train',
                                                'cm_oos', 'th_test'])
      # df_score['regime'] = md
      # s_file = self.s_path + 'score_details.csv'
      # b_head = True  # print header
      # if os.path.exists(s_path + 'score_details.csv'): b_head= False 
      # df_score.to_csv(s_file, mode='a', header=b_head)
      print(df_score)

      print('\n[{}] Execution time: {}'.format(self.s_regime, time.time() - start_time))



#%% SAVE and LOAD MODELS
def save_models(model, s_name, s_path):

  file_to_store = open(os.path.join(s_path, s_name + ".pkl"), "wb")
  pickle.dump(model, file_to_store)
  file_to_store.close()


def load_models(s_name, s_path):

  file_to_store = open(os.path.join(s_path, s_name + ".pkl"), "rb")
  model_object = pickle.load(file_to_store)
  file_to_store.close()

  return model_object


def base_norm(base, clip=0.001, norm_cols=[]):

  mup_perc = base.describe(percentiles=[clip,(1.0-clip)]).iloc[[4,6]]
  
  if 'y' in list(base.columns):
    mup_clip = base.drop(['y'], axis=1).copy()
  else:
    mup_clip = base.copy()
  
  cols = mup_clip.columns if norm_cols == [] else norm_cols
  for col in cols:
      mup_clip[col] = base[col].clip(mup_perc.iloc[0][col], mup_perc.iloc[1][col])
      mup_clip[col] = (mup_clip[col]-mup_clip[col].mean())/mup_clip[col].std()

  mup_clip.fillna(0.0, inplace=True)

  return mup_clip


def ft_importance_plot(model, base, plot_result=True):

  imp = pd.DataFrame({'cols':base.columns.tolist(), 'importance': model.feature_importances_})
  imp.sort_values(by=['importance'], inplace=True, ascending=False)
  
  if plot_result:
    plot_importance(imp)

  return imp  


def plot_importance(rf_imp):

  rf_imp_main = rf_imp.head(50)
  plt.figure(dpi=100, figsize=(16, 16))
  plt.bar(np.arange(rf_imp_main.shape[0]), rf_imp_main['importance'])
  plt.xticks(np.arange(rf_imp_main.shape[0]), rf_imp_main['cols'].tolist(), rotation = 90)


def optimize_threshold(y_true, y_prob):

  accuracies = []
  thresholds = np.arange(0.2,1,0.01)
  
  for th in thresholds:
    y_pred = np.greater(y_prob, th).astype(int)
    # acc = f1_score(y_true, y_pred)
    # acc = average_precision_score(y_true, y_pred)
    acc = fbeta_score(y_true, y_pred, beta=0.4)
    accuracies.append(acc)
  
  accuracies = np.array(accuracies)
  # max_accuracy = accuracies.max() 
  th_max_acc = thresholds[accuracies.argmax()]
  yhat_th = np.greater(y_prob, th_max_acc).astype(int)

  return yhat_th, th_max_acc


def yhat_threshold(y_prob, th):
  
  yhat_th = np.greater(y_prob, th).astype(int)
  
  return yhat_th


def report_results(X_train, X_test, y_train, y_test, fitted_model=None, th=0.0, verbose=True, y_prob_train=None, y_prob_test=None, ready_probs = False):

  if not ready_probs:
    y_prob_train = fitted_model.predict_proba(X_train)[:, 1]
    y_prob_test = fitted_model.predict_proba(X_test)[:, 1]

  if th == 0.0:
    yhat_train, th = optimize_threshold(y_train, y_prob_train)  # passing probs and getting integers y 
  else:
    yhat_train = yhat_threshold(y_prob_train,th)
    
  train_auc = roc_auc_score(y_train, y_prob_train)
  train_f1 = f1_score(y_train, yhat_train, average='binary')
  train_fbeta = fbeta_score(y_train, yhat_train, beta=0.4)
  cm_train = confusion_matrix(y_train, yhat_train)  # passing integers to CM

  yhat_test = yhat_threshold(y_prob_test,th)
  test_auc = roc_auc_score(y_test, y_prob_test)
  test_f1 = f1_score(y_test, yhat_test, average='binary')
  test_fbeta = fbeta_score(y_test, yhat_test, beta=0.4)
  cm_test = confusion_matrix(y_test, yhat_test) 

  ## SHOW SCORES
  if verbose:
    print('train_auc: {:.4f}, test_auc: {:.4f} (overfit: {:.4f})'.format(train_auc, test_auc, train_auc - test_auc))
    print('train_f1:  {:.4f}, test_f1:  {:.4f} (overfit: {:.4f})'.format(train_f1, test_f1, train_f1 - test_f1))
    print('train_f*:  {:.4f}, test_f*:  {:.4f} (overfit: {:.4f})'.format(train_fbeta, test_fbeta, train_fbeta - test_fbeta))
    print('--------')
    # print('test_auc: {:.3f}, overfit: {:.3f}'.format(test_auc, train_auc - test_auc))
    # print('test_f1: {:.3f}, overfit: {:.3f}'.format(test_f1, train_f1 - test_f1))
    # print('test_f1_balanced: {:.3f}, overfit: {:.3f}'.format(test_f1_balanced, train_f1_balanced - test_f1_balanced))
    # print('--------')
    print('cm_train: ')
    print(cm_train)
    print('cm_test: ')
    print(cm_test)
    print('threshold: {:.3f}'.format(th))
    
  return [train_auc, test_auc, train_f1, test_f1, train_fbeta, test_fbeta, cm_train, cm_test, yhat_train, yhat_test, th]


def rscv_clf(clf, d_params, X_train, y_train, X_test, y_test):

    start_time = time.time()
    
    clf_cv = RandomizedSearchCV(estimator=clf, param_distributions=d_params, n_iter=10, 
                              cv=5, verbose=1, return_train_score=True, scoring='roc_auc')  # or roc_auc_ovo_weighted
    clf_cv.fit(X_train, y_train)

    print("\nBest: %f using %s" % (clf_cv.best_score_, clf_cv.best_params_))
    search_means = clf_cv.cv_results_['mean_test_score']
    search_stds = clf_cv.cv_results_['std_test_score']
    overfit_devs = abs(clf_cv.cv_results_['mean_train_score'] - clf_cv.cv_results_['mean_test_score'])
    estimate = clf_cv.cv_results_['mean_test_score'] - clf_cv.cv_results_['std_test_score'] / 3 - overfit_devs / 2
    search_params = clf_cv.cv_results_['params']
    for mean, stdev, justfit, param in zip(search_means, search_stds, estimate, search_params):
      print("%f (%f) just_fit: %f with: %r" % (mean, stdev, justfit, param))

    l_results = report_results(X_train, X_test, y_train, y_test, fitted_model=clf_cv)

    print('\nrscv_clf - Execution time: {}'.format(time.time() - start_time))

    return l_results, clf_cv


def create_dl(learn_rate, layers_drops, regularizer):
    
    keras_dl = Sequential()

    ## LAYER (FIRST)
    keras_dl.add(Dense(layers_drops[0][0], input_dim = ddim, activation = 'relu', 
                kernel_regularizer = regularizers.l2(regularizer)))
    keras_dl.add(BatchNormalization())
    keras_dl.add(Dropout(rate = layers_drops[1][0]))

    # HIDDEN LAYERS
    for i in range(len(layers_drops[0])-1):
        keras_dl.add(Dense(layers_drops[0][i+1], activation = 'relu', 
                    kernel_regularizer = regularizers.l2(regularizer)))
        keras_dl.add(BatchNormalization())
        keras_dl.add(Dropout(rate = layers_drops[1][i+1]))

    ## LAYER (LAST)
    keras_dl.add(Dense(1, activation = 'sigmoid'))

    loss_function = 'binary_crossentropy'
    opt = keras.optimizers.Adam(learning_rate = learn_rate)
    keras_dl.compile(loss=loss_function, optimizer = opt, metrics = [metrics.AUC(),metrics.Accuracy()])

    return keras_dl





 
 