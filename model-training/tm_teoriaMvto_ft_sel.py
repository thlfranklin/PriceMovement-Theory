#%%
import os
import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import Lasso
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pd.set_option('display.max_columns',100)
pd.set_option('precision', 3)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

DEBUG = True
d_path = { 'train': '..\\neutrino-lab\\historical_files\\TrainFiles\\',
           'test': '..\\neutrino-lab\\historical_files\\TestFiles\\'}

s_path = d_path['train']

# LOADING BASES
file_name = 'X'
file_to_store = open(s_path + file_name + ".pickle", "rb")
X = pickle.load(file_to_store)
file_to_store.close()

file_name = 'y'
file_to_store = open(s_path + file_name + ".pickle", "rb")
y = pickle.load(file_to_store)
file_to_store.close()

#########################################################################################
########################## ######   ######  #######  ######  ############################
########################## ##    #  ##      ##      ##       ############################
########################## ##    #  #####   #####   #######  ############################
########################## ##    #  ##      ##           ##  ############################
########################## ######   ######  ##      ######   ############################
#########################################################################################
#%%

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

def make_mi_scores_c(X, y, discrete_features='auto'):
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features)
    # mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    # mi_scores = mi_scores.sort_values(ascending=False)
    mi_scores = pd.DataFrame({'cols': X.columns.tolist(), 'importance': mi_scores})
    mi_scores = mi_scores.sort_values(by='importance', ascending=False)
    return mi_scores

def mi_random(y):
    l_rands = []
    for i in range(50):
        l_rands += [np.random.random(y.count())]
        l_rands += [np.exp(np.random.random(y.count()))]
        l_rands += [np.random.lognormal(0,1,y.count())]
        l_rands += [np.random.randn(y.count())]
        l_rands += [np.zeros(y.count())]
        l_rands += [np.ones(y.count())]

    mi_random = make_mi_scores_c(pd.DataFrame(l_rands).T, y)
    mi_random = pd.DataFrame({'cols':mi_random.index.tolist(), 'importance': mi_random.values.tolist()})

    return mi_random

def plot_importance(rf_imp):
  rf_imp_main = rf_imp.head(50)
  plt.figure(dpi=100, figsize=(16, 16))
  plt.bar(np.arange(rf_imp_main.shape[0]), rf_imp_main['importance'])
  plt.xticks(np.arange(rf_imp_main.shape[0]), rf_imp_main['cols'].tolist(), rotation = 90)

def ft_importance_rf(x,y, plot_result=False, n_estimators=700, max_depth=5, max_samples=0.7):
  rf = RandomForestClassifier(class_weight='balanced_subsample', n_jobs=-1,
                              n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)
  rf.fit(x,y)
  rf_imp=pd.DataFrame({'cols':x.columns.tolist(), 'importance': rf.feature_importances_})
  rf_imp.sort_values(by=['importance'], inplace=True, ascending=False)

  if plot_result:
    plot_importance(rf_imp)

  return rf_imp

def ft_importance_xgb(x,y, plot_result=False, n_estimators=500, 
                      colsample_bytree= 0.7, subsample=0.5, learning_rate=0.001,
                      max_depth=7, gamma=0.5):
  xgb = XGBClassifier(n_estimators=n_estimators, objective='binary:logistic', eval_metric='auc',
                      use_label_encoder=False, colsample_bytree=colsample_bytree, 
                      subsample=subsample, learning_rate=learning_rate,
                      max_depth=max_depth, gamma=gamma, tree_method='hist')
  
  xgb.fit(x, y)
  xgb_imp = pd.DataFrame({'cols':x.columns.tolist(), 'importance': xgb.feature_importances_})
  xgb_imp.sort_values(by=['importance'], inplace=True, ascending=False)
  
  if plot_result:
    plot_importance(xgb_imp)

  return xgb_imp

def ft_importance_knn(x_norm,y, plot_result=False, n_neighbors=16, 
                      weights='distance', test_size=0, use_sample_weights=True):
  knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
  knn.fit(x_norm, y)
  if use_sample_weights:
    sample_weights = y * ((y.count()-y.sum())/y.sum() - 1) + 1
  else:
    sample_weights = y + 1
  end = x_norm.shape[0] if test_size == 0 else test_size
  results = permutation_importance(knn, x_norm.iloc[0:end], y.iloc[0:end], 
      sample_weight=sample_weights.iloc[0:end], scoring='roc_auc', n_jobs=-1, n_repeats=2)
  importance = results.importances_mean
  knn_imp = pd.DataFrame({'cols':x_norm.columns.tolist(), 'importance': importance})
  knn_imp.sort_values(by=['importance'], inplace=True, ascending=False)
  
  if plot_result:
    plot_importance(knn_imp)

  return knn_imp

def ft_importance_svm(x_norm,y, plot_result=False, test_size = 0):
  svm = SVC(class_weight='balanced')
  svm.fit(x_norm, y)
  end = x_norm.shape[0] if test_size == 0 else test_size
  results = permutation_importance(svm, x_norm.iloc[0:end], y.iloc[0:end], scoring='roc_auc', n_jobs=-1, n_repeats=2)
  importance = results.importances_mean
  imp = pd.DataFrame({'cols':x_norm.columns.tolist(), 'importance': importance})
  imp.sort_values(by=['importance'], inplace=True, ascending=False)
  
  if plot_result:
    plot_importance(imp)

  return imp

def ft_importance_lasso(x_norm,y, plot_result=False):
  # construir importancia via coeficientes de lasso, fazendo intersecao, com samplings diferentes
  # OU fazer simples e tratar por fora
  pass

def ft_importance_linear(x,y, plot_result=False):
  model = LogisticRegression(class_weight='balanced', penalty='l1')
  model.fit(x, y)
  importance = model.coef_[0]
  lr_imp=pd.DataFrame({'cols':x.columns.tolist(), 'importance': importance})
  lr_imp['imp_abs'] = lr_imp['importance'].abs()
  lr_imp.sort_values(by=['imp_abs'], inplace=True, ascending=False)
  lr_imp.drop('imp_abs', axis=1, inplace=True)

  if plot_result:
    plot_importance(lr_imp)

  return lr_imp

  # score = 'accuracy'
  # # score = 'roc_auc'
  # cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
  # scores = cross_val_score(model, x, y, scoring=score, cv=cv, n_jobs=-1)
  # print('Mean score: %.5f' % np.mean(scores))


def list_union(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list

def cut_features_union(base=None, ranks=None, thres = .2, return_only_cols=False):
  # ranks is a list of lists of columns of base, sorted descending by some importance metric
  final_cols = []
  for cols in ranks:
    final_cols = list_union(final_cols, cols[0:round(thres * len(cols))])
  
  if return_only_cols:
    return final_cols

  return base[final_cols].copy()

def cut_features_inter(base=None, ranks=None, thres = .2, return_only_cols=False):
  # ranks is a list of lists of columns of base, sorted descending by some importance metric
  del_cols = []
  for cols in ranks:
    del_cols = list_union(del_cols, cols[round(thres * len(cols)):])
  
  x = base.drop(columns=del_cols)

  if return_only_cols:
    return x.columns.tolist()

  return x

def corr_clean(base_col_sorted, threshold=.77):
  
  if list(base_col_sorted.columns) == list(base_col_sorted.index):
    corr = base_col_sorted
  else:
    corr = base_col_sorted.corr(method='spearman')

  del_cols = []
  for col in corr:    
    l_cols = list(corr[col].loc[(corr[col].abs() >= threshold)].index)
    
    if col in l_cols: 
      l_cols.remove(col)
    
    del_cols += l_cols

  if list(base_col_sorted.columns) == list(base_col_sorted.index):
    return del_cols
  
  return base_col_sorted.drop(columns = del_cols)

#%%
#################################################################################################
###############################  #####  #####   ######   ###### #################################
############################### ##     ##   ##  ##   ##  ##     #################################
############################### ##     ##   ##  ##   ##  ####   #################################
############################### ##     ##   ##  ##   ##  ##     #################################
###############################  #####  #####   ######   ###### #################################
#################################################################################################

X_train_norm, X_test_norm, y_train, y_test = train_test_split(base_norm(X), y, test_size = 0.15, shuffle = False)

# x = X_train_norm
# y = y_train

def ft_selection_model(x, y, model_type = None, n_max = 15):

  mi = make_mi_scores_c(x, y)

  if model_type == 'tree':
    fi_rf = ft_importance_rf(x, y, plot_result=False)      
    fi_rf2 = ft_importance_rf(x, y, plot_result=False, max_depth=4, max_samples=0.5)      
    fi_xgb = ft_importance_xgb(x, y, plot_result=False)      
    fi_xgb2 = ft_importance_xgb(x, y, plot_result=False, colsample_bytree= 0.8, subsample=0.7, 
                                learning_rate=0.005, max_depth=6, gamma=0)      
    # NORMALIZA TOP 25%

    # SOMA IMPORTANCIA DE CADA FEATURE (PONDERADA PELA METRICA JUST_FIT DE CADA MODELO), 
    # FAZ UM PRIMEIRO CORR_CLEAN COM BASE NA UNIAO, 
    
    # RE-RODA FEATURE IMPORTANCES, 
    
    # NORMALIZA TOPS, 
    # SOMA IMPORTANCIA DE CADA FEATURE (PONDERADA PELA METRICA JUST_FIT DE CADA MODELO)

    # SELECIONA TOP n_max

  elif model_type == 'knn':
    fi_knn = ft_importance_knn(x, y, plot_result=False, test_size = 7000)      
    fi_knn2 = ft_importance_knn(x, y, plot_result=False, test_size = 7000, 
                                            use_sample_weights=False, n_neighbors=4, weights='uniform')
  
  elif model_type == 'svm':
    fi_svm = ft_importance_svm(x, y, plot_result=True, test_size = 7000)      
  
  elif model_type == 'linear':
    fi_linear = ft_importance_linear(x, y, plot_result=True)      
    fi_linear2 = ft_importance_linear(x, y, plot_result=True)
    fi_linear3 = ft_importance_linear(x, y, plot_result=True)
    fi_linear4 = ft_importance_linear(x, y, plot_result=True)

  elif model_type == 'lasso':
    fi_lasso = ft_importance_lasso(x, y, plot_result=True)
    fi_lasso = ft_importance_lasso(x, y, plot_result=True)      

  else: 
    print("ERROR: ft_selection_model INVALID MODEL TYPE: {}".format(model_type))

def ft_selection_recursive(x, y, model_type = None, n_max = 15):
  pass

###############################
## IMPORTANCE VIA RECURSIVE FEATURE ELIMINATION

# from numpy import mean
# from numpy import std
# from sklearn.model_selection import cross_val_score

rfe = RFE(estimator=LogisticRegression(penalty='l1'), n_features_to_select=5)
model = LogisticRegression(penalty='l1')
pipeline = Pipeline(steps=[('s',rfe),('m',model)])
n_scores = cross_val_score(pipeline, x, y, scoring='roc_auc', cv=5, n_jobs=-1, error_score='raise')
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

########################################

def fi_ga(x, y, model_type = None):
  pass

## AGGREGATING IMPORTANCES 
ranks = [fi_rf['cols'].tolist(),
        fi_rf2['cols'].tolist(),
        fi_xgb['cols'].tolist(),
        fi_xgb2['cols'].tolist(),
        # fi_knn['cols'].tolist(),
        # fi_knn2['cols'].tolist(),
        # mi['cols'].tolist(),
        mi_norm['cols'].tolist(),
        fi_svm['cols'].tolist()]

ranks_tree = [fi_rf['cols'].tolist(),
        fi_rf2['cols'].tolist(),
        fi_xgb['cols'].tolist(),
        fi_xgb2['cols'].tolist()]

select_minimum = cut_features_union(x, ranks=ranks, return_only_cols=True, thres=0.04)

select_tree = cut_features_inter(x, ranks=ranks_tree, return_only_cols=True, thres=0.3)
select_tree

select = list_union(select_minimum, select_tree)
select

good = ['agg_net_10'
,'rng_smart_10'
,'ohlc_10'
,'smart_price_dif'
,'chgfreq'
,'n_trd'
,'rng_ewma_dif'
,'aggbig_net_d'
,'msg_imb_mean_dif_cp'
,'agg_net_40'
,'rng_ewma_dif_80'
,'smart_price_50'
,'book_imb'
,'pagg_c_best_0.7'
,'aggbig_net_m'
,'loc_agg_net_d'
,'agg_net_80'
,'loc_aggbig_imb_m']

gs = list_union(select, good)

knn1 = ['vewma','last_d_s'
,'vewmag_dif'
,'aggpior_c'
,'imp_v_0.9'
,'vewma_g_p']

gsk = list_union(gs, knn1)

knn2 = ['int_dif_c','aggbig_net_d'
,'s_run'
,'int_c'
,'int_c_0.6'
,'last_d_s']

gsk = list_union(gsk, knn2)

select_tree = cut_features_inter(x, ranks=ranks_tree, return_only_cols=True, thres=0.4)
bc = list_union(best_cols, select_tree)

x_selected_cols_union = cut_features_union(x, ranks=ranks, return_only_cols=True)
x_selected_cols_inter = cut_features_inter(x, ranks=ranks, return_only_cols=True)

fi_exp = pd.concat([fi_rf.reset_index(),
                    fi_rf2.reset_index(),
                    fi_xgb.reset_index(),
                    fi_xgb2.reset_index(),
                    fi_svm.reset_index(),
                    # fi_knn.reset_index(),
                    # fi_knn2.reset_index(),
                    mi.reset_index()], axis=1, ignore_index=True)

fi_exp.to_csv(s_path + 'feature_importances_original_short.csv')

fi_svm.to_csv(s_path + 'fi_svm.csv')

pd.Series(data=x_selected_cols_union).to_csv(s_path + 'best_features_union.csv')
pd.Series(data=x_selected_cols_inter).to_csv(s_path + 'best_features_inter.csv')

inter = ['book_imb', 'msg_imb', 'smart_price_dif', 'rng_smart_10', 'agg_net_10']
union = ['int_v_0.6','int_v_0.9','book_imb_mean_40','escora_ask_2.5_2','agg_net_40','rng_ewma_dif_40','imp_v_0.9',
'movesc_bid_2.5_0.5','book_imb_mean_us_20','loc_agg_imb_m','msg_imb','msg_imb_mean_dif_cp','agg_net_80','aggimb',
'movesc_ask_3.5_0.5','imp_v_0.8','escora_bid_2.5_2','int_v_0.7','movesc_ask_2.5_0.9','int_c_0.8','int_dif_v',
'n_trd','imp_c_0.6','escora_bid_2.5_1.2','smart_price_50','book_imb_mean_us_5','movesc_ask_2.5_0.5','chgfreq',
'rng_ewma_dif_80','rng_smart_50','agg_net_d','imp_c_0.9','int_c_0.6','movesc_ask_4.5','smart_price_dif','s_run',
 'agg_net_10','int_v_0.8','aggbig_net_d','book_imb_mean_dif_cp','escora_bid_3.5_2','imp_v_0.6','movesc_ask_3.5',
 'int_c_0.7','imp_c_0.7','sspread_mean_us_5','movesc_ask_2.5_0.7','book_imb_dif','book_imb_mean_10','vewma_c_v',
 'msg_imb_dif','pagg_c_best_0.9','rng_ewma','pagg_v_best_0.5','imp_c_0.8','ohlc_50','ohlc_10','escora_ask_4.5_2',
 'rng_smart_10','escora_ask_4.5_0.8','imp_v','imp_c','escora_ask_2.5_1.2','movesc_ask_3.5_0.9','rng_ewma_dif',
 'book_imb','pagg_c_best_0.5','pagg_c_best_0.7','loc_aggbig_imb_m']

##################################################################################
#%% LIMPAR FEATURES VIA CORRELATION MATRIX

# miup_corr = miup_new.corr(method='spearman')
# midown_corr = midown_new.corr(method='spearman')
# mwup_corr = mwup_new.corr(method='spearman')
# mwdown_corr = mwdown_new.corr(method='spearman')

# miup_new.drop(columns = corr_clean(miup_corr), inplace = True)
# midown_new.drop(columns = corr_clean(midown_corr), inplace = True)
# mwup_new.drop(columns = corr_clean(mwup_corr), inplace = True)
# mwdown_new.drop(columns = corr_clean(mwdown_corr), inplace = True)


# ordenar colunas
miup_new = miup_new[fi_miup['cols'].tolist()]
midown_new = midown_new[fi_midown['cols'].tolist()]
mwup_new = mwup_new[fi_mwup['cols'].tolist()]
mwdown_new = mwdown_new[fi_mwdown['cols'].tolist()]

miup_new = corr_clean(miup_new)
midown_new = corr_clean(midown_new)
mwup_new = corr_clean(mwup_new)
mwdown_new = corr_clean(mwdown_new)

##################################################################################
# this section will be replaced by cluster clean
d_ft = {}

d_ft['int_c'] = ['int_c', 'int_c_0.6', 'int_c_0.7', 'int_c_0.8', 'int_c_0.9', 'int_dif_c',]
d_ft['int_v'] = ['int_v', 'int_v_0.6', 'int_v_0.7', 'int_v_0.8', 'int_v_0.9', 'int_dif_v',]
d_ft['imp_c'] = ['imp_c', 'imp_c_0.6', 'imp_c_0.7', 'imp_c_0.8', 'imp_c_0.9', ]
d_ft['imp_v'] = ['imp_v', 'imp_v_0.6', 'imp_v_0.7', 'imp_v_0.8', 'imp_v_0.9',]
d_ft['volume'] = ['vol_trd','n_trd','vol_big', 'n_big', 'big_c', 'big_v', 'n_p', 'vol_big_ratio']
d_ft['aggimb'] = ['loc_agg_imb_m', 'loc_aggbig_imb_m', 'aggimb', 'aggimb_big', 'n_aggimb', 'agg_net_m', 'aggbig_net_m', 'loc_agg_saldo_m', 'loc_aggbig_saldo_m']
d_ft['netagg'] = ['loc_agg_saldo_d', 'loc_aggbig_saldo_d', 'vewma', 'rng_ewma', 'vewma_g_p', 'vewmag_dif', 'agg_net_d',]
d_ft['relev'] = ['book_imb', 'abs_c', 'abs_v', 'PA_up', 'PA_down', 'smartprice_dif', 'sspread_mean',]
d_ft['others'] = ['abagg', 'aggpior_c', 'aggpior_v', 'chgfreq', 'last_d_s',  'msg_imb',]  #'s_run', 
d_ft['pagg'] = ['pagg_c_best', 'pagg_c_best_0.5', 'pagg_c_best_0.7', 'pagg_c_best_0.9',
                'pagg_v_best', 'pagg_v_best_0.5', 'pagg_v_best_0.7', 'pagg_v_best_0.9',]

d_ft['esc_bid'] = ['escora_bid_2.5_0.8', 'escora_bid_2.5_1.2', 'escora_bid_2.5_2',
                   'escora_bid_3.5_0.8', 'escora_bid_3.5_1.2', 'escora_bid_3.5_2', 
                   'escora_bid_4.5_0.8', 'escora_bid_4.5_1.2', 'escora_bid_4.5_2',]

d_ft['esc_ask'] = ['escora_ask_2.5_0.8', 'escora_ask_2.5_1.2', 'escora_ask_2.5_2',
                  'escora_ask_3.5_0.8', 'escora_ask_3.5_1.2', 'escora_ask_3.5_2',]

d_ft['movesc_b'] = ['movesc_bid_2.5', 'movesc_bid_2.5_0.5', 'movesc_bid_2.5_0.7', 'movesc_bid_2.5_0.9', 
                      'movesc_bid_3.5', 'movesc_bid_3.5_0.5', 'movesc_bid_3.5_0.7',  'movesc_bid_3.5_0.9',
                      'movesc_bid_4.5', 'movesc_bid_4.5_0.5', 'movesc_bid_4.5_0.7', 'movesc_bid_4.5_0.9',]

d_ft['movesc_a'] = ['movesc_ask_2.5', 'movesc_ask_2.5_0.5', 'movesc_ask_2.5_0.7',  'movesc_ask_2.5_0.9', 
                      'movesc_ask_3.5',  'movesc_ask_3.5_0.5',  'movesc_ask_3.5_0.7', 'movesc_ask_3.5_0.9',]

d_ft['esc_4'] = ['escora_ask_4.5_0.8',  'escora_ask_4.5_1.2',  'escora_ask_4.5_2', 
                 'movesc_ask_4.5',  'movesc_ask_4.5_0.5',  'movesc_ask_4.5_0.7',   'movesc_ask_4.5_0.9',]

def random_choice():
  import random 
  ft_p = list()
  for i in range(4):
    aux = list()
    for l_group in d_ft.keys():
      aux.append(random.choice(d_ft[l_group]))
    
    ft_p.append(aux)

  print('\n{} groups of features picked from d_ft to train and test.'.format(len(ft_p)))
