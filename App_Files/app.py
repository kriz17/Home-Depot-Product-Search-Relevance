import pandas as pd
import numpy as np
import regex as re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import sent_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
import math
from numpy.linalg import norm
import pickle
from flask import Flask, jsonify, request

import flask
app = Flask(__name__)
print('Imports Done!')

print('Loadings starting')
###############################################################################################
###############################################################################################
database  = pd.read_csv('database.csv')
print('database loaded')
product_text = database['text'].values
with open('BM25_model.pkl', 'rb') as f:
  bm25_model = pickle.load(f)
with open('Final/cleaning/unique_brands.pkl','rb') as f:
  unique_brands = pickle.load(f)
def words(text): return re.findall(r'\w+', text.lower())
WORDS = Counter(words(open('Final/cleaning/corpus.txt').read()))
porter = PorterStemmer()
stp_wrds = set(stopwords.words('english'))
corrected_feat_set = ['num_common_ST','num_common_SD', 'num_common_SB', 'cosine_ST', 'cosine_SD', \
                'cosine_SB', 'jacquard_ST', 'jacquard_SD', 'jacquard_SB',\
                'len_description', 'len_brand', 'len_title', 'len_search',\
                'islast_ST', 'islast_SD', 'islast_SB']
raw_feat_set = ['num_common_r_ST', 'num_common_r_SD', 'num_common_r_SB', 'cosine_r_ST', 'cosine_r_SD',\
                'cosine_r_SB',  'jacquard_r_ST', 'jacquard_r_SD', 'jacquard_r_SB',\
                'len_description', 'len_brand', 'len_title', 'len_r_search',\
                'islast_r_ST','islast_r_SD', 'islast_r_SB']
feat_set1_comb =  ['num_common_ST','num_common_SD', 'num_common_SB', 'cosine_ST', 'cosine_SD',\
                'cosine_SB', 'jacquard_ST', 'jacquard_SD', 'jacquard_SB',\
                'len_description', 'len_brand', 'len_title', 'len_search',\
                'islast_ST', 'islast_SD', 'islast_SB', 'num_common_r_ST', 'num_common_r_SD', 'num_common_r_SB', 'cosine_r_ST', 'cosine_r_SD',\
                'cosine_r_SB',  'jacquard_r_ST', 'jacquard_r_SD', 'jacquard_r_SB',\
                'len_r_search', 'islast_r_ST','islast_r_SD', 'islast_r_SB']
with open('Final/featurization/F2_tfidf_search.pkl','rb') as f:
  F2_tfidf_search = pickle.load(f)
with open('Final/featurization/F2_tsvd_search.pkl','rb') as f:
  F2_tsvd_search = pickle.load(f)
with open('Final/featurization/F2_tfidf_title.pkl','rb') as f:
  F2_tfidf_title = pickle.load(f)
with open('Final/featurization/F2_tsvd_title.pkl','rb') as f:
  F2_tsvd_title = pickle.load(f)
with open('Final/featurization/F2_tfidf_desc.pkl','rb') as f:
  F2_tfidf_desc = pickle.load(f)
with open('Final/featurization/F2_tsvd_desc.pkl','rb') as f:
  F2_tsvd_desc = pickle.load(f)
with open('Final/featurization/F2_tfidf_lsi.pkl','rb') as f:
  F2_tfidf_lsi = pickle.load(f)
with open('Final/featurization/F2_tsvd_lsi.pkl','rb') as f:
  F2_tsvd_lsi = pickle.load(f)
with open('Final/featurization/F3_LM_params_title.pkl','rb') as f:
  params_title_LM = pickle.load(f)
with open('Final/featurization/F3_LM_params_brand.pkl','rb') as f:
  params_brand_LM = pickle.load(f)
with open('Final/featurization/F3_LM_params_desc.pkl','rb') as f:
  params_desc_LM = pickle.load(f)
with open('Final/featurization/F3_bm25_params_title.pkl','rb') as f:
  params_title_bm25 = pickle.load(f)
with open('Final/featurization/F3_bm25_params_desc.pkl','rb') as f:
  params_desc_bm25 = pickle.load(f)
with open('Final/featurization/F3_bm25_params_brand.pkl','rb') as f:
  params_brand_bm25 = pickle.load(f)
with open('glove_vectors', 'rb') as f:
    model = pickle.load(f)
    glove_words =  set(model.keys())
with open('Final/featurization/F3_tfidf_w2v_params_search.pkl','rb') as f:
  tfidf_w2v_params_search = pickle.load(f)
with open('Final/featurization/F3_tfidf_w2v_params_title.pkl','rb') as f:
  tfidf_w2v_params_title = pickle.load(f)
with open('Final/featurization/F3_tfidf_w2v_params_desc.pkl','rb') as f:
  tfidf_w2v_params_desc = pickle.load(f)
with open('Final/featurization/F3_SmM_params_title.pkl','rb') as f:
  F3_SmM_params_title = pickle.load(f)
with open('Final/featurization/F3_SmM_params_brand.pkl','rb') as f:
  F3_SmM_params_brand = pickle.load(f)
with open('Final/featurization/F3_SmM_params_desc.pkl','rb') as f:
  F3_SmM_params_desc = pickle.load(f)

F1_scaler_ridge = pickle.load(open('Final/modelling/base_models/F1_scaler_ridge.pkl', 'rb'))
F1_scaler_lasso = pickle.load(open('Final/modelling/base_models/F1_scaler_lasso.pkl', 'rb'))
F1_scaler_en = pickle.load(open('Final/modelling/base_models/F1_scaler_en.pkl', 'rb'))
F1_scaler_svr = pickle.load(open('Final/modelling/base_models/F1_scaler_svr.pkl', 'rb'))

F1_xgb = pickle.load(open('Final/modelling/base_models/F1_xgb.pkl', 'rb'))
F1_rf = pickle.load(open('Final/modelling/base_models/F1_rf.pkl', 'rb'))
F1_ridge = pickle.load(open('Final/modelling/base_models/F1_ridge.pkl', 'rb'))
F1_lasso = pickle.load(open('Final/modelling/base_models/F1_lasso.pkl', 'rb'))
F1_en = pickle.load(open('Final/modelling/base_models/F1_en.pkl', 'rb'))
F1_svr = pickle.load(open('Final/modelling/base_models/F1_svr.pkl', 'rb'))
F1_dt = pickle.load(open('Final/modelling/base_models/F1_dt.pkl', 'rb'))

#Loading the standard scalers
F2_scaler_ridge = pickle.load(open('Final/modelling/base_models/F2_scaler_ridge.pkl', 'rb'))
F2_scaler_lasso = pickle.load(open('Final/modelling/base_models/F2_scaler_lasso.pkl', 'rb'))
F2_scaler_en = pickle.load(open('Final/modelling/base_models/F2_scaler_en.pkl', 'rb'))
F2_scaler_svr = pickle.load(open('Final/modelling/base_models/F2_scaler_svr.pkl', 'rb'))

#Loading the models
F2_svr = pickle.load(open('Final/modelling/base_models/F2_svr.pkl', 'rb'))
F2_rf = pickle.load(open('Final/modelling/base_models/F2_rf.pkl', 'rb'))
F2_ridge = pickle.load(open('Final/modelling/base_models/F2_ridge.pkl', 'rb'))
F2_lasso = pickle.load(open('Final/modelling/base_models/F2_lasso.pkl', 'rb'))
F2_en = pickle.load(open('Final/modelling/base_models/F2_en.pkl', 'rb'))
F2_dt = pickle.load(open('Final/modelling/base_models/F2_dt.pkl', 'rb'))

#Loading the standard scalers
F3_scaler_en = pickle.load(open('Final/modelling/base_models/F3_scaler_en.pkl', 'rb'))
F3_scaler_ridge = pickle.load(open('Final/modelling/base_models/F3_scaler_ridge.pkl', 'rb'))
F3_scaler_lasso = pickle.load(open('Final/modelling/base_models/F3_scaler_lasso.pkl', 'rb'))

#Loading the models
F3_ridge = pickle.load(open('Final/modelling/base_models/F3_ridge.pkl', 'rb'))
F3_lasso = pickle.load(open('Final/modelling/base_models/F3_lasso.pkl', 'rb'))
F3_en = pickle.load(open('Final/modelling/base_models/F3_en.pkl', 'rb'))
F3_dt = pickle.load(open('Final/modelling/base_models/F3_dt.pkl', 'rb'))

final_scaler = pickle.load(open('Final/modelling/meta_scaler.pkl', 'rb'))
final_ridge = pickle.load(open('Final/modelling/meta_ridge.pkl', 'rb'))
print('All files loaded')

###############################################################################################
###############################################################################################

print('starting with function definitions')
def get_top100(search, bm25_model, corpus, database):
  tokenized_query = search_query.split()
  top100 = bm25_model.get_top_n(tokenized_query, corpus, n=100)
  top100_products = database[database['text'].isin(top100)].drop('text', axis=1)
  top100_products['search_term'] = search_query
  reorder_cols = ['product_uid', 'product_title', 'search_term', 'combined_attr', 'brand', 'product_description']
  return top100_products[reorder_cols]

def first_n(n, sent):
  if n > len(sent.split()):
    return 'error101'
  return ' '.join(sent.split()[:n])

def fillna_brand(data, unique_brnds):
  null_df = data[data['brand'].isnull()]
  notnull_df = data.dropna()

  for i, row in null_df.iterrows():
    title = row['product_title']
    if first_n(4, title) in unique_brnds:
      null_df['brand'].loc[i] = first_n(4, title)
    elif first_n(3, title) in unique_brnds:
      null_df['brand'].loc[i] = first_n(3, title)
    elif first_n(2, title) in unique_brnds:
      null_df['brand'].loc[i] = first_n(2, title)
    else:
      null_df['brand'].loc[i] = first_n(1, title)

  data['brand'].loc[null_df.index] = null_df['brand'].values
  return data

def fillna_attributes(data):
  null_df = data[data['combined_attr'].isnull()]
  null_df['combined_attr'] = null_df['product_description'].copy()
  data['combined_attr'].loc[null_df.index] = null_df['combined_attr'].values
  return data

def standardize_units(text):
  text = " " + text + " "
  text = re.sub('( gal | gals | galon )',' gallon ',text)
  text = re.sub('( ft | fts | feets | foot | foots )',' feet ',text)
  text = re.sub('( squares | sq )',' square ',text)
  text = re.sub('( lb | lbs | pounds )',' pound ',text)
  text = re.sub('( oz | ozs | ounces | ounc )',' ounce ',text)
  text = re.sub('( yds | yd | yards )',' yard ',text)
  return text

def preprocessing(sent):
  sent = sent.replace('in.', ' inch ')
  words = re.split(r'\W+', sent)
  words = [word.lower() for word in words]
  res = re.sub("[A-Za-z]+", lambda ele: " " + ele[0] + " ", ' '.join(words))
  cleaned = standardize_units(res)
  cleaned = ' '.join(cleaned.split())
  return cleaned

def preprocessing_search(sent):
  sent = sent.replace('in.', ' inch ')
  words = re.split(r'\W+', sent)
  words = [word.lower() for word in words]
  res = re.sub("[A-Za-z]+", lambda ele: " " + ele[0] + " ", ' '.join(words))
  res = standardize_units(res)
  res = res.replace(' in ', ' inch ')
  cleaned = ' '.join(res.split())
  return cleaned

def words(text): return re.findall(r'\w+', text.lower())

def P(word, N=sum(WORDS.values())):
    "Probability of `word`."
    return WORDS[word] / N
def correction(word):
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)
def candidates(word):
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or set([word]))
def known(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)
def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)
def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))
def corrected_term(term):
  temp = term.lower().split()
  temp = [correction(word) for word in temp]
  return ' '.join(temp)

def futher_preprocessing(sent):
  sent = sent.replace('_', ' _ ')
  words = sent.split()
  words = [w for w in words if not w in stp_wrds]
  words = [porter.stem(word) for word in words]
  return ' '.join(words)
def futher_preprocessing_without_stem(sent):
  sent = sent.replace('_', ' _ ')
  words = sent.split()
  words = [w for w in words if not w in stp_wrds]
  return ' '.join(words)

def common_words(df, col1, col2):
  common_list = []
  for i, row in df[[col1,col2]].iterrows():
    set1 = set(row[col1].split())
    set2 = set(row[col2].split())
    common = set1 & set2
    common = ' '.join(common)
    common_list.append(common)
  return common_list

def cosine_similarity_sent(sent1, sent2):
  set1 = set(sent1.split())
  set2 = set(sent2.split())
  numerator = len(set1 & set2)
  denominator = math.sqrt(len(set1)) * math.sqrt(len(set2))
  if not denominator:
      return 0.0
  else:
      return numerator / denominator

def jacquard_coefficient_sent(sent1, sent2):
  set1 = set(sent1.split())
  set2 = set(sent2.split())
  numerator = len(set1 & set2)
  denominator = len(set1 | set2)
  if not denominator:
      return 0.0
  else:
      return numerator / denominator

def cosine_similarity_vec(a, b):
  num = np.dot(a, b)
  den = norm(a)*norm(b)
  if den != 0:
    return num/den
  else:
    return 0

def jacquard_similarity_vec(a, b):
  num = np.dot(a,b)
  den = norm(a)**2 + norm(b)**2 - np.dot(a,b)
  if den != 0:
    return num/den
  else:
    return 0

def inner_product_vec(a, b):
  return np.dot(a,b)

def lmir_fit(corpus):
  words = ' '.join(corpus).split()
  freq_dict = Counter(words)
  total_words = len(words)
  params = {
      'freq_dict':freq_dict,
      'total_words':total_words
  }
  return params

def lmir_jm_score(query, doc, params, lambd):
  query = query.split()
  doc = doc.split()
  if len(doc) != 0 and len(query) != 0:
    eps = 0.0001/(params['total_words'])
    score = 0
    for word in query:
      p_ml = doc.count(word) / len(doc)
      if word in params['freq_dict'].keys():
        p_c = params['freq_dict'][word] / params['total_words']
      else:
        p_c = 0
      score += np.log(lambd*p_ml + (1-lambd)*p_c + eps)
    return score

def lmir_dir_score(query, doc, params, mu):
  query = query.split()
  doc = doc.split()
  if len(doc) != 0 and len(query) != 0:
    eps = 0.0001/(params['total_words'])
    score = 0
    for word in query:
      p_ml = doc.count(word) / len(doc)
      if word in params['freq_dict']:
        p_c = params['freq_dict'][word] / params['total_words']
      else:
        p_c = 0
      lambd = len(doc) / (len(doc) + mu)
      score += np.log(lambd*p_ml + (1-lambd)*p_c + eps)
    return score

def lmir_abs_score(query, doc, alpha):
  query = query.split()
  doc = doc.split()
  if len(doc) != 0 and len(query) != 0:
    score = 0
    temp_dict = {k:v+alpha for k,v in Counter(doc).items()}
    for word in query:
      if word in temp_dict:
        pass
      else:
        temp_dict[word] = alpha
    denominator = sum(temp_dict.values())
    for word in query:
      score += temp_dict[word] / denominator
  return score

def bm25_fit(corpus):
  tfidf_model = TfidfVectorizer(smooth_idf=False, token_pattern=r"(?u)\b\w+\b")
  tfidf_model.fit(corpus)
  idf_dict = dict(zip(tfidf_model.get_feature_names(), list(tfidf_model.idf_)))
  avgdl = np.mean([len(doc.split()) for doc in corpus])
  params = {'idf_dict': idf_dict,
            'avgdl' : avgdl,
            'N' : N}
  return params

def bm25_score(query, doc, params, k=0.1, b=0.5):
  idf_dict = params['idf_dict']
  avgdl = params['avgdl']
  N = params['N']
  score_query = 0
  for word in query.split():
    dl = len(doc.split())
    tf = doc.count(word)
    if word in idf_dict.keys():
      idf = idf_dict[word]
    else:
      idf = np.log(N+1)
    score_word = idf*(tf*(k+1))/(tf + k*(1-b) + b*dl/avgdl)
    score_query += score_word
  return score_query

def get_search_relevance(test_set):
  test_set = fillna_brand(test_set, unique_brands)
  test_set = fillna_attributes(test_set)
  test_set = test_set.fillna('')

  test_set['cleaned_title'] = test_set['product_title'].apply(lambda x : preprocessing(x))
  test_set['cleaned_brand'] = test_set['brand'].apply(lambda x : preprocessing(x))
  test_set['cleaned_description'] = test_set['product_description'].apply(lambda x : preprocessing(x))
  test_set['cleaned_attributes'] = test_set['combined_attr'].apply(lambda x : preprocessing(x))
  test_set['cleaned_search'] = test_set['search_term'].apply(lambda x : preprocessing_search(x))

  test_set['corrected_search'] = test_set['cleaned_search'].apply(lambda x: corrected_term(x))

  cleaned_test_set = pd.DataFrame()
  cleaned_test_set['title'] = test_set['cleaned_title'].apply(lambda x : futher_preprocessing(x))
  cleaned_test_set['brand'] = test_set['cleaned_brand'].apply(lambda x : futher_preprocessing(x))
  cleaned_test_set['description'] = test_set['cleaned_description'].apply(lambda x : futher_preprocessing(x))
  cleaned_test_set['attributes'] = test_set['cleaned_attributes'].apply(lambda x : futher_preprocessing(x))
  cleaned_test_set['search'] = test_set['cleaned_search'].apply(lambda x : futher_preprocessing(x))
  cleaned_test_set['corrected_search'] = test_set['corrected_search'].apply(lambda x : futher_preprocessing(x))

  cleaned_test_set2 = pd.DataFrame()
  cleaned_test_set2['title'] = test_set['cleaned_title'].apply(lambda x : futher_preprocessing_without_stem(x))
  cleaned_test_set2['brand'] = test_set['cleaned_brand'].apply(lambda x : futher_preprocessing_without_stem(x))
  cleaned_test_set2['description'] = test_set['cleaned_description'].apply(lambda x : futher_preprocessing_without_stem(x))
  cleaned_test_set2['attributes'] = test_set['cleaned_attributes'].apply(lambda x : futher_preprocessing_without_stem(x))
  cleaned_test_set2['search'] = test_set['cleaned_search'].apply(lambda x : futher_preprocessing_without_stem(x))
  cleaned_test_set2['corrected_search'] = test_set['corrected_search'].apply(lambda x : futher_preprocessing_without_stem(x))

  cleaned_test_set['brand'] = cleaned_test_set['brand'].replace(to_replace =[""], value ="missing_brand")
  cleaned_test_set2['brand'] = cleaned_test_set2['brand'].replace(to_replace =[""], value ="missing_brand")
  cleaned_test_set['search'] = cleaned_test_set['search'].replace(to_replace =[""], value ="missing_search")
  cleaned_test_set2['search'] = cleaned_test_set2['search'].replace(to_replace =[""], value ="missing_search")

  cleaned_test_set['attributes'] = cleaned_test_set['attributes'].apply(lambda x: re.sub('bullet \d\d ', '', x))
  cleaned_test_set2['attributes'] = cleaned_test_set2['attributes'].apply(lambda x: re.sub('bullet \d\d ', '', x))

  cleaned_test_set['description'] = cleaned_test_set['description'].apply(lambda x: re.sub('bullet \d\d ', '', x))
  cleaned_test_set2['description'] = cleaned_test_set2['description'].apply(lambda x: re.sub('bullet \d\d ', '', x))

  cleaned_test_set.rename(columns={"search": "raw_search"}, inplace=True)
  cleaned_test_set2.rename(columns={"search": "raw_search"}, inplace=True)

  #FEATURIZATION
  #set1
  data1 = cleaned_test_set.copy()

  data1['common_ST'] = common_words(data1,'corrected_search', 'title')
  data1['common_SD'] = common_words(data1,'corrected_search', 'description')
  data1['common_SB'] = common_words(data1,'corrected_search', 'brand')
  data1['common_r_ST'] = common_words(data1,'raw_search', 'title')
  data1['common_r_SD'] = common_words(data1,'raw_search', 'description')
  data1['common_r_SB'] = common_words(data1,'raw_search', 'brand')

  data1['num_common_ST'] = data1['common_ST'].apply(lambda x : len(x.split()))
  data1['num_common_SD'] = data1['common_SD'].apply(lambda x : len(x.split()))
  data1['num_common_SB'] = data1['common_SB'].apply(lambda x : len(x.split()))
  data1['num_common_r_ST'] = data1['common_r_ST'].apply(lambda x : len(x.split()))
  data1['num_common_r_SD'] = data1['common_r_SD'].apply(lambda x : len(x.split()))
  data1['num_common_r_SB'] = data1['common_r_SB'].apply(lambda x : len(x.split()))

  data1['cosine_ST'] = data1.apply(lambda row: cosine_similarity_sent(row['corrected_search'], row['title']), axis=1)
  data1['cosine_SD'] = data1.apply(lambda row: cosine_similarity_sent(row['corrected_search'], row['description']), axis=1)
  data1['cosine_SB'] = data1.apply(lambda row: cosine_similarity_sent(row['corrected_search'], row['brand']), axis=1)
  data1['cosine_r_ST'] = data1.apply(lambda row: cosine_similarity_sent(row['raw_search'], row['title']), axis=1)
  data1['cosine_r_SD'] = data1.apply(lambda row: cosine_similarity_sent(row['raw_search'], row['description']), axis=1)
  data1['cosine_r_SB'] = data1.apply(lambda row: cosine_similarity_sent(row['raw_search'], row['brand']), axis=1)

  data1['jacquard_ST'] = data1.apply(lambda row: jacquard_coefficient_sent(row['corrected_search'], row['title']), axis=1)
  data1['jacquard_SD'] = data1.apply(lambda row: jacquard_coefficient_sent(row['corrected_search'], row['description']), axis=1)
  data1['jacquard_SB'] = data1.apply(lambda row: jacquard_coefficient_sent(row['corrected_search'], row['brand']), axis=1)
  data1['jacquard_r_ST'] = data1.apply(lambda row: jacquard_coefficient_sent(row['raw_search'], row['title']), axis=1)
  data1['jacquard_r_SD'] = data1.apply(lambda row: jacquard_coefficient_sent(row['raw_search'], row['description']), axis=1)
  data1['jacquard_r_SB'] = data1.apply(lambda row: jacquard_coefficient_sent(row['raw_search'], row['brand']), axis=1)

  data1['len_description'] = data1['description'].apply(lambda x : len(x.split()))
  data1['len_brand'] = data1['brand'].apply(lambda x : len(x.split()))
  data1['len_title'] = data1['title'].apply(lambda x : len(x.split()))
  data1['len_search'] = data1['corrected_search'].apply(lambda x : len(x.split()))
  data1['len_r_search'] = data1['raw_search'].apply(lambda x : len(x.split()))

  data1['islast_ST'] = data1.apply(lambda row: row['corrected_search'].split()[-1] in row['title'].split(), axis=1)
  data1['islast_SD'] = data1.apply(lambda row: row['corrected_search'].split()[-1] in row['description'].split(), axis=1)
  data1['islast_SB'] = data1.apply(lambda row: row['corrected_search'].split()[-1] in row['brand'].split(), axis=1)
  data1['islast_r_ST'] = data1.apply(lambda row: row['raw_search'].split()[-1] in row['title'].split(), axis=1)
  data1['islast_r_SD'] = data1.apply(lambda row: row['raw_search'].split()[-1] in row['description'].split(), axis=1)
  data1['islast_r_SB'] = data1.apply(lambda row: row['raw_search'].split()[-1] in row['brand'].split(), axis=1)

  bool_cols = ['islast_ST', 'islast_SD', 'islast_SB', 'islast_r_ST', 'islast_r_SD', 'islast_r_SB']
  for col in bool_cols:
    data1[col] = data1[col].astype(int)

  #set2
  data2 = cleaned_test_set.copy()

  X_search = F2_tfidf_search.transform(data2['corrected_search'])
  truncated_search = F2_tsvd_search.transform(X_search)
  X_title = F2_tfidf_title.transform(data2['title'])
  truncated_title = F2_tsvd_title.transform(X_title)
  X_desc = F2_tfidf_desc.transform(data2['description'])
  truncated_desc = F2_tsvd_desc.transform(X_desc)
  trun_arr = np.hstack((truncated_search,truncated_title,truncated_desc))
  truncated_df = pd.DataFrame(trun_arr, index=cleaned_test_set.index)

  title_desc = data2["title"].astype(str) + ' ' + data2["description"].astype(str)
  X_title_desc = F2_tfidf_lsi.transform(title_desc)
  truncated_title_desc = F2_tsvd_lsi.transform(X_title_desc)
  X_search_ = F2_tfidf_lsi.transform(data2['corrected_search'])
  transformed_search = F2_tsvd_lsi.transform(X_search_)

  cos_sim = []
  for i in range(len(transformed_search)):
    cos_sim.append(cosine_similarity_vec(truncated_title_desc[i], transformed_search[i]))
  data2['lsi_cos_sim'] = cos_sim
  jaq_sim = []
  for i in range(len(transformed_search)):
    jaq_sim.append(jacquard_similarity_vec(truncated_title_desc[i], transformed_search[i]))
  data2['lsi_jaq_sim'] = jaq_sim
  inn_prod = []
  for i in range(len(transformed_search)):
    inn_prod.append(inner_product_vec(truncated_title_desc[i], transformed_search[i]))
  data2['lsi_inn_prod'] = inn_prod

  data2 = data2[['lsi_cos_sim', 'lsi_jaq_sim',  'lsi_inn_prod']]

  #set3
  data3 = cleaned_test_set.copy()

  data3['JM_ST'] = data3.apply(lambda row: lmir_jm_score(row['corrected_search'], row['title'], params_title_LM, 0.9), axis=1)
  data3['Dir_ST'] = data3.apply(lambda row: lmir_dir_score(row['corrected_search'], row['title'], params_title_LM, 12 ), axis=1)
  data3['AD_ST'] = data3.apply(lambda row: lmir_abs_score(row['corrected_search'], row['title'], 0.01 ), axis=1)
  data3['JM_SB'] = data3.apply(lambda row: lmir_jm_score(row['corrected_search'], row['brand'], params_brand_LM, 0.9), axis=1)
  data3['Dir_SB'] = data3.apply(lambda row: lmir_dir_score(row['corrected_search'], row['brand'], params_brand_LM, 1.5 ), axis=1)
  data3['AD_SB'] = data3.apply(lambda row: lmir_abs_score(row['corrected_search'], row['brand'], 0.01 ), axis=1)
  data3['JM_SD'] = data3.apply(lambda row: lmir_jm_score(row['corrected_search'], row['description'], params_desc_LM, 0.9), axis=1)
  data3['Dir_SD'] = data3.apply(lambda row: lmir_dir_score(row['corrected_search'], row['description'], params_desc_LM, 106 ), axis=1)
  data3['AD_SD'] = data3.apply(lambda row: lmir_abs_score(row['corrected_search'], row['description'], 0.01 ), axis=1)

  data3['bm25_ST'] = data3.apply(lambda row: bm25_score(row['corrected_search'], row['title'], params_title_bm25 ), axis=1)
  data3['bm25_SD'] = data3.apply(lambda row: bm25_score(row['corrected_search'], row['description'], params_desc_bm25 ), axis=1)
  data3['bm25_SB'] = data3.apply(lambda row: bm25_score(row['corrected_search'], row['brand'], params_brand_bm25 ), axis=1)


  dictionary = tfidf_w2v_params_search['dictionary']
  tfidf_words = tfidf_w2v_params_search['tfidf_words']
  search_tfidf_w2v_test = []; # the avg-w2v for each sentence/review is stored in this list
  for sentence in cleaned_test_set2['corrected_search']: # for each review/sentence
      vector = np.zeros(300) # as word vectors are of zero length
      tf_idf_weight =0; # num of words with a valid vector in the sentence/review
      for word in sentence.split(): # for each word in a review/sentence
          if (word in glove_words) and (word in tfidf_words):
              vec = model[word] # getting the vector for each word
              # here we are multiplying idf value(dictionary[word]) and the tf value((sentence.count(word)/len(sentence.split())))
              tf_idf = dictionary[word]*(sentence.count(word)) # getting the tfidf value for each word
              vector += (vec * tf_idf) # calculating tfidf weighted w2v
              tf_idf_weight += tf_idf
      if tf_idf_weight != 0:
          vector /= tf_idf_weight
      search_tfidf_w2v_test.append(vector)



  dictionary = tfidf_w2v_params_title['dictionary']
  tfidf_words = tfidf_w2v_params_title['tfidf_words']
  title_tfidf_w2v_test = []; # the avg-w2v for each sentence/review is stored in this list
  for sentence in cleaned_test_set2['title']: # for each review/sentence
      vector = np.zeros(300) # as word vectors are of zero length
      tf_idf_weight =0; # num of words with a valid vector in the sentence/review
      for word in sentence.split(): # for each word in a review/sentence
          if (word in glove_words) and (word in tfidf_words):
              vec = model[word] # getting the vector for each word
              # here we are multiplying idf value(dictionary[word]) and the tf value((sentence.count(word)/len(sentence.split())))
              tf_idf = dictionary[word]*(sentence.count(word)) # getting the tfidf value for each word
              vector += (vec * tf_idf) # calculating tfidf weighted w2v
              tf_idf_weight += tf_idf
      if tf_idf_weight != 0:
             vector /= tf_idf_weight
      title_tfidf_w2v_test.append(vector)


  dictionary = tfidf_w2v_params_desc['dictionary']
  tfidf_words = tfidf_w2v_params_desc['tfidf_words']
  desc_tfidf_w2v_test = []; # the avg-w2v for each sentence/review is stored in this list
  for sentence in cleaned_test_set2['description']: # for each review/sentence
      vector = np.zeros(300) # as word vectors are of zero length
      tf_idf_weight =0; # num of words with a valid vector in the sentence/review
      for word in sentence.split(): # for each word in a review/sentence
          if (word in glove_words) and (word in tfidf_words):
              vec = model[word] # getting the vector for each word
              # here we are multiplying idf value(dictionary[word]) and the tf value((sentence.count(word)/len(sentence.split())))
              tf_idf = dictionary[word]*(sentence.count(word)) # getting the tfidf value for each word
              vector += (vec * tf_idf) # calculating tfidf weighted w2v
              tf_idf_weight += tf_idf
      if tf_idf_weight != 0:
          vector /= tf_idf_weight
      desc_tfidf_w2v_test.append(vector)

  arr1 = np.array(search_tfidf_w2v_test)
  arr2 = np.array(title_tfidf_w2v_test)
  arr3 = np.array(desc_tfidf_w2v_test)
  tfidf_w2v_df = pd.DataFrame(np.hstack((arr1, arr2, arr3)), index=cleaned_test_set2.index)

  idf_dict = F3_SmM_params_title['idf_dict']
  N = F3_SmM_params_title['N']
  max_tf = []
  max_idf = []
  max_tfidf = []
  min_tf = []
  min_idf = []
  min_tfidf = []
  sum_tf = []
  sum_idf = []
  sum_tfidf = []
  for ind, row in cleaned_test_set.iterrows():
    search = row['corrected_search']
    text = row['title']
    tf_vals = []
    idf_vals = []
    tfidf_vals = []
    for word in search.split():
      if word in idf_dict.keys():
        tf = text.count(word)
        idf = idf_dict[word]
      else:
        tf = text.count(word)
        idf = np.log(N+1)
      tf_vals.append(tf)
      idf_vals.append(idf)
      tfidf_vals.append(tf*idf)
    max_tf.append(max(tf_vals))
    min_tf.append(min(tf_vals))
    sum_tf.append(sum(tf_vals))
    max_idf.append(max(idf_vals))
    min_idf.append(min(idf_vals))
    sum_idf.append(sum(idf_vals))
    max_tfidf.append(max(tfidf_vals))
    min_tfidf.append(min(tfidf_vals))
    sum_tfidf.append(sum(tfidf_vals))

  data3['max_tf_ST'] = max_tf
  data3['max_idf_ST'] = max_idf
  data3['max_tfidf_ST'] = max_tfidf
  data3['min_tf_ST'] = min_tf
  data3['min_idf_ST'] = min_idf
  data3['min_tfidf_ST'] = min_tfidf
  data3['sum_tf_ST'] = sum_tf
  data3['sum_idf_ST'] = sum_idf
  data3['sum_tfidf_ST'] = sum_tfidf

  idf_dict = F3_SmM_params_brand['idf_dict']
  N = F3_SmM_params_brand['N']
  max_tf = []
  max_idf = []
  max_tfidf = []
  min_tf = []
  min_idf = []
  min_tfidf = []
  sum_tf = []
  sum_idf = []
  sum_tfidf = []
  for ind, row in cleaned_test_set.iterrows():
    search = row['corrected_search']
    text = row['brand']
    tf_vals = []
    idf_vals = []
    tfidf_vals = []
    for word in search.split():
      if word in idf_dict.keys():
        tf = text.count(word)
        idf = idf_dict[word]
      else:
        tf = text.count(word)
        idf = np.log(N+1)
      tf_vals.append(tf)
      idf_vals.append(idf)
      tfidf_vals.append(tf*idf)
    max_tf.append(max(tf_vals))
    min_tf.append(min(tf_vals))
    sum_tf.append(sum(tf_vals))
    max_idf.append(max(idf_vals))
    min_idf.append(min(idf_vals))
    sum_idf.append(sum(idf_vals))
    max_tfidf.append(max(tfidf_vals))
    min_tfidf.append(min(tfidf_vals))
    sum_tfidf.append(sum(tfidf_vals))

  data3['max_tf_SB'] = max_tf
  data3['max_idf_SB'] = max_idf
  data3['max_tfidf_SB'] = max_tfidf
  data3['min_tf_SB'] = min_tf
  data3['min_idf_SB'] = min_idf
  data3['min_tfidf_SB'] = min_tfidf
  data3['sum_tf_SB'] = sum_tf
  data3['sum_idf_SB'] = sum_idf
  data3['sum_tfidf_SB'] = sum_tfidf

  idf_dict = F3_SmM_params_desc['idf_dict']
  N = F3_SmM_params_desc['N']
  max_tf = []
  max_idf = []
  max_tfidf = []
  min_tf = []
  min_idf = []
  min_tfidf = []
  sum_tf = []
  sum_idf = []
  sum_tfidf = []
  for ind, row in cleaned_test_set.iterrows():
    search = row['corrected_search']
    text = row['description']
    tf_vals = []
    idf_vals = []
    tfidf_vals = []
    for word in search.split():
      if word in idf_dict.keys():
        tf = text.count(word)
        idf = idf_dict[word]
      else:
        tf = text.count(word)
        idf = np.log(N+1)
      tf_vals.append(tf)
      idf_vals.append(idf)
      tfidf_vals.append(tf*idf)
    max_tf.append(max(tf_vals))
    min_tf.append(min(tf_vals))
    sum_tf.append(sum(tf_vals))
    max_idf.append(max(idf_vals))
    min_idf.append(min(idf_vals))
    sum_idf.append(sum(idf_vals))
    max_tfidf.append(max(tfidf_vals))
    min_tfidf.append(min(tfidf_vals))
    sum_tfidf.append(sum(tfidf_vals))

  data3['max_tf_SD'] = max_tf
  data3['max_idf_SD'] = max_idf
  data3['max_tfidf_SD'] = max_tfidf
  data3['min_tf_SD'] = min_tf
  data3['min_idf_SD'] = min_idf
  data3['min_tfidf_SD'] = min_tfidf
  data3['sum_tf_SD'] = sum_tf
  data3['sum_idf_SD'] = sum_idf
  data3['sum_tfidf_SD'] = sum_tfidf

  data3 = data3.iloc[:,6:]

  #MODELING
  X1 = pd.concat([data1[feat_set1_comb], data2, data3], axis=1)
  pred_xgb = F1_xgb.predict(X1)
  pred_rf = F1_rf.predict(X1)
  pred_ridge = F1_ridge.predict(F1_scaler_ridge.transform(X1))
  pred_lasso = F1_lasso.predict(F1_scaler_lasso.transform(X1))
  pred_en = F1_en.predict(F1_scaler_en.transform(X1))
  pred_svr = F1_svr.predict(F1_scaler_svr.transform(X1))
  pred_dt = F1_dt.predict(X1)
  arr = np.hstack((pred_xgb.reshape(-1,1),
                  pred_rf.reshape(-1,1),
                  pred_dt.reshape(-1,1),
                  pred_svr.reshape(-1,1),
                  pred_ridge.reshape(-1,1),
                  pred_lasso.reshape(-1,1),
                  pred_en.reshape(-1,1)))
  F1_df = pd.DataFrame(arr, columns=['f1_xgb', 'f1_rf', 'f1_dt', 'f1_svr', 'f1_ridge', 'f1_lasso', 'f1_en'], index=X1.index)

  X2 = pd.concat([data1[feat_set1_comb], data2, data3, tfidf_w2v_df], axis=1)
  pred_svr = F2_svr.predict(F2_scaler_svr.transform(X2))
  pred_rf = F2_rf.predict(X2)
  pred_ridge = F2_ridge.predict(F2_scaler_ridge.transform(X2))
  pred_lasso = F2_lasso.predict(F2_scaler_lasso.transform(X2))
  pred_en = F2_en.predict(F2_scaler_en.transform(X2))
  pred_dt = F2_dt.predict(X2)
  arr = np.hstack((pred_svr.reshape(-1,1),
                  pred_rf.reshape(-1,1),
                  pred_ridge.reshape(-1,1),
                  pred_lasso.reshape(-1,1),
                  pred_en.reshape(-1,1),
                  pred_dt.reshape(-1,1),))
  F2_df = pd.DataFrame(arr, columns=['f2_svr', 'f2_rf', 'f2_ridge', 'f2_lasso', 'f2_en', 'f2_dt'], index=X2.index)

  X3 = pd.concat([data1[feat_set1_comb], data2, data3, truncated_df], axis=1)
  pred_dt = F3_dt.predict(X3)
  pred_ridge = F3_ridge.predict(F3_scaler_ridge.transform(X3))
  pred_lasso = F3_lasso.predict(F3_scaler_lasso.transform(X3))
  pred_en = F3_en.predict(F3_scaler_en.transform(X3))
  arr = np.hstack((pred_dt.reshape(-1,1),
                  pred_ridge.reshape(-1,1),
                  pred_lasso.reshape(-1,1),
                  pred_en.reshape(-1,1)))
  F3_df = pd.DataFrame(arr, columns=['f3_dt', 'f3_ridge', 'f3_lasso', 'f3_en'], index=X3.index)

  #FINAL
  test_x = pd.concat([F1_df, F2_df, F3_df], axis=1)
  test_x_std = final_scaler.transform(test_x)
  test_y_pred = final_ridge.predict(test_x_std)

  return test_y_pred

def get_candidates(search, N):
  search = preprocessing_search(search)
  search = corrected_term(search)
  tokenized_query = search.split(" ")
  candidates = bm25_model.get_top_n(tokenized_query, product_text, n=N)
  candidate_products = database[database['text'].isin(candidates)].drop('text', axis=1)
  candidate_products['search_term'] = search
  reorder_cols = ['product_uid', 'product_title', 'search_term', 'combined_attr', 'brand', 'product_description']
  return candidate_products[reorder_cols]

def main(srch, n):
  test_set = get_candidates(srch, 100)
  test_set['relevance'] = get_search_relevance(test_set)
  return test_set.sort_values('relevance', ascending=False).iloc[:n]['product_title']

print('function definitions done!')
###############################################################################################
###############################################################################################

@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    search = request.form.to_dict()['search_query']
    results = main(search, 10)
    params = {'results':results, 'search':search}
    return flask.render_template('predict.html', params=params)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
