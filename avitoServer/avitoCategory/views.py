from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import os
import preproc
from ufal.udpipe import Model as Mod, Pipeline
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import zipfile
from gensim import models
import torch
import torch.nn as nn
import numpy as np
from bs4 import BeautifulSoup as zbs
import requests


class ModelNN(nn.Module):

    def __init__(self, num_numerical_cols, output_size, layers, p=0.4):
        super().__init__()
        # self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size])
        # self.embedding_dropout = nn.Dropout(p)
        self.batch_norm_num = nn.BatchNorm1d(num_numerical_cols)

        all_layers = []
        # num_categorical_cols = sum((nf for ni, nf in embedding_size))
        input_size = num_numerical_cols

        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, x_numerical):
        # embeddings = []
        # for i,e in enumerate(self.all_embeddings):
        #    embeddings.append(e(x_categorical[:,i]))
        # x = torch.cat(embeddings, 1)
        # x = self.embedding_dropout(x)

        # print(x_numerical)

        x_numerical = self.batch_norm_num(x_numerical)
        # print(x_numerical)
        x = x_numerical
        # print(x)
        x = self.layers(x)
        return x


udpipe_filename = '/Users/tikhon/Desktop/ml avito/udpipe_syntagrus.model'
modell = Mod.load(udpipe_filename)
process_pipeline = Pipeline(modell, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')

model_file = '/Users/tikhon/Desktop/CHOK/prog-Y/data/182.zip'
with zipfile.ZipFile(model_file, 'r') as archive:
    stream = archive.open('model.bin')
    model = models.KeyedVectors.load_word2vec_format(stream, binary=True)
w2v = dict(zip(model.wv.index2word, model.wv.syn0))

nn_file = '/Users/tikhon/Desktop/ml avito/last_model'
model1 = ModelNN(601, 54, [328, 328], p=0.4)
model1.load_state_dict(torch.load(nn_file))
model1.eval()

cats_path = '/Users/tikhon/Desktop/ml avito/avito_category.csv'
cat_descr = pd.read_csv(cats_path)
dat = (dict(zip(cat_descr.category_id, cat_descr.name)))


def proctext(text):
    return preproc.process(process_pipeline, text)


def main(request):
    if request.method == 'GET':
        return render(request, 'main.html')
    if request.method == 'POST':
        link = request.POST['link']
        if link != '':
            title, description, price, category = parser(link)
            try:
                price = float(price)
            except ValueError:
                price = 0.0
            classs = getPredict(title, description, price)
            objcts = getCategory(classs)
            return render(request, 'response.html',
                          {'title': title, 'text': description, 'price': int(price), 'catrs': objcts, 'islink': True,
                           'truecatrs': category})
        else:
            title = request.POST['title']
            description = request.POST['description']
            price = float(request.POST['price'])
            classs = getPredict(title, description, price)
            objcts = getCategory(classs)
            return render(request, 'response.html',
                          {'title': title, 'text': description, 'price': price, 'catrs': objcts, 'islink': False})


def parser(link):
    soup = zbs(requests.get(link).text, 'html.parser')
    title = soup.find('span', {'class': 'title-info-title-text'}).text
    description = []
    for i in soup.find('div', {'class': 'item-description-text'}).find_all('p'):
        description.append(i.text)
    description = "\n".join(description)
    price = soup.find('span', {'class': 'js-item-price'})
    if price == None:
        price = soup.find('span', {'class': 'price-value-string js-price-value-string'}).text
    else:
        price = price['content']
    category = []
    for i in soup.find('div', {'class': 'breadcrumbs js-breadcrumbs'}).find_all('a', {
        'class': 'js-breadcrumbs-link js-breadcrumbs-link-interaction'}):
        category.append(i['title'])
    return title, description, price, category


def preprocess(review):
    stops = set(stopwords.words("english")) | set(stopwords.words("russian"))
    words = review.split()
    words = [w for w in words if (not w in stops) and (len(list(w)) > 2)]
    words = proctext(" ".join(words))
    return " ".join(words)


class tfidf_vectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(next(iter(w2v.values())))

    def fit(self, X):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] * self.word2weight[w]
                     for w in words.split() if w in self.word2vec] or
                    [np.zeros(self.dim)], axis=0)
            for words in X
        ])


def getPredict(title, description, price):
    data = pd.DataFrame([[title, description, price]], columns=['title', 'description', 'price'])
    data['title_processed'] = data['title'].apply(preprocess)
    data['description_processed'] = data['description'].apply(preprocess)
    title_counted = tfidf_vectorizer(w2v).fit(data['title_processed']).transform(data['title_processed'])
    description_counted = tfidf_vectorizer(w2v).fit(data['description_processed']).transform(
        data['description_processed'])
    price = data.drop(['title', 'description', 'title_processed', 'description_processed'], axis=1)
    out_data = pd.concat([pd.DataFrame(title_counted),
                          pd.DataFrame(description_counted, columns=[i for i in range(300, 600)]), price],
                         axis=1)
    out_data.astype('float64')
    out_data = np.stack([out_data[col].values for col in out_data.columns], 1)
    out_data = torch.tensor(out_data.astype('float64'), dtype=torch.float)
    out_data[out_data == 0] = 0.01
    with torch.no_grad():
        y_val = model1(out_data)
    y_val = np.argmax(y_val, axis=1)
    return y_val[0].item()


def getCategory(cl):
    cat_des = dat[cl]
    return cat_des.split(sep='|')
