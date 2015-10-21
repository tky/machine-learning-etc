#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 準備
# 下記サイトにあるlivedoor ニュースコーパスから（通常テキスト）をダウンロードしてこのファイルと同じ場所におく。
# http://www.rondhuit.com/download.html#ldcc
# 
# こちらがダウンロード先リンク。
# http://www.rondhuit.com/download/ldcc-20140209.tar.gz
#
# TODO:
# 構文解析の辞書登録  サッカー　選手は多分サッカー選手という一つの単語
#
# stop word + 解析した単語の長さでフィルタ。 こと、の、ん、といったよく分からんキーワードが出てくる。
# ひらがな１文字だったらノイズのように見える。
#
# 出力例
#0.036*こと + 0.031*の + 0.016*ん + 0.015*選手 + 0.012*野球 + 0.011*戦 + 0.008*プロ + 0.008*ファン + 0.008*サッカー + 0.007*自分
#0.038*こと + 0.021*の + 0.018*ん + 0.016*戦 + 0.015*サッカー + 0.014*選手 + 0.011*チーム + 0.008*自分 + 0.007*女子 + 0.007*これ
#0.023*こと + 0.015*の + 0.014*五輪 + 0.014*ん + 0.012*選手 + 0.012*自分 + 0.010*ロンドン + 0.006*女子 + 0.006*芸能 + 0.006*チーム
#0.021*こと + 0.020*ん + 0.019*の + 0.015*五輪 + 0.013*世界 + 0.012*選手 + 0.009*大会 + 0.008*女子 + 0.008*月 + 0.007*選手権
#0.024*の + 0.020*ん + 0.018*こと + 0.011*自分 + 0.011*選手 + 0.007*番組 + 0.007*もの + 0.007*サッカー + 0.007*人 + 0.006*ファン

import os
from janome.tokenizer import Tokenizer
from gensim import corpora, models, similarities
import codecs

def lists(dir):
    """
        list all article file except for a LICENSE.txt
    """
    for f in os.listdir(dir):
        if not f.startswith('LICENSE'):
            yield f

def read_article(dir, file):
    with codecs.open(os.path.join(dir, file), encoding='utf-8') as f:
        data = f.readlines()
        return Article(data[0], data[1], data[2], "".join(data[3:]))

def load_articles(dir):
    for f in lists(dir):
        yield read_article(dir, f)

class Article:
    def __init__(self, url, published_timestamp, title, body):
        self._title = title
        self._published_timestamp = published_timestamp
        self._url = url
        self._body = body

    @property
    def published_timestamp(self):
        return self._published_timestamp

    @property
    def title(self):
        return self._title

    @property
    def body(self):
        return self._body

    @property
    def url(self):
        return self._url



def is_general_noun(token):
    """
        return true if token is general(一般) noun(名詞)
    """
    part_of_speech = token.part_of_speech.split(",")
    return u"名詞" in part_of_speech and u"一般" in part_of_speech

t = Tokenizer()

def to_tokens(text):
    return [token.surface for token in t.tokenize(text) if is_general_noun(token)]


## main ##
# 落としてきたディレクトリを指定。（今回はsports-watchを選択)
dir = 'text/sports-watch/'
analyzed_tokens = [to_tokens(article.body) for article in load_articles(dir)]

# make corpus
dictionary = corpora.Dictionary(analyzed_tokens)
corpus = [dictionary.doc2bow(token) for token in analyzed_tokens]

lda = models.ldamodel.LdaModel(corpus=corpus, num_topics=20, id2word=dictionary)
# この辺りを見れば他の使い方も分かるか。。
# http://radimrehurek.com/gensim/models/ldamodel.html
for topic in lda.show_topics(-1):
    print topic
