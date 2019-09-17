import pandas as pd
import numpy as np
import multiprocessing as mp
from ast import literal_eval
from functools import partial
import corpus_preprocess as cp
import taggin as tg

SPANGLISH = "spanglish/"
HINDINGLISH = "hindinglish/"
TRIAL = "trial_conll.txt"
TRAIN = "train_conll.txt"


CORPUS = "corpus.tsv"
FREQ = "freqs.tsv"
SINTAGMATRIX = "sintagmatrix.tsv"

LOW_CORPUS = "low_corpus.tsv"
LOW_FREQ = "low_freq.tsv"
LOW_SINTAGMATRIX = "low_sintagmatrix.tsv"

TAGGED_TW_CORPUS = "tagged_tw_corpus.tsv"
TAGGED_TW_FREQ = "tagged_tw_freq.tsv"
TAGGED_TW_SINTAGMATRIX = "tagged_tw_sintagmatrix.tsv"

TAGGED_TW_LOW_CORPUS = "low_tagged_tw_corpus.tsv"
TAGGED_TW_LOW_FREQ = "low_tagged_tw_freq.tsv"
TAGGED_TW_LOW_SINTAGMATRIX = "low_tagged_tw_sintagmatrix.tsv"




corpus = pd.read_csv(SPANGLISH+CORPUS,
                     sep="\t",
                     encoding="utf-8",
                     index_col="tweetid",
                     converters={"tweet":literal_eval})










"""
tg_corp = cp.tagged_corpus(corpus)
cp.corpus2tsv(tg_corp,SPANGLISH,TAGGED_TW_CORPUS)

lw_corp = cp.lower_corpus(corpus)
cp.corpus2tsv(lw_corp,SPANGLISH,LOW_CORPUS)

tg_lw_corp = cp.tagged_corpus(lw_corp)
cp.corpus2tsv(tg_lw_corp,SPANGLISH,TAGGED_TW_LOW_CORPUS)


corpus = pd.read_csv("data/corpus.tsv",
                     sep="\t",
                     encoding="utf-8",
                     index_col="tweetid",
                     converters={"tweet":literal_eval})




#corpus = HorribleConLL2df(TRAIN)
#corpus2tsv(corpus, "data/", "corpus.tsv")
#freqdf = parallelize_df(corpus, 60, sentimix_freqdf).groupby(['word']).sum()

freqdf =  pd.read_csv("data/freq.tsv",
                     sep="\t",
                     encoding="utf-8")
vocab = list(freqdf.word)
#print(vocab)
#print(freqdf.head())
#corpus2tsv(freqdf, "data/", "freq.tsv")

parallelize_sintagmatrix(corpus, vocab)


#corpus2tsv(sintagmatrix(corpus, freqdf), "data/", "sintagmatrix.tsv")
"""
