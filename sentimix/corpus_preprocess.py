import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import partial
import csv

import taggin as tg
from parallelize import parallelize_df, parallelize_df_np


def HorribleConLL2df(horrible_data_file):
        """
        Takes "supposed conll file and returns a dataframe"

        Parameters:
        horrible_data_file: Supossed conll file

        Returns:
        pd.Dataframe: Dataframe whith list of tokens and languages
        """

        dataframe = pd.read_csv(horrible_data_file,
                                encoding="utf_8",
                                sep=r"[\s]",
                                header=None,
                                names=["words","lang","sentiment"])
        lastrow  = len(dataframe)-1
        indexes = dataframe["lang"].apply(str.isdigit)
        index_list = indexes[indexes].index.tolist()
        index_list.append(lastrow)
        phrase_df = pd.DataFrame(columns=["tweetid",
                                          "tweet",
                                          "lang",
                                          "length",
                                          "sentiment"])
        phrase_df = phrase_df.set_index('tweetid')
        for counter, limit in enumerate(index_list):
                if counter < len(index_list)-1:
                        tweetid = dataframe.iloc[limit]["lang"]
                        sentiment = dataframe.iloc[limit]["sentiment"].strip()
                        tokenlist = [x for x in dataframe.iloc[limit+1:index_list[counter+1]]["words"]]
                        metalist = [x for x in dataframe.iloc[limit+1:index_list[counter+1]]["lang"]]
                        print(tokenlist)
                phrase_df.loc[int(tweetid)] = [tokenlist, metalist, len(tokenlist),sentiment]
        return phrase_df

def lower_corpus(corpus):
        """
        Takes a corpus DataFrame whith one colum called "tweet" and
        returns the same dataframe whit each token of the "tweet" colum
        in lower case.
        """
        lower_cor = corpus
        corpus = None
        lower_cor["tweet"] =  lower_cor["tweet"].apply(lambda x: " ".join(x).lower().split())
        return lower_cor

def tagged_corpus(corpus):
        """
        Takes a corpus DataFrame whith one colum called "tweet" and
        returns the same dataframe whit each token of the "tweet" colum
        in lower case.
        """
        tagged_cor = corpus
        corpus = None
        tagged_cor["tweet"] =  tagged_cor["tweet"].apply(lambda x: tg.taggin_prices(" ".join(x)).split())
        tagged_cor["tweet"] =  tagged_cor["tweet"].apply(lambda x: tg.taggin_numbers(" ".join(x)).split())
        tagged_cor["tweet"] =  tagged_cor["tweet"].apply(lambda x: tg.internet_element_taggin(" ".join(x)).split())
        return tagged_cor



def sentimix_vocab(corpus):
        """
        Takes a tokenized corpus and returns a vocabulary with frecuences

        Parameters:
        corpus (pd.Series): A series whith ordered list of token in each row

        Returns:
        pd.Dataframe:
        """
        vocab = []
        for sentence in corpus.items():
                sent = list(map(lambda x: internet_element_taggin(x), sentence[1]))
                sent = list(map(lambda x: taggin_numbers(x), sentence))
                vocab = vocab + sent
        return vocab

def sentimix_freqdf(corpus, normalize="Off"):
        total_vocab = sentimix_vocab(corpus)
        positive_vocab = sentimix_vocab(corpus[corpus['sentiment']=='positive'])
        negative_vocab = sentimix_vocab(corpus[corpus['sentiment']=='negative'])
        neutral_vocab = sentimix_vocab(corpus[corpus['sentiment']=='neutral'])
        if normalize == "lower":
                total_vocab = list(map(lambda x: x.lower(), total_vocab))
                positive_vocab = list(map(lambda x: x.lower(), positive_vocab))
                negative_vocab = list(map(lambda x: x.lower(), negative_vocab))
                neutral_vocab = list(map(lambda x: x.lower(), neutral_vocab))
        set_total_vocab = set(total_vocab)
        set_positive_vocab = set(positive_vocab)
        set_negative_vocab = set(negative_vocab)
        set_neutral_vocab = set(neutral_vocab)
        freqdf = pd.DataFrame(columns=["widx",
                                       "word",
                                       "pos_freq",
                                       "neg_freq",
                                       "neu_freq",
                                       "freq",
                                       "total_idf",
                                       "pos_tdf",
                                       "neg_tdf",
                                       "neu_tdf"])
        freqdf = freqdf.set_index(['widx'])
        for counter, word in enumerate(set_total_vocab):
                freqdf.loc[counter] = [word,
                                       positive_vocab.count(word),
                                       negative_vocab.count(word),
                                       neutral_vocab.count(word),
                                       total_vocab.count(word)]
        max_pos_freq = freqdf.pos_freq.max()
        max_neg_freq = freqdf.neg_freq.max()
        max_neu_freq = freqdf.neu_freq.max()
        max_freq = freqdf.freq.max()

        return freqdf


def corpus2tsv(dataframe, path, filename):
        """
        Takes an txt JSON object per line file and make a tsv file with the same data.

        Parameters:
        dataframe (pd.DataFrame): Dataframe to dump into tsv file
        path (str): path to file
        filename(str): filename
        """

        dataframe.to_csv(path+filename, encoding="utf-8", sep='\t')

def sentimixInternetTagging(dataframe):
        """
        Takes sentimix dataframe and return a dataframe whith tagg LINKS, RT's, AT's, HASTAGS,

        Parameters:
	dataframe (pd.DataFrame): Dataframe to dump into tsv file

        Returns:
        pd.DataFrame: Tagged texts
        """

        dataframe["tweet"] = dataframe["tweet"].apply(internet_element_taggin)
        dataframe["tweet"] = dataframe["tweet"].apply(taggin_numbers)
        print (dataframe["tweet"])

def get_position(ls, element):
        positions = [i for i, x in enumerate(ls) if x == element]
        """if positions:
                print("--------------------------------")
                print(element)
                print(ls)
                print(positions)
                print("-------------------------------\n\n")"""
        return positions

def sintagmatrix(corpus, vocab):
        sintag_mat = pd.DataFrame(columns = ["sidx"]+vocab)
        sintag_mat = sintag_mat.set_index("sidx")
        for sentence in corpus.iterrows():
                print(sentence)
                row = [get_position(sentence[1].tweet,x) for x in list(sintag_mat.columns)]
                new_row = [sentence[0]] + row
                with open("data/sintagmatrix.tsv", 'a', encoding="utf-8", newline='') as f:
                        writer = csv.writer(f,  delimiter='\t')
                        writer.writerow(new_row)
                f.close()


def parallelize_sintagmatrix(df, secondarg):
        sintag_mat = pd.DataFrame(columns = ["sidx"]+vocab)
        sintag_mat = sintag_mat.set_index("sidx")
        with open("data/sintagmatrix.tsv", 'w') as f:
                sintag_mat.to_csv(f, encoding="utf-8", sep='\t')
        df_split = np.array_split(df, 100)
        pool = mp.Pool(mp.cpu_count()-1)
        new_func = partial(sintagmatrix, vocab = secondarg)
        df = pd.concat(pool.map(new_func,df_split))
        pool.close()
        pool.join()
