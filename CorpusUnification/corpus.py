import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby


class Document():

    def __init__(self, id, name, text, category=None):
        """
        Parameters:
        uid (int): Unique Document Index.
        name (str): Name od the document.
        text (list(list(str))): List of sentences each sentece is a list of orered tokens.
        category (str): Category of the document
        """
        self.id = id
        self.name = name
        self.text = text
        self.cate = category


    def ngap_gram(self, gaps=[0]):
        """
        Extract ngrams frecuencies whit gaps between each n-gram element.
        the gaps length defines de n and de value of each is the gaps
        between them. A gaps = [0,1] is a traditional bigram, takes the element next to the other
        whitout gaps. A gaps = [0,1,2] is a trigram, gaps=[0,2,4] is a trigram whith a gap of one
        element between them.
        """
        doc_n_grams = []
        for sentence in self.text:
            if len(gaps) > 1:
                sentence.insert(0, "<-sentence_begin!->")
            sentence_n_grams = []
            lambda_finish = lambda x, y : sentence[x+y] if x+y < len(sentence) else("<-sentence_end!->" if x+y==len(sentence) else "<-beyond_limits!->")
            for i in range (0, len(sentence)):
                sentence_n_grams.append(" ".join([lambda_finish(gap,i) for gap in gaps]))
            doc_n_grams += sentence_n_grams
        doc_n_grams.sort(key=str.lower)
        bow = {x:len(list(y)) for x, y in groupby(doc_n_grams)}
        return bow

    def tf_idf(self, gaps=[0]):
        bow_dataframe = pd.DataFrame.from_dict(self.ngap_gram(gaps), orient="index", columns=["RAW_COUNT"]).sort_values(by="RAW_COUNT")
        bow_dataframe["BIN"] = bow_dataframe["RAW_COUNT"]>0
        bow_dataframe["TF_MEAN"] = bow_dataframe["RAW_COUNT"]/bow_dataframe["RAW_COUNT"].sum()
        bow_dataframe["LOG_NOR"] = np.log(bow_dataframe["RAW_COUNT"])
        bow_dataframe["DN04"] = 0.2+(bow_dataframe["RAW_COUNT"]/bow_dataframe["RAW_COUNT"].max())*0.8
        bow_dataframe["DN04"] = 0.4+(bow_dataframe["RAW_COUNT"]/bow_dataframe["RAW_COUNT"].max())*0.6
        bow_dataframe["DN05"] = 0.5+(bow_dataframe["RAW_COUNT"]/bow_dataframe["RAW_COUNT"].max())/2
        bow_dataframe["DN06"] = 0.6+(bow_dataframe["RAW_COUNT"]/bow_dataframe["RAW_COUNT"].max())*0.4
        bow_dataframe["DN08"] = 0.8+(bow_dataframe["RAW_COUNT"]/bow_dataframe["RAW_COUNT"].max())*0.2
        # Nota: Normalizar rangos solo al comparar
        bow_dataframe["AVE_RANK"] = bow_dataframe["RAW_COUNT"].rank(method="average", ascending=False)
        bow_dataframe["MIN_RANK"] = bow_dataframe["RAW_COUNT"].rank(method="min", ascending=False)
        bow_dataframe["MAX_RANK"] = bow_dataframe["RAW_COUNT"].rank(method="max", ascending=False)
        bow_dataframe["FIR_RANK"] = bow_dataframe["RAW_COUNT"].rank(method="first", ascending=False)
        bow_dataframe["DEN_RANK"] = bow_dataframe["RAW_COUNT"].rank(method="dense", ascending=False)
        bow_dataframe = bow_dataframe.sort_values(by="FIR_RANK")
        #parametros
        zipf_k = np.log(bow_dataframe["RAW_COUNT"].max())
        zipf_a = zipf_k/np.log(bow_dataframe["DEN_RANK"].max())
        #Zipf's values
        bow_dataframe["IdealZipF"] = -zipf_a*np.log(bow_dataframe["DEN_RANK"])+zipf_k
        bow_dataframe["ERR"] = bow_dataframe["LOG_NOR"] - bow_dataframe["IdealZipF"]
        bow_dataframe["ERR2"] = np.sqrt(bow_dataframe["ERR"])
        self.err2 = bow_dataframe["ERR2"].mean()
        return bow_dataframe

    def zipflaw_graph(self,gapz=[0]):
        dataset = self.tf_idf(gaps=gapz)[["LOG_NOR","IdealZipF","ERR","ERR2","DEN_RANK"]]
        x = np.log(np.array(dataset["DEN_RANK"]))
        y = np.array(dataset["LOG_NOR"])
        ideal = np.array(dataset["IdealZipF"])
        err = np.array(dataset["ERR"])
        err2= np.array(dataset["ERR2"])
        plt.plot(x,y)
        plt.plot(x,ideal)
        plt.plot(x,err)
        plt.plot(x,err2)
        plt.ylabel(self.err2)
        plt.show()

class Collection():
    name = ""
    document_list = pd.DataFrame(columns=["Document"])

    def __init__(self, name):
        pass
