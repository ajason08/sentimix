import pandas as pd
import numpy as np
from time import sleep
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

    def tf_idf(self, power, gaps=[0]):
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

        bow_dataframe["RANK_COUNT"] = bow_dataframe.groupby('DEN_RANK')['DEN_RANK'].transform('count')

        #parametros
        zipf_k = bow_dataframe["LOG_NOR"].max()
        zipf_a = zipf_k/np.log(bow_dataframe["DEN_RANK"].max())
        #Zipf's values
        bow_dataframe["IdealZipF"] = -zipf_a*np.log(bow_dataframe["DEN_RANK"])+zipf_k
        #power Zipf
        zipf_power_k =  np.power(bow_dataframe["LOG_NOR"].max(),1/power)
        zipf_power_a =  zipf_power_k/np.log(bow_dataframe["DEN_RANK"].max())
        bow_dataframe["PowerZipf"] =  np.power(-zipf_power_a*np.log(bow_dataframe["DEN_RANK"])+zipf_power_k, power)
        bow_dataframe["ERR"] = bow_dataframe["LOG_NOR"] - bow_dataframe["IdealZipF"]
        bow_dataframe["POWER_ERR"] =  bow_dataframe["LOG_NOR"] - bow_dataframe["PowerZipf"]
        self.quad_err = np.square(bow_dataframe["ERR"]).mean()
        self.quad_power_err = np.square(bow_dataframe).mean()
        return bow_dataframe

    def rank_of_rank_stabilization(self, power=1, gapz=[0]):
        frecuency = self.tf_idf(power, gaps=gapz)[["RAW_COUNT", "DEN_RANK"]]
        """frecuency = pd.DataFrame.from_dict({"RAW_COUNT":[200,200,11,11,11,4,3,2,2,2,1],
                                            "DEN_RANK":[1,1,2,2,2,3,4,5,5,5,6]})
        """
        print("--sin tocar----")
        print(frecuency)
        lengths = [len(frecuency)]
        while len(frecuency) > 1:
            frecuency = frecuency.groupby("DEN_RANK").count()
            print("-------Contó---")
            print(frecuency)
            frecuency["DEN_RANK"] = frecuency["RAW_COUNT"].rank(method="dense", ascending=False)
            print("-------Rankeo--")
            print(frecuency)
            frecuency.index.name="Index"
            print("-----unifico----")
            print(frecuency)
            lengths.append(len(frecuency))
            print(lengths)
        n  = len(lengths)
        armonico = 0
        for x in range(1,n+1):
            armonico +=1/x
        print(n,armonico,lengths)
        return



    """def zipflaw_graph(self, power, ranklevel,dim=3 ,gapz=[0]):


        dataset = self.tf_idf(power, gaps=gapz)
        x = np.log(np.array(dataset["DEN_RANK"]))
        z_real = np.array(dataset["LOG_NOR"])
        z_ideal = np.array(dataset["IdealZipF"])
        z_sqrt = np.array(dataset["PowerZipf"])
        z_err2= np.array(dataset["ERR"])
        y = np.log(np.array(dataset[ranklevel]))


        if dim==3:
            mpl.rcParams['legend.fontsize'] = 10
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot(x, y, z_real, label='Real')
            ax.plot(x, y, z_ideal, label="IdealZipF")
            ax.plot(x, y, z_sqrt, label="Error")
            ax.legend()
            plt.show()

        else:
            plt.plot(x,z_real)
            plt.plot(x,z_ideal)
            plt.plot(x,z_sqrt)
            #plt.plot(x,z_err2, label="kk2")
            plt.title("Zipf for {}\nStructure: {}\nZipfError{}\nPowerZipfErr:{}".format(self.name,
                                                                   gapz,
                                                                   self.quad_err,
                                                                   self.quad_power_err))
            plt.xlabel("log(Rank)")
            plt.ylabel("log(Freq)")
            plt.show()

        dataset = self.tf_idf(gaps=gapz)[["LOG_NOR","IdealZipF","ERR","ERR2","DEN_RANK"]]
        x = np.log(np.array(dataset["DEN_RANK"]))
        y = np.array(dataset["LOG_NOR"])


        """

class Collection():
    name = ""
    document_list = pd.DataFrame(columns=["Document"])

    def __init__(self, name):
        pass
