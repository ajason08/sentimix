import pandas as pd
import numpy as np
from scipy.special import digamma
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def harmonic(x):
    """
    Returns the armonic number of x
    If s is complex the result becomes complex.
    """
    return digamma(x + 1) + np.euler_gamma

def harmonic_g1(c,n,m):
    """
    Return the armonic generalized  number
    for parameters by a for cicle

    Parameters:
    c (int): add constant
    n (int): sumatory count limit
    m (float): add power
    """
    harm = 0
    for k in range (1,n):
        oldharm = harm
        suma = c+k
        potencia = suma**m
        cociente = 1/potencia
        harm += cociente
        if format(oldharm, '.16g') == format(harm, '.16g'):
            return harm
    return harm

def harmonic_g2(c,n,m):
    return harmonic_g1(0,c+n,m)-harmonic_g1(0,c+1,m)

def euler_aprox(x):
    e = 0
    for i in range(0, int(x)):
        e += 1/np.math.factorial(i)
    return e


def zipf_type_of_rank_comparision_graph(zipf,
                                        log = True,
                                        list_of_types = ["AvgRank",
                                                         "DenRank",
                                                         "MinRank",
                                                         "MaxRank",
                                                         "FirRank"],
                                        comparision_list=["Error", "Types", "Tokens"]):
    data = zipf.freq_rank
    zipf_model = zipf.zipf
    number_of_types = len(list_of_types)

    number_of_comparitions = len(comparision_list)

    if log == False:
        zipf_model = np.exp(zipf_model)
        info = "Freq vs Rank"
        x_label = "Rank"
        zipf_ylabel = "Freq"
    elif log == "Freq":
        data["Freq"] = np.log(data["Freq"])
        info = "ln(Freq) vs Rank"
        x_label = "Rank"
        zipf_ylabel = "ln(Freq)"
    elif log == "Rank":
        data[list_of_types] = np.log(data[list_of_types])
        zipf_model = np.exp(zipf_model)
        print(zipf_model)
        info = "Freq vs ln(Rank)"
        x_label = "ln(Rank)"
        zipf_ylabel = "Freq"
    else:
        data = np.log(data)
        info = "ln(Freq) vs ln(Rank)"
        x_label = "ln(Rank)"
        zipf_ylabel = "ln(Freq)"

    y = data["Freq"]

    y_lables = [zipf_ylabel]

    if "Error" in comparision_list:
        y_lables.append("Square error")
    if "Types" in comparision_list:
        y_lables.append("Types per rank")
    if "Tokens" in comparision_list:
        y_lables.append("Tokens per rank")


    # Set the graph
    fig, axs = plt.subplots(number_of_comparitions+1, number_of_types, sharex="col", sharey="row")
    fig.suptitle("Zipf law in diferents ranks\n"+info)

    for i in range (0, number_of_types):
        x = data[list_of_types[i]]
        comp = []
        zipf_y = zipf_model[list_of_types[i]]
        axs[0,i].plot(x,y)
        axs[0,i].plot(x,zipf_y)
        axs[0,i].set_title("{}".format(list_of_types[i]))
        if "Error" in comparision_list:
            error = np.array(zipf_model["SqrEr{}".format(list_of_types[i])])
            comp.append(error)
        if "Types" in comparision_list:
            types = np.array(zipf_model["SqrEr{}".format(list_of_types[i])])
            comp.append(types)
        if "Tokens" in comparision_list:
            tokens =np.array(zipf_model["SqrEr{}".format(list_of_types[i])])
            comp.append(tokens)
        for count, element in enumerate(comp,1):
            axs[count,i].plot(x,element)

    for ax in axs.flat:
        ax.set(xlabel=x_label)

    for ax, row in zip(axs[:,0], y_lables):
        ax.set_ylabel(row)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.show()






class zipf():
    freq_rank = pd.DataFrame()
    zipf_k = {}
    zipf_alpha = {}
    #subranks_level = {"Avg":0,"Den":0, "Min":0, "Max":, "Fir":0}
    errors = {}

    def __init__(self, bow):
        """
        Takes a BOW of words of a text and build his Zipfian estructure.

        Params:
        bow: A BOW of text in dict format which each token as keyword
        """

        # Takes  the BOW dict and transform it into pd_data
        self.freq_rank["Freq"] =  pd.Series(bow)
        # Order the frecuencies from high to low
        self.freq_rank.sort_values(by="Freq", ascending = False)
        # Dense rank frecuencies from high to low
        self.freq_rank["AvgRank"] = self.freq_rank["Freq"].rank(method="average", ascending=False)
        self.freq_rank["DenRank"] = self.freq_rank["Freq"].rank(method="dense", ascending=False)
        self.freq_rank["MinRank"] = self.freq_rank["Freq"].rank(method="min", ascending=False)
        self.freq_rank["MaxRank"] = self.freq_rank["Freq"].rank(method="max", ascending=False)
        self.freq_rank["FirRank"] = self.freq_rank["Freq"].rank(method="first", ascending=False)

        # Minimum values
        self.min_raw_freq = self.freq_rank["Freq"].min()
        self.min_avr_rank = self.freq_rank["AvgRank"].min()
        self.mix_max_rank = self.freq_rank["MaxRank"].min()

        # Maximum values
        self.max_raw_freq = self.freq_rank["Freq"].max()
        self.max_avg_rank = self.freq_rank["AvgRank"].max()
        self.max_den_rank = self.freq_rank["DenRank"].max()
        self.max_min_rank = self.freq_rank["MinRank"].max()
        self.max_max_rank = self.freq_rank["MaxRank"].max()
        self.max_fir_rank = self.freq_rank["FirRank"].max()

        # Sums
        self.total_tokens = self.freq_rank["Freq"].sum()
        self.sum_avg_rank = self.freq_rank["AvgRank"].sum()
        self.sum_den_rank = self.freq_rank["DenRank"].sum()
        self.sum_min_rank = self.freq_rank["MinRank"].sum()
        self.sum_max_rank = self.freq_rank["MaxRank"].sum()
        self.sum_fir_rank = self.freq_rank["FirRank"].sum()

        # count
        self.total_types = self.freq_rank.shape[0]

        self.zipf_structure()

        self.avg_nest, self.sub_avgrank = self.fractal_str("AvgRank")
        self.den_nest, self.sub_denrank = self.fractal_str("DenRank")
        self.min_nest, self.sub_minrank = self.fractal_str("MinRank")
        self.max_nest, self.sub_maxrank = self.fractal_str("MaxRank")
        self.fir_nest, self.sub_firrank = self.fractal_str("FirRank")

        self.harm_avg = harmonic(self.avg_nest)
        self.harm_den = harmonic(self.den_nest)
        self.harm_min = harmonic(self.min_nest)
        self.harm_max = harmonic(self.max_nest)
        self.harm_fir = harmonic(self.fir_nest)

        self.e_avg = euler_aprox(self.avg_nest)
        self.e_den = euler_aprox(self.den_nest)
        self.e_min = euler_aprox(self.min_nest)
        self.e_max = euler_aprox(self.max_nest)
        self.e_fir = euler_aprox(self.fir_nest)

    def fractal_str(self, rank):
        method_dict = {"AvgRank":"average","DenRank":"dense","MinRank":"min","MaxRank":"max","FirRank":"first"}
        element = self.freq_rank[["Freq",rank]]
        element = element.rename(columns={"Freq": "F0", rank: "R0"})
        element["F1"] = element.groupby("R0")["R0"].transform("count")
        element["AF1"] = element.groupby(["R0"])["F0"].transform("sum")
        element["R1"] = element["F1"].rank(method=method_dict[rank], ascending=False)

        str_level = 1
        finish = False
        while finish == False:
            element = element
            str_level+=1
            freq_key = "F{}".format(str_level)
            rank_key = "R{}".format(str_level)
            afre_key = "AF{}".format(str_level)
            prev_freq_key = "F{}".format(str_level-1)
            prev_rank_key = "R{}".format(str_level-1)
            prev_afre_key = "AF{}".format(str_level-1)
            element[freq_key] = element.groupby(prev_rank_key)[prev_rank_key].transform("count")
            element[afre_key] = element.groupby([prev_rank_key])["F0"].transform("sum")
            element[rank_key] = element[freq_key].rank(method=method_dict[rank], ascending=False)
            if element[freq_key].equals(element[prev_freq_key]):
                finish = True

        return str_level, element


    def subrank_stabilization(self):

        """ This ignore the value of the original frequencies, only counts subrank freq"""
        self.stabilization = self.freq_rank[["Freq","DenRank"]]
        self.lengths = [len(self.stabilization)]
        while len(self.stabilization) != 1:
            self.stabilization = self.stabilization.groupby("DenRank").count()
            self.stabilization.index.name=""
            self.stabilization["DenRank"] = self.stabilization["Freq"].rank(method="dense", ascending=False)
            self.lengths.append(len(self.stabilization))
            """print(self.stabilization)
            print(self.lengths)"""
        self.n = len(self.lengths)-2
        self.e_aprox_by_subrankstabilization = 0
        for x in range (0, self.n):
            self.e_aprox_by_subrankstabilization += 1/np.math.factorial(x)



    def zipf_structure(self):
        self.zipf = np.log(self.freq_rank)
        for rank in ["AvgRank", "DenRank", "MinRank", "MaxRank", "FirRank"]:
            base_df = self.freq_rank[["Freq",rank]].groupby(rank).first()
            x = np.log(np.array(base_df.index)).reshape(-1,1)
            y = np.log(np.array(base_df["Freq"])).reshape(-1,1)
            regressor = LinearRegression()
            regressor.fit(x, y)


            self.zipf_k[rank] = regressor.intercept_[0]
            self.zipf_alpha[rank] = regressor.coef_[0][0]


            self.zipf[rank] = self.zipf_alpha[rank]*self.zipf[rank]+self.zipf_k[rank]
            self.zipf["AbsEr"+rank] = abs(self.zipf[rank] - self.zipf["Freq"])
            self.zipf["SqrEr"+rank] = (self.zipf[rank] - self.zipf["Freq"])**2

            self.errors["ZMeanAE"+rank] = self.zipf["AbsEr"+rank].mean()
            self.errors["ZMeanSE"+rank] = self.zipf["SqrEr"+rank].mean()
            self.errors["ZMeanRSE"+rank] = (self.zipf["SqrEr"+rank].mean())**(1/2)
