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



class zipf():

    """
    Define elements
    """
    freq_rank = pd.DataFrame()

    # Tokens / Types
    token_count = 0
    types_count = 0
    TypTokRatio = 0

    # ZipfModel Cero cross (To calculate data truncation)
    zipf_cero_cross = pd.DataFrame()
    alpha_cero_cross = {}
    k_cero_cross = {}
    cero_trunkated = pd.DataFrame()

    # ZipfModel 1 (alpha and k calc with sklearn linear regression)
    zipf_skl_lr = pd.DataFrame()
    alpha_skl_lr = {}
    k_skl_lr = {}

    # ZipfModel 2 (alpha and k calculated with Gelbuk-Sidorov method)
    zipf_gs_lr = pd.DataFrame()
    alpha_gs_lr = {}
    k_gs_lr = {}

    # ZipfModel 3 (alpha and k whith maximum likehood estimation)
    zipf_mle = pd.DataFrame()
    alpha_mle = {}
    k_mle = {}

    # ZipfModel 4 (Alfa and beta calculated with sklearn whith trucated data)
    zipf_skl_lr_tr = pd.DataFrame()
    alpha_skl_lr_tr = {}
    k_skl_lr_tr = {}

    # PoliZipfModel (Model whith polinomic linear regresion)
    polizipf = pd.DataFrame()
    polizipf_coef =  {}

    # Mandelbroot maximum likehood
    mandel_mle = pd.DataFrame()
    mandel_k = {}
    mandel_alpha = {}

    # My model (Alpha and k caculated by maximum likehood estimation)
    rod_mle = pd.DataFrame()
    rod_k_mle = 0
    rod_alpha_mle = 0

    # My model (K by zero cross, alpha whit e and e aprox)
    rod = pd.DataFrame()
    rod_k_mle = {}
    rod_alpha = {}
    rod_sigma = {}

    # Errors
    Error = pd.DataFrame()
    SqrEr = pd.DataFrame()

    # kness
    knees = {"AvgRank":pd.DataFrame(),
             "DenRank":pd.DataFrame(),
             "MinRank":pd.DataFrame(),
             "MaxRank":pd.DataFrame(),
             "FirRank":pd.DataFrame()}
    max_knees = {"AvgRank":[],
             "DenRank":[],
             "MinRank":[],
             "MaxRank":[],
             "FirRank":[]}

    # Fractal structures and correspondig harmonics and e aproximations
    avg_nest, sub_avgrank = 0, pd.DataFrame()
    den_nest, sub_denrank = 0, pd.DataFrame()
    min_nest, sub_minrank = 0, pd.DataFrame()
    max_nest, sub_maxrank = 0, pd.DataFrame()
    fir_nest, sub_firrank = 0, pd.DataFrame()

    harm_avg = 0
    harm_den = 0
    harm_min = 0
    harm_max = 0
    harm_fir = 0

    e_avg = 0
    e_den = 0
    e_min = 0
    e_max = 0
    e_fir = 0

    def __init__(self, bow):
        """
        Takes a BOW of words of a text and build his Zipfian estructure.

        Params:
        bow: A BOW of text in dict format which each token as keyword
        """

        # Takes  the BOW dict and transform it into pd_data
        self.freq_rank["Freq"] =  pd.Series(bow)

        # Order and calculate ranks
        self.Rank()

        # Types and token
        self.token_count = self.freq_rank["Freq"].sum()
        self.types_count = self.freq_rank.shape[0]
        self.TypTokRatio = self.types_count/self.token_count

        # Set zerocross line
        self.ZipfZeroCross()

        # Set knees
        self.Knee()



    def Rank(self):
        # Order the frecuencies from high to low
        self.freq_rank.sort_values(by="Freq", ascending = False)
        # Dense rank frecuencies from high to low
        self.freq_rank["AvgRank"] = self.freq_rank["Freq"].rank(method="average", ascending=False)
        self.freq_rank["DenRank"] = self.freq_rank["Freq"].rank(method="dense", ascending=False)
        self.freq_rank["MinRank"] = self.freq_rank["Freq"].rank(method="min", ascending=False)
        self.freq_rank["MaxRank"] = self.freq_rank["Freq"].rank(method="max", ascending=False)
        self.freq_rank["FirRank"] = self.freq_rank["Freq"].rank(method="first", ascending=False)

    def ZipfZeroCross(self):
        self.k_cero_cross = np.log(self.freq_rank["Freq"].max())
        k = self.k_cero_cross
        for rank in ["AvgRank", "DenRank", "MinRank", "MaxRank", "FirRank"]:
            self.alpha_cero_cross = self.k_cero_cross/np.log(self.freq_rank[rank].max())
            alpha = self.alpha_cero_cross
            self.zipf_cero_cross[rank] = -alpha*np.log(self.freq_rank[rank])+k
            self.Error["ZZC".format(rank)] = np.log(self.freq_rank["Freq"])-self.zipf_cero_cross[rank]
            self.SqrEr["ZZC".format(rank)] = (np.log(self.freq_rank["Freq"])-self.zipf_cero_cross[rank])**2

    def Knee(self):
        base = np.log(self.freq_rank)
        for rank in ["AvgRank", "DenRank", "MinRank", "MaxRank", "FirRank"]:
            base_rank = base[["Freq",rank]].groupby(rank).first()
            x = np.array(list(base_rank.index))
            y = np.array(base_rank["Freq"])
            p1 = np.array([x[0], y[0]])
            p2 = np.array([x[-1], y[-1]])
            self.knees[rank] = base_rank
            for i in range (0, len(x)-1):
                p3 = np.array([x[i],y[i]])
                print(p1,p2,p3)
                distance = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
                self.knees[rank]["Freq"].iloc[i] = distance
            self.max_knees[rank] = self.knees[rank]["Freq"].max()
