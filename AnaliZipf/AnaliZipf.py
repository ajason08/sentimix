import pandas as pd
import numpy as np
from scipy.special import digamma, zeta
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def harmonic(n):
    """
    Calculates the harmonic number of x. If s is complex the result becomes complex.
    sum of 1/x in range (1,x)

    Parameters:
    x (int): The upper index of the sumatory.

    Returns:
    (float): The harmonic number for x
    """
    return float(digamma(n + 1) + np.euler_gamma)

def harmonic_g1(k,n,s):
    """
    Calculates the generalized form of the harmonic number with parameters (k, n, s).
    sum of 1/(x+k)**s for x in rank (1,n)

    Parameters:
    k (int): Constant to add to the counter.
    n (int): The upper limit of the sumatory
    m (float): The power to apply over (x+k)

    Returns:
    (float): The generalized harmonic number for (k,n,s)

    """
    harm = 0
    for x in range (1,int(n)):
        oldharm = harm
        suma = k+x
        potencia = suma**m
        cociente = 1/potencia
        harm += cociente
        if format(oldharm, '.16g') == format(harm, '.16g'):
            return harm
    return harm

def harmonic_g2(k ,n, s):
    """
    Calculates the generalized form of the harmonic number with parameters (c, n m)
    sum of 1/(x+c)**m for x in rank (1,n), uses harmonic_g1 and one generalized
    harmonic number form propiety, takes less time.

    Parameters:
    k (int): Constant to add to the counter.
    n (int): The upper limit of the sumatory
    m (float): The power to apply over (x+k)

    Returns:
    (float): The generalized harmonic number for (k,n,s)
    Parameters:
    c (int): add constant
    n (int): sumatory count limit
    m (float): add power
    """
    return harmonic_g1(0,k+n,s)-harmonic_g1(0,k+1,m)

def euler_aprox(n):
    """
    Calculates the sum of (1/x!) in the range (0, to n). this is an aprox to
    euler's constant of n terms.

    Parameters:
    n (int): The upper index of the summatory

    Returns:
    (float): An aprox to e with the sum of (1/x!) for n
    """
    e = 0
    for i in range(0, int(x)):
        e += 1/np.math.factorial(i)
    return e

def regresor(x,y):
    regr = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=8)
    regr.fit(x,y)
    k = regr.intercept_[0]
    alpha = regr.coef_[0][0]
    return (k, alpha)

def gelbuck_sidorov(x,y,c):
    cxi = c**x
    bn1 = np.sum(x/cxi)
    bn2 = np.sum((x*y)/cxi)
    bn12 = bn1*bn2
    bn3 = np.sum(x**2/cxi)
    bn4 = np.sum(y/cxi)
    bn34 = bn3*bn4
    bn = bn12-bn34
    bd1 = (bn1)**2
    bd2 = bn3
    bd3 = np.sum(1/cxi)
    bd23 = bd2*bd3
    bd = bd1-bd23
    b = bn/bd
    an1 = bn4
    an2 = b*bd3
    an = an1-an2
    ad = bn1
    a = an/ad
    return (b,a)

def graph_2d_common_x(x, y, knee,title, x_label, y_label, leyends, curves = False):
    n = len(y)
    simbols = ["-", "^", "s", "o", "^", "^", "d"]

    for i in range(0,n):
        if not curves:
            plt.plot(x, y[i], simbols[i], label=leyends[i], markeredgewidth=0.05)
        else:
            plt.plot(x, y[i], "-", label=leyends[i], markeredgewidth=0.05)
    plt.axvline(x=knee, ymin=0, color="r",  linestyle='dashed')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

def graph_3d_common_x(x, y, z, title, x_label, y_label, leyends):
    mpl.rcParams["legend.fontsize"] = 10
    fig = plt.figure
    ax = fig.gca(projection="3d")
    n = len(y)
    for i in range(0,n):
        ax.plot(x,y[i],z[i], label=leyends[i])
    ax.leyend()
    plt.show()

def Zipf_Model(r, k, s):
    return k-s*r

def Mandelbrot_Model(r, k, q, s):
    if  q > r.max():
        return k/(r+q)**s
    else:
        return -5


def MandriguezSModel(r, k, alpha, s):
    return np.exp((k+s*np.log(r)))

def Zipf_Fit(r,f):
    popt, pcov  = curve_fit(Zipf_Model, np.array(r), np.array(f), p0=[f.max(),1] ,maxfev=5000)
    return (popt)

def Mandelbrot_Fit(r,f,p):
    n = 1/harmonic_g1(np.e,r.max(),1)
    popt, pcov = curve_fit(Mandelbrot_Model, r, f, p0=p ,maxfev=500000)
    return (popt)



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
    cero_trunkated = {}

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

    # Mandelbroot curve_fit
    mandel = pd.DataFrame()
    mandel_coef = {}

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
    LinError = pd.DataFrame()
    LinSqrEr = pd.DataFrame()
    LogError = pd.DataFrame()
    LogSqrEr = pd.DataFrame()

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

        # Set zipf models
        self.ZipfSkl_lr(one_per_rank=True)
        self.ZipfGS()
        self.ZipfMLE()
        self.ZipfSklLrTr()


        # Set mandelbroot
        self.Mandelbrot()

        # Set My Models




    def Rank(self):
        # Order the frecuencies from high to low
        self.freq_rank = self.freq_rank.sort_values(by="Freq", ascending = False)
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

            self.LinError["ZZC{}".format(rank)] = abs(self.freq_rank["Freq"]-np.exp(self.zipf_cero_cross[rank]))
            self.LinSqrEr["ZZC{}".format(rank)] = (self.freq_rank["Freq"]-np.exp(self.zipf_cero_cross[rank]))**2

            self.LogError["ZZC{}".format(rank)] = abs(np.log(self.freq_rank["Freq"])-(self.zipf_cero_cross[rank]))
            self.LogSqrEr["ZZC{}".format(rank)] = (np.log(self.freq_rank["Freq"])-self.zipf_cero_cross[rank])**2


    def Knee(self):
        base = np.log(self.freq_rank)
        for rank in ["AvgRank", "DenRank", "MinRank", "MaxRank", "FirRank"]:

            base_rank = base[["Freq",rank]].groupby(rank).first()
            x = np.array(base_rank.index)
            y = np.array(base_rank["Freq"])
            p1 = np.array([x[0], y[0]])
            p2 = np.array([x[-1], y[-1]])
            self.knees[rank] = base_rank
            self.knees[rank]["Dist"] = pd.Series()
            for i in range (int(len(x)/2), len(x)):
                p3 = np.array([x[i],y[i]])
                distance = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
                self.knees[rank]["Dist"].iloc[i] = distance
            knee_rank = self.knees[rank]["Dist"].idxmax()
            knee_freq = self.knees[rank].loc[knee_rank, "Freq"]
            self.max_knees[rank] = [knee_rank, knee_freq, self.knees[rank]["Dist"].max()]
            self.cero_trunkated[rank] = self.knees[rank].truncate(after=self.max_knees[rank][0])

    """Zipfs"""

    def ZipfSkl_lr(self, one_per_rank = True):
        self.zipf_skl_lr = np.log(self.freq_rank)
        for rank in ["AvgRank", "DenRank", "MinRank", "MaxRank", "FirRank"]:
            if one_per_rank:
                base_df = self.zipf_skl_lr.groupby(rank).first()
                r = np.array(base_df.index).reshape(-1,1)
                f = np.array(base_df["Freq"]).reshape(-1,1)
            else:
                base_df = self.zipf_skl_lr
                r = np.array(base_df[rank]).reshape(-1,1)
                f = np.array(base_df["Freq"]).reshape(-1,1)
            self.k_skl_lr[rank], self.alpha_skl_lr[rank] = regresor(r,f)
            self.zipf_skl_lr[rank] = self.alpha_skl_lr[rank]*self.zipf_skl_lr[rank]+self.k_skl_lr[rank]

            self.LinError["ZSLR{}".format(rank)] = self.freq_rank["Freq"]-np.exp(self.zipf_skl_lr[rank])
            self.LinSqrEr["ZSLR{}".format(rank)] = (self.freq_rank["Freq"]-np.exp(self.zipf_skl_lr[rank]))**2

            self.LogError["ZSLR{}".format(rank)] = abs(np.log(self.freq_rank["Freq"])-self.zipf_skl_lr[rank])
            self.LogSqrEr["ZSLR{}".format(rank)] = (np.log(self.freq_rank["Freq"])-self.zipf_skl_lr[rank])**2


    def  ZipfGS(self):
        self.zipf_gs_lr = np.log(self.freq_rank)
        for rank in ["AvgRank", "DenRank", "MinRank", "MaxRank", "FirRank"]:
            base_df = self.zipf_gs_lr
            r = np.array(base_df[rank]).reshape(-1,1)
            f = np.array(base_df["Freq"]).reshape(-1,1)
            self.k_gs_lr[rank], self.alpha_gs_lr[rank] = gelbuck_sidorov(r,f,np.e)
            self.zipf_gs_lr[rank] = self.alpha_gs_lr[rank]*self.zipf_gs_lr[rank]+self.k_gs_lr[rank]

            self.LinError["ZGS{}".format(rank)] = abs(self.freq_rank["Freq"]-np.exp(self.zipf_gs_lr[rank]))
            self.LinSqrEr["ZGS{}".format(rank)] = (self.freq_rank["Freq"]-np.exp(self.zipf_gs_lr[rank]))**2

            self.LogError["ZGS{}".format(rank)] = abs(np.log(self.freq_rank["Freq"])-self.zipf_gs_lr[rank])
            self.LogSqrEr["ZGS{}".format(rank)] = (np.log(self.freq_rank["Freq"])-self.zipf_gs_lr[rank])**2

    def ZipfMLE(self): # Really not MLE but curve_fit
        self.zipf_mle = np.log(self.freq_rank)
        for rank in ["AvgRank", "DenRank", "MinRank", "MaxRank", "FirRank"]:
            base_df = self.zipf_mle.groupby(rank).first()
            r = np.array(base_df.index)
            f = np.array(base_df["Freq"])
            const = Zipf_Fit(r,f)
            self.k_mle[rank] = const[0]
            self.alpha_mle[rank] = const[1]
            self.zipf_mle[rank] = self.k_mle[rank]-(self.alpha_mle[rank]*self.zipf_mle[rank])

            self.LinError["ZMLE{}".format(rank)] = abs(self.freq_rank["Freq"]-np.exp(self.zipf_mle[rank]))
            self.LinSqrEr["ZMLE{}".format(rank)] = (self.freq_rank["Freq"]-np.exp(self.zipf_mle[rank]))**2

            self.LogError["ZMLE{}".format(rank)] = abs(np.log(self.freq_rank["Freq"])-self.zipf_mle[rank])
            self.LogSqrEr["ZMLE{}".format(rank)] = (np.log(self.freq_rank["Freq"])-self.zipf_mle[rank])**2


    def ZipfSklLrTr(self):
        self.zipf_skl_lr_tr = np.log(self.freq_rank)
        for rank in ["AvgRank", "DenRank", "MinRank", "MaxRank", "FirRank"]:

            base_df = self.zipf_skl_lr_tr.groupby(rank).first().truncate(after=self.max_knees[rank][0])
            r = np.array(base_df.index).reshape(-1,1)
            f = np.array(base_df["Freq"]).reshape(-1,1)

            self.k_skl_lr_tr[rank], self.alpha_skl_lr_tr[rank] = regresor(r,f)
            self.zipf_skl_lr_tr[rank] = self.alpha_skl_lr_tr[rank]*self.zipf_skl_lr_tr[rank]+self.k_skl_lr_tr[rank]

            self.LinError["ZSLRTR{}".format(rank)] = self.freq_rank["Freq"]-np.exp(self.zipf_skl_lr_tr[rank])
            self.LinSqrEr["ZSLRTR{}".format(rank)] = (self.freq_rank["Freq"]-np.exp(self.zipf_skl_lr_tr[rank]))**2

            self.LogError["ZSLRTR{}".format(rank)] = abs(np.log(self.freq_rank["Freq"])-self.zipf_skl_lr_tr[rank])
            self.LogSqrEr["ZSLRTR{}".format(rank)] = (np.log(self.freq_rank["Freq"])-self.zipf_skl_lr_tr[rank])**2



    def Mandelbrot(self):
        self.mandel = np.log(self.freq_rank)
        for rank in ["AvgRank", "DenRank", "MinRank", "MaxRank", "FirRank"]:
            base_df = self.freq_rank.groupby(rank).first()

            r = np.array(base_df.index)
            f = np.array(base_df["Freq"])

            r_knee = np.exp(self.max_knees[rank][0])
            f_knee = np.exp(self.max_knees[rank][1])
            d_knee = np.exp(self.max_knees[rank][1])


            def Mandelbroot_coef_for_zero_and_knee(p):
                k,q,s = p
                f0 = k-(r.max()+q)**s
                f1 = k-f.max()*(1+q)**s
                f2 = k-f_knee*(r_knee+q)**s
                return (np.array([f0,f1,f2]))

            # Hints
            s_hint = np.log(f.max()/r.max())
            q_hint = ((r_knee+1)*f_knee**(1/s_hint))/(f.max()**(1/s_hint))
            k_hint = self.token_count/(harmonic_g1(q_hint,r.max(),s_hint))
            #k_hint = f.max()*(1+q_hint)**s_hint

            k, q, s = fsolve(Mandelbroot_coef_for_zero_and_knee, [k_hint,q_hint,s_hint], maxfev = 80000)
            Hqrs = f.max()/harmonic_g1(q,r.max(),s)
            self.mandel_coef[rank] = Mandelbrot_Fit(r, f, [k, q, s])
            q_fit = self.mandel_coef[rank][1]
            s_fit = self.mandel_coef[rank][2]
            k_fit = self.mandel_coef[rank][0]

            print ("------"+rank+"---")
            print(k_hint, k, k_fit)
            print(q_hint, q,q_fit)
            print(s_hint, s,s_fit)
            print ("------"+rank+"---\n\n")


            self.mandel[rank] = np.log(k_hint) - s_hint*np.log(self.freq_rank[rank]+q_hint)
            #self.mandel[rank] = np.log(k_fit)-s_fit*np.log(self.freq_rank[rank]+q_fit)
            #self.mandel[rank] = np.log(k) - s*np.log(self.freq_rank[rank]+q)

            self.LinError["MA{}".format(rank)] = self.freq_rank["Freq"]-np.exp(self.mandel[rank])
            self.LinSqrEr["MA{}".format(rank)] = (self.freq_rank["Freq"]-np.exp(self.mandel[rank]))**2

            self.LogError["MA{}".format(rank)] = abs(np.log(self.freq_rank["Freq"])-self.mandel[rank])
            self.LogSqrEr["MA{}".format(rank)] = (np.log(self.freq_rank["Freq"])-self.mandel[rank])**2


    def ManDriguezE(self):
        pass

    def MandriguezS(self):
        pass
