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
    Calculates the sum of (1/x!) in the range (0, to n), this is an aprox to
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

def gelbuck_sidorov(x,y,c):
    """
    Calculates the zipf parameters through Gelbuk-Sidorov linear regression.

    Parameters:
    x (np.array): The frecuency vector
    y (np.array): The rank vector
    c (float): The logarithm base

    Returns:
    b (float): The proporcionality constant (zero cross in logarithmic scale)
    a (float): The power (slope of the curve in logarithmic scale)
    """

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


def zipf_belevitch(x):
    """
    Calculates the zipf parameters through Belevitch paper

    Parameters:
    x (pd.DataFrame): The frecuencies dataframe with cols ["Word", "Freq"]

    Returns:
    mean list(rank (float), freq(float), value (float)): The mean position and value.
    sqr_sigma (float): The square sigma.
    h = mean_information
    """

    # Text statistics
    n_tok = x["Freq"].sum().values[0]
    n_typ = x.shape[0]
    text_prob = x/n_tok
    text_prob.columns = ["P(i)"]
    text_prob["Inf"] = -np.log(text_prob["P(i)"])

    # Cataloge (Bow)
    catalogue = pd.Series(text_prob.groupby("P(i)").groups).apply(list).to_frame().sort_index(ascending=False)
    catalogue["P(i)"] = catalogue.index
    catalogue.columns = ["Elements","P(i)"]

    groups = catalogue.shape[0]
    catalogue["N(i)"] = catalogue["Elements"].apply(len)
    catalogue = catalogue[["Elements","N(i)","P(i)"]]
    catalogue["i"] = catalogue["N(i)"].cumsum()
    catalogue.set_index('i', inplace=True)
    catalogue["F(i)"] = catalogue["N(i)"]/n_typ
    catalogue["Phi(i)"] = catalogue["F(i)"].cumsum()
    # catalogue["Phi2(i)"] = catalogue.index/n_typ just for check the identity
    catalogue["x(i)"] = -np.log(catalogue["P(i)"])
    catalogue["NiXi"] = catalogue["N(i)"]*catalogue["x(i)"]

    # Statistics
    check = int(round(1/(catalogue["F(i)"]*np.e**(-catalogue["x(i)"])).sum(),0)) #must be the number of types
    means = [0,0,0]
    means[0] = catalogue["x(i)"].mean()
    means[1] = catalogue["NiXi"].sum()/n_typ
    means[2] = text_prob["Inf"].mean()

    variance = [0,0,0]
    variance[0] = catalogue["x(i)"].var()
    variance[1] = ((catalogue["N(i)"]*(catalogue["x(i)"]-means[1])**2).sum())/n_typ
    variance[2] = text_prob["Inf"].var()

    print(means)
    print(variance)

    ###print(text_prob)
    ###print(catalogue)
