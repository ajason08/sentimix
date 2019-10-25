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
    #print("Text statistics")
    n_tok = x["Freq"].sum()
    n_typ = x.shape[0]


    # Catalogue
    #print("build catalogue")
    catalogue = x
    x = None

    catalogue.index.name = "Types"
    catalogue.columns = ["Counts"]
    catalogue["P(i)"] = catalogue["Counts"]/n_tok # The probability of the type in the text
    catalogue.sort_values(by="P(i)",ascending=False,inplace=True)

    # Count and rank
    #print("count and rank")
    catalogue["N(i)"] = catalogue.groupby("P(i)")["P(i)"].transform("count") #Words whit the same probability
    """The number of words whit propability <= x, the RANK!!!"""
    catalogue["i"] = catalogue["P(i)"].rank(method="max", ascending=False)
    catalogue["NormRank"] = catalogue["i"]/catalogue["i"].max()
    catalogue.sort_values(by="NormRank",ascending=False,inplace=True)
    catalogue["PHI(i)"] = catalogue["i"]/n_typ


    # Probability in cataloge and comulative probability in the catalogue
    #print("F and information")
    catalogue["F(i)"] = catalogue["N(i)"]/catalogue.shape[0]

    # Information
    catalogue["x(i)"] = -np.log(catalogue["P(i)"]) #The negative entropy of the word
    catalogue["h(i)"] = catalogue["P(i)"]*catalogue["x(i)"]
    catalogue["h(c)"] = -catalogue["F(i)"]*np.log(catalogue["F(i)"])





    # Prepare to checks
    #print("Checks")
    catalogue["NiPi"] = catalogue["N(i)"]*catalogue["P(i)"]
    catalogue["NiEXi"] = catalogue["N(i)"]*np.exp(-catalogue["x(i)"])
    catalogue["FiEXi"] = catalogue["F(i)"]*np.exp(-catalogue["x(i)"])
    catalogue["NiXi"] = catalogue ["N(i)"]*catalogue["x(i)"]
    # Same rank groups
    same_rank = catalogue.groupby("i").first()
    # Closure checks
    closure_1 = int(round(same_rank["NiEXi"].sum(),0)) - 1
    closure_n = int(round(1/same_rank["FiEXi"].sum(),0))-n_typ

    # Mean and variance checks
    mean_data = round(catalogue["x(i)"].mean(),3)

    ind1 = (abs(same_rank['x(i)'] - mean_data)).sort_values().index[0]
    # Get the index in the cataloge
    ind_cat = catalogue[catalogue["i"]==ind1].iloc[0].name
    # Check if all elements have the same rank
    if same_rank.shape[0] > 1:
        ind2 = (abs(same_rank['x(i)'] - mean_data)).sort_values().index[1]
    else:
        ind2 = ind1
    mean_ind = [min(ind1,ind2),max(ind1,ind2)]


    mean_formula = round(same_rank["NiXi"].sum()/n_typ,3)

    var_data = round(catalogue["x(i)"].var(),3)
    var_formula = same_rank["N(i)"]*(same_rank["x(i)"]-mean_formula)**2
    var_formula = round(var_formula.sum()/n_typ,3)


    mean_t_var = round(np.log(n_typ)+var_formula/2,3)


    # Information checks
    h = round(catalogue["h(i)"].sum(),3)
    h_form1 = round(np.log(n_typ) - var_formula/2,3)
    h_form2 = round(mean_t_var - var_formula,3)
    h_c = round(same_rank["h(c)"].sum(),3)

    var_inf = abs(round(h-h_c, 2))



    if (closure_1!=0 or closure_n!=0):
        closure = False
    else:
        closure = True

    if mean_data != mean_formula:
        means = False
        variance = False
    else:
        means = True
        if var_data != var_formula:
            variance = False

        else:
            variance = True

    #constants1 (Zipf slope)
    #Esta es la línea cool pero en la otra formula de ZAX1 hay que dividirla por n_tok**2, lo mimso pa la siguiente AX0 = -np.log(catalogue["P(i)"]/catalogue.iloc[0]["P(i)"])/np.log(catalogue["i"]/catalogue.iloc[0]["i"])
    max_prob = catalogue["P(i)"].min()
    #Esta tambien funciona por lo mismo AX = -np.log(catalogue["P(i)"]/max_prob)/np.log(catalogue["i"]/catalogue["i"].max())


    catalogue["AX"] = catalogue["x(i)"]/np.log(catalogue["NormRank"])


    if mean_ind[1] !=  mean_ind[0]:
        Xm1 = same_rank.loc[mean_ind[1]]["NormRank"]
        Xm2 = same_rank.loc[mean_ind[0]]["NormRank"]
        Xm = Xm2-Xm1
        Ym1 = same_rank.loc[mean_ind[1]]["x(i)"]
        Ym2 = same_rank.loc[mean_ind[0]]["x(i)"]
        Ym = Ym2-Ym1
        AXm = Ym/Xm
    else:
        AXm=catalogue["AX"][0]
    #A = (np.log(catalogue.iloc[0]["P(i)"]))/()


    #Zipf aprox with AX1 and AXm
    catalogue["ZAX1"] = np.exp((-catalogue["AX"]*np.log(catalogue["i"]/catalogue["i"][0])+catalogue["x(i)"][0]))/(n_tok)
    catalogue["ZAXm"] = np.exp((catalogue["AX"][-1]*np.log(catalogue["i"]/catalogue["i"][-1])+catalogue["x(i)"][-1]))*(catalogue["NormRank"].sum())/(catalogue["i"].max()**2)

    print(n_typ,n_tok, same_rank.shape, catalogue["N(i)"].max(), catalogue["i"].max(), same_rank.shape[0]*3000,
          same_rank.shape[0]*catalogue["N(i)"].max())

    print(catalogue["AX"][0], catalogue["AX"][-1], catalogue.loc[ind_cat, "AX"])
    #print((-np.log(catalogue["ZAXm"][0])+np.log(catalogue["ZAXm"][-1]))/(catalogue["NormRank"][0]-catalogue["NormRank"][-1]))



    return(catalogue[["i", "NormRank","P(i)", "F(i)", "ZAX1", "ZAXm"]],
           [np.exp(-mean_data), np.exp(-mean_formula), np.exp(-mean_t_var)],
           [Xm1,Xm2,np.exp(-Ym1),np.exp(-Ym2)])
