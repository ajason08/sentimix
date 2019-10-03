import pandas as pd
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

class zipf():

    def __init__(self, frequency_df):
        self.freq_rank = frequency_df
        print(self.freq_rank)
        self.freq_rank["LogFreq"] = np.log(self.freq_rank["RawFreq"])
        #self.freq_rank["DenseRank"] = self.freq_rank["RawFreq"].rank(method="dense", ascending=False)
        self.freq_rank["LogDenseRank"] = np.log(self.freq_rank["DenseRank"])

        # Maxima
        self.max_freq = self.freq_rank["RawFreq"].max()
        self.max_rank = self.freq_rank["DenseRank"].max()
        self.e_aprox_by_max_rank = 0
        for x in range (0, int(self.max_rank)):
            self.e_aprox_by_max_rank += 1/np.math.factorial(x)


        # Minima
        self.min_freq = self.freq_rank["RawFreq"].min()
        self.min_rank = self.freq_rank["DenseRank"].min()

        # Sum and probability
        self.sum_freq = self.freq_rank["RawFreq"].sum()
        self.freq_rank["Probabilty"] = self.freq_rank["RawFreq"]/self.sum_freq

        # Ideal Zipf Values
        self.zipf_k = np.log(self.max_freq)
        self.zipf_alpha =  self.zipf_k/np.log(self.max_rank)
        self.freq_rank["Zipf"] = -self.zipf_alpha*np.log(self.freq_rank["DenseRank"])+self.zipf_k
        self.freq_rank["ZipfSqError"] = np.square(self.freq_rank["LogFreq"]-self.freq_rank["Zipf"])
        self.zipf_square_err =  self.freq_rank["ZipfSqError"].mean()

        # My model values (assuming e)
        self.e_k = np.power(self.zipf_k,np.e)
        self.e_alpha = self.e_k/np.log(self.max_rank)
        self.freq_rank["AsumingE"] = np.power(-self.e_alpha*np.log(self.freq_rank["DenseRank"])+self.e_k,1/np.e)
        self.freq_rank["AsumingSqError"] = np.square(self.freq_rank["LogFreq"]-self.freq_rank["AsumingE"])
        self.assuminge_square_err =  self.freq_rank["AsumingSqError"].mean()

        # My model values (aproximing e by number of reranks):
        self.subrank_stabilization()
        self.e_aprox_k_subranks = np.power(self.zipf_k,self.e_aprox_by_subrankstabilization)
        self.e_aprox_alpha_subranks = self.e_aprox_k_subranks/np.log(self.max_rank)
        self.freq_rank["AproxEbySubranks"] = np.power(-self.e_aprox_alpha_subranks*np.log(self.freq_rank["DenseRank"])+self.e_aprox_k_subranks,1/self.e_aprox_by_subrankstabilization)
        self.freq_rank["AproxBySubranksSqError"] = np.square(self.freq_rank["LogFreq"]-self.freq_rank["AproxEbySubranks"])
        self.aprox_sr_square_err = self.freq_rank["AproxBySubranksSqError"].mean()

        # My model values (aproximing e by number of ranks)
        self.e_aprox_k_ranks = np.power(self.zipf_k,self.e_aprox_by_max_rank)
        self.e_aprox_alpha_ranks = self.e_aprox_k_ranks/np.log(self.max_rank)
        self.freq_rank["AproxEbyRanks"] = np.power(-self.e_aprox_alpha_ranks*np.log(self.freq_rank["DenseRank"])+self.e_aprox_k_ranks,1/self.e_aprox_by_max_rank)
        self.freq_rank["AproxByRanksSqError"] = np.square(self.freq_rank["LogFreq"]-self.freq_rank["AproxEbyRanks"])
        self.aprox_mr_square_err = self.freq_rank["AproxByRanksSqError"].mean()


        # Modelo con simplificacion de la idea de Mandelbrot
        # ln(f) = alpha*ln(r+m)^x
        self.mandel_power = np.e
        self.mandel_q = (np.power(self.max_freq,1/self.mandel_power)-self.max_rank)/(1-np.power(self.max_freq,1/self.mandel_power))
        self.freq_rank["Harmonic"] = self.freq_rank["DenseRank"].apply(self.harmonic)
        self.freq_rank["e"] = self.freq_rank["DenseRank"].apply(self.euler)

        print(">>>>>>>>>>>>>>>>>>>>>>>")
        print(self.freq_rank[["DenseRank","Harmonic", "e"]])
        print("<<<<<<<<<<<<<<<<<<<<<<<<<")

        self.freq_rank["MyMandel"] = np.log(self.sum_freq/np.power(self.freq_rank["DenseRank"]+self.mandel_q,self.freq_rank["e"])*np.array(self.freq_rank["Harmonic"]))
        self.freq_rank["MyMandelSqErr"] = np.square(self.freq_rank["LogFreq"]-self.freq_rank["MyMandel"])
        self.mymandel_square_err = self.freq_rank["MyMandel"].mean()

    def harmonic(self, x):
        har = 0
        for x in range (1, int(x)):
            har +=  1/np.power((x + self.mandel_q),self.mandel_power)
        return har

    def euler(self, x):
        e = 0
        for x in range(0, int(x)):
            e += 1/np.math.factorial(x)
        return e

    def subrank_stabilization(self):
        """self.stabilization = self.freq_rank[["RawFreq", "DenseRank"]]
        print (self.stabilization)"""

        """ This ignore the value of the original frequencies, only counts subrank freq"""
        self.stabilization = self.freq_rank[["RawFreq","DenseRank"]]
        self.lengths = [len(self.stabilization)]
        while len(self.stabilization) != 1:
            self.stabilization = self.stabilization.groupby("DenseRank").count()
            self.stabilization.index.name=""
            self.stabilization["DenseRank"] = self.stabilization["RawFreq"].rank(method="dense", ascending=False)

            self.lengths.append(len(self.stabilization))
            print(self.stabilization)
            print(self.lengths)
        self.n = len(self.lengths)-2
        self.e_aprox_by_subrankstabilization = 0
        for x in range (0, self.n):
            self.e_aprox_by_subrankstabilization += 1/np.math.factorial(x)
    """
    def subrank_stabilization(self):
         This fuctions takes in acount original frecuencies in reranks
        pass"""


    def power_proof(self, power):
        self.power = power
        self.power_k = np.power(self.zipf_k,self.power)
        self.power_alpha = self.power_k/np.log(self.max_rank)
        self.freq_rank["Power"] = np.power(-self.power_alpha*np.log(self.freq_rank["DenseRank"])+self.power_k ,1/self.power)
        self.freq_rank["PowerError"] = self.freq_rank["LogFreq"]-self.freq_rank["Power"]
        self.power_square_err =  np.square(self.freq_rank["PowerError"]).mean()


    def rankfreq_graph(self,  euler = False, aprox_sr = False, aprox_mr=False ,power_rank=False, power = False):
        x = np.array(self.freq_rank["DenseRank"])
        z = np.array(self.freq_rank["RawFreq"])
        z_zipf = np.exp(np.array(self.freq_rank["Zipf"]))

        #mpl.rcParams["leyend.fontsize"] = 10
        print(x,z)
        mpl.pyplot.plot(x,z)
        mpl.pyplot.plot(x,z_zipf)
        legends = ["Data", "Zifp\n Error:{}".format(self.zipf_square_err)]

        if euler:
            z_euler = np.exp(np.array(self.freq_rank["AsumingE"]))
            mpl.pyplot.plot(x,z_euler, )
            legends.append("Power:e\n Error:{}".format(self.assuminge_square_err))

        if aprox_sr:
            z_aprox = np.exp(np.array(self.freq_rank["AproxEbySubranks"]))
            mpl.pyplot.plot(x,z_aprox)
            legends.append("Power: Aproximacion de e a n=maximo subrank ({})\n Error: {}".format(self.n, self.aprox_sr_square_err))

        if aprox_mr:
            z_aprox = np.exp(np.array(self.freq_rank["AproxEbyRanks"]))
            mpl.pyplot.plot(x,z_aprox)
            legends.append("Power: Aproximacion de e a n=maximo rank ({})\n Error: {}".format(self.max_rank, self.aprox_mr_square_err
            ))


        """if mymandel:
            z_mymandel = np.array(self.freq_rank["MyMandel"])
            mpl.pyplot.plot(x,z_mymandel)
            legends.append("MyMandel")"""

        if power:
            self.power_proof(power)
            z_power = np.exp(np.array(self.freq_rank["Power"]))
            mpl.pyplot.plot(x,z_power)
            legends.append("Power:{}".format(power))

        mpl.pyplot.legend(legends)
        mpl.pyplot.xlabel("Rank")
        mpl.pyplot.ylabel("Freq")
        mpl.pyplot.title("Freq vs Rank")
        mpl.pyplot.show()

    def rankloggraph(self,  euler = False, aprox_sr = False, aprox_mr=False ,power_rank=False, power = False):

        x = np.log(np.array(self.freq_rank["DenseRank"]))
        z = np.array(self.freq_rank["RawFreq"])
        z_zipf = np.exp(np.array(self.freq_rank["Zipf"]))

        #mpl.rcParams["leyend.fontsize"] = 10

        mpl.pyplot.plot(x,z)
        mpl.pyplot.plot(x,z_zipf)
        legends = ["Data", "Zifp\n Error:{}".format(self.zipf_square_err)]

        if euler:
            z_euler = np.exp(np.array(self.freq_rank["AsumingE"]))
            mpl.pyplot.plot(x,z_euler, linestyle=":" )
            legends.append("Power:e\n Error:{}".format(self.assuminge_square_err))

        if aprox_sr:
            z_aprox = np.exp(np.array(self.freq_rank["AproxEbySubranks"]))
            mpl.pyplot.plot(x,z_aprox)
            legends.append("Power: Aproximacion de e a n=maximo subrank ({})\n Error: {}".format(self.n, self.aprox_sr_square_err))

        if aprox_mr:
            z_aprox = np.exp(np.array(self.freq_rank["AproxEbyRanks"]))
            mpl.pyplot.plot(x,z_aprox)
            legends.append("Power: Aproximacion de e a n=maximo rank ({})\n Error: {}".format(self.max_rank, self.aprox_mr_square_err
            ))


        """if mymandel:
            z_mymandel = np.array(self.freq_rank["MyMandel"])
            mpl.pyplot.plot(x,z_mymandel)
            legends.append("MyMandel")"""

        if power:
            self.power_proof(power)
            z_power = np.exp(np.array(self.freq_rank["Power"]))
            mpl.pyplot.plot(x,z_power)
            legends.append("Power:{}".format(power))

        mpl.pyplot.legend(legends)
        mpl.pyplot.xlabel("Ln(Rank)")
        mpl.pyplot.ylabel("Freq")
        mpl.pyplot.title("Freq vs ln(Rank)")
        mpl.pyplot.show()

    def freqloggraph(self,  euler = False, aprox_sr = False, aprox_mr=False ,power_rank=False, power = False):

        x = np.array(self.freq_rank["DenseRank"])
        z = np.array(self.freq_rank["LogFreq"])
        z_zipf = np.array(self.freq_rank["Zipf"])

        #mpl.rcParams["leyend.fontsize"] = 10

        mpl.pyplot.plot(x,z)
        mpl.pyplot.plot(x,z_zipf)
        legends = ["Data", "Zifp\n Error:{}".format(self.zipf_square_err)]

        if euler:
            z_euler = np.array(self.freq_rank["AsumingE"])
            mpl.pyplot.plot(x,z_euler)
            legends.append("Power:e\n Error:{}".format(self.assuminge_square_err))

        if aprox_sr:
            z_aprox = np.array(self.freq_rank["AproxEbySubranks"])
            mpl.pyplot.plot(x,z_aprox)
            legends.append("Power: Aproximacion de e a n=maximo subrank ({})\n Error: {}".format(self.n, self.aprox_sr_square_err))

        if aprox_mr:
            z_aprox = np.array(self.freq_rank["AproxEbyRanks"])
            mpl.pyplot.plot(x,z_aprox)
            legends.append("Power: Aproximacion de e a n=maximo rank ({})\n Error: {}".format(self.max_rank, self.aprox_mr_square_err
            ))


        """if mymandel:
            z_mymandel = np.array(self.freq_rank["MyMandel"])
            mpl.pyplot.plot(x,z_mymandel)
            legends.append("MyMandel")"""

        if power:
            self.power_proof(power)
            z_power = np.array(self.freq_rank["Power"])
            mpl.pyplot.plot(x,z_power)
            legends.append("Power:{}".format(power))

        mpl.pyplot.legend(legends)
        mpl.pyplot.xlabel("Rank")
        mpl.pyplot.ylabel("ln(Freq)")
        mpl.pyplot.title("ln(Freq) vs Rank")
        mpl.pyplot.show()



    def lograph(self,  euler = False, aprox_sr = False, aprox_mr=False , my_mandel=False, power = False):

        x = np.log(np.array(self.freq_rank["DenseRank"]))
        z = np.array(self.freq_rank["LogFreq"])
        z_zipf = np.array(self.freq_rank["Zipf"])

        #mpl.rcParams["leyend.fontsize"] = 10

        mpl.pyplot.plot(x,z)
        mpl.pyplot.plot(x,z_zipf)
        legends = ["Data", "Zifp\n Error:{}".format(self.zipf_square_err)]

        if euler:
            z_euler = np.array(self.freq_rank["AsumingE"])
            mpl.pyplot.plot(x,z_euler, linestyle=":",  linewidth=10)
            legends.append("Power:e\n Error:{}".format(self.assuminge_square_err))

        if aprox_sr:
            z_aprox = np.array(self.freq_rank["AproxEbySubranks"],)
            mpl.pyplot.plot(x,z_aprox, linestyle="-.")
            legends.append("Power: Aproximacion de e a n=maximo subrank ({})\n Error: {}".format(self.n, self.aprox_sr_square_err))

        if aprox_mr:
            z_aprox = np.array(self.freq_rank["AproxEbyRanks"])
            mpl.pyplot.plot(x,z_aprox, linestyle="--")
            legends.append("Power: Aproximacion de e a n=maximo rank ({})\n Error: {}".format(self.max_rank, self.aprox_mr_square_err
            ))

        if my_mandel:
            z_mymandel = np.array(self.freq_rank["MyMandel"])
            mpl.pyplot.plot(x,z_mymandel)
            legends.append("MyMandel")

        if power:
            self.power_proof(power)
            z_power = np.array(self.freq_rank["Power"])
            mpl.pyplot.plot(x,z_power)
            legends.append("Power:{}".format(power))

        mpl.pyplot.legend(legends)
        mpl.pyplot.xlabel("Ln(Rank)")
        mpl.pyplot.ylabel("ln(Freq)")
        mpl.pyplot.title("ln(Freq) vs ln(Rank)")
        mpl.pyplot.show()


    def errors_graph(self,  euler = False, aprox_sr = False, aprox_mr=False ,power_rank=False, power = False):

        x = np.log(np.array(self.freq_rank["DenseRank"]))
        z_zipf = np.array(self.freq_rank["ZipfSqError"])

        #mpl.rcParams["leyend.fontsize"] = 10

        mpl.pyplot.plot(x,z_zipf)
        legends = ["Error cuadratico a zipf\n Media: {}".format(self.zipf_square_err)]

        if euler:
            z_euler = np.array(self.freq_rank["AsumingSqError"])
            mpl.pyplot.plot(x,z_euler)
            legends.append("Error cuadratico a modelo parametro e :e\n Error:{}".format(self.assuminge_square_err))

        if aprox_sr:
            z_aprox = np.array(self.freq_rank["AproxBySubranksSqError"])
            mpl.pyplot.plot(x,z_aprox)
            legends.append("Error cuadratico a modelo en  n=maximo subrank ({})\n Error: {}".format(self.n, self.aprox_sr_square_err))

        if aprox_mr:
            z_aprox = np.array(self.freq_rank["AproxByRanksSqError"])
            mpl.pyplot.plot(x,z_aprox)
            legends.append("Power: Aproximacion de e a n=maximo rank ({})\n Error: {}".format(self.max_rank, self.aprox_mr_square_err
            ))


        """if mymandel:
            z_mymandel = np.array(self.freq_rank["MyMandel"])
            mpl.pyplot.plot(x,z_mymandel)
            legends.append("MyMandel")"""

        if power:
            self.power_proof(power)
            z_power = np.array(self.freq_rank["Power"])
            mpl.pyplot.plot(x,z_power)
            legends.append("Power:{}".format(power))

        mpl.pyplot.legend(legends)
        mpl.pyplot.legend(legends)
        mpl.pyplot.xlabel("Ln(Rank)")
        mpl.pyplot.ylabel("Square Error")
        mpl.pyplot.title("Square Error vs ln(Rank)")
        mpl.pyplot.show()

        mpl.pyplot.show()
