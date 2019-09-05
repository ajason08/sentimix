import pandas as pd
import json
from internet_taggin import internet_element_taggin, taggin_numbers

CORPUS = "./data/spanglish_trial_release.txt"

def open_data(data_file):
	"""
	Read txt data file whit one JSON object by line.

	Open a text file contains one JSON object by line
	without any separator. Put each line in a list, put
	the list in a Pandas dataframe , and return this
	dataframe.

	Parameters:
	data_file (file) = Text file whit one JSON object by line.

	Returns:
	pd.DataFrame: Pandas datagrame with all objects of the txt file

	"""
	list_data = []
	with open(data_file, "r") as read_file:
		for line in read_file:
			list_data.append(json.loads(line))
	dataframe = pd.DataFrame(list_data)
	return(dataframe)


def sentimix_data_reorder(sentimix_dataframe):
	"""
	Reorder sentimix dataframe to easy vizualize

	Parameters:
	sentimix_dataframe (pd.DataFrame): Original df

	Returns:
	pd.Dataframe: Ordered dataframe
	"""
	sentimix_dataframe.set_index("tweetid", inplace= True)
	sentimix_dataframe = sentimix_dataframe[["tweet",
											 "tokens",
											 "langid",
											 "sentiment"]]
	return(sentimix_dataframe)


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




sentimixInternetTagging(open_data(CORPUS))
