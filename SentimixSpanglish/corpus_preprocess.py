import pandas as pd
import json
from internet_taggin import internet_element_taggin, taggin_numbers


TRIAL = "./data/spanglish_trial_release.txt"
TRAIN = "./data/train.tsv"

def open_data_JsonPerLine(data_file):
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

def open_data_Horrible_TSV(horrible_data_file):
        dataframe = pd.read_csv(horrible_data_file,
                                encoding="utf_8",
                                sep="\t",
                                header=None,
                                names=["words","lang"])
        dataframe = dataframe.fillna(value = "_____")
        lastrow  = len(dataframe)-1
        positives = dataframe[dataframe['lang'] == "positive"]
        negatives = dataframe[dataframe['lang'] == "negative"]
        neutral = dataframe[dataframe['lang'] == "neutral"]
        concatena = pd.concat([positives,negatives, neutral])
        phrase_limits = concatena.sort_index().index.tolist()
        start = 0
        phrase_df = pd.DataFrame(columns=["tweet","languages", "sentiment"])
        for limit in phrase_limits:
                sentiment = dataframe.iloc[limit]["lang"]
                print (dataframe.iloc[start:limit])
                sentence = "".join([dataframe.iloc[start:limit]["words"]])
        return sentence

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

def trigrams(text):
        pass


print(open_data_Horrible_TSV(TRAIN))
