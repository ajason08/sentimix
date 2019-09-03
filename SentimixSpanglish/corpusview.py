import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
"""
Culumnas de datos:
- langid
- sentiment
- tokens
- tweet
- tweeid
"""

def open_data(data_file):
    list_data = []
    with open(data_file, "r") as read_file:
        for line in read_file:
            list_data.append(json.loads(line))
    dataframe = pd.DataFrame(list_data)
    dataframe.set_index("tweetid", inplace= True)
    return(dataframe)


print(open_data("./data/spanglish_trial_release.txt").info())
