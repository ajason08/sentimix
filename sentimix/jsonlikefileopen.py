import json
import pandas as pd

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
