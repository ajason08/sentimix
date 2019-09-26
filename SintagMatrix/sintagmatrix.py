def get_position(sentence, token):
    """
    Takes a sentece as a list of ordered tokens and return
    a list of positions of the token in the sentense, (an
    empty list if token does not appears in sentence)

    Parameters:
    sentence (lst): Sentence as a list of tokens.
    token (str): Token as string

    Returns
    lst(int):  A integers list, one of each represent one position
    of the token in the sentence.
    """

    positions = [i for i, x enumerate(sentence) if x==element]
