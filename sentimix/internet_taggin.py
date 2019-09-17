import pandas as pd
import re

# Twitter common marks
AT = re.compile(r'@([^{\s}*^{\,\.\!\?\:\;\"}]*)')
HT = re.compile(r'#([^{\s}*^{\,\.\!\?\:\;\"}]*)')
RT = re.compile(r'RT\ @([^{\s}*^{\,\.\!\?\:\;\"}]*):')
LI = re.compile(r"https?://([^\s\\]*)")
NU = re.compile(r"\s([{\d}*{\.\,}]*)\s")

# CamelCase
UpperCamelCase = re.compile(r'\s([A-Z][a-z])*\s')
lowerCamelCase = re.compile(r'\s([A-Z][a-z])*\s')


def internet_element_taggin(text, preserve_con="NotInLinks"):
        """
        Detect and tagg common tweet elements: links, rt, hastag, pics.

        Parameters:
        text (str):

        Return
        str : tagged text
        """
        if preserve_con == True:
                print(text)
                # Its necesary to detect RT first
                tagged_text = re.sub(RT, r"<RT=\1>", text)
                # Now we can catch ats adn hastags:
                tagged_text = re.sub(AT, r"<AT=\1>", tagged_text)
                tagged_text = re.sub(HT, r"<HT=\1>", tagged_text)
                # Links
                tagged_text = re.sub(LI, r"<LINK=\1>", tagged_tex)
        elif preserve_con == "NotInLinks":
                # Its necesary to detect RT first
                tagged_text = re.sub(RT, r"<RT=\1>", text)
                if RT.fullmatch(text):
                        print(text)
                        print(tagged_text)
                # Now we can catch ats adn hastags:
                tagged_text = re.sub(AT, r"<AT=\1>", tagged_text)
                tagged_text = re.sub(HT, r"<HT=\1>", tagged_text)
                # Links
                tagged_text = re.sub(LI, r"<LINK>", tagged_text)
        return tagged_text


def taggin_numbers(text, preserve_con=False):
        """
        Detect and tagg numbers.

        Parameters:
        text (str):

        Return
        str : tagged text
        """
        if preserve_con:
                tagged_text = re.sub(NU,r" <NUMBER=\1> ", text)
        else:
                tagged_text = re.sub(NU,r" <NUMBER> ", text)
        return tagged_text

def split_cammel_case(text):
        camelcases = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
        splitted = [o.group(0) for o in matches]
        joined = " ".join(splitted)
        return joined
