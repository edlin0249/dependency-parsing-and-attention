import nltk
import re, string
import jsonlines

class DataLoader(object):
    """
    This is a class for loading data file
    """
    def __init__(self, data_dir):
        """
        The constructor for DataLoader class

        Args : 
            data_dir (string) : Data directory
        """
        self.data_dir = data_dir
        # self.data_file = self.data_dir + '/multinli_0.9_train.jsonl'
        

    def load(self, filename):
        """
        The function to load data from multinli_0.9_train.jsonl

        Returns :
            datas (list): a list of dict

        """
        data_file = self.data_dir + filename
        datas = []
        with jsonlines.open(data_file, 'r') as reader:
            for obj in reader:
                datas.append(obj)

        return datas

    def remove_puncts(self, lines):
        """
        The function to remove punctions from lines

        Args :
            lines : a list of string, including punctions

        Returns :
            lines_ : a list of string, excluding punctions

        """
        lines_ = []
        for line in lines:
            line = line.strip()
            regex = re.compile('[%s]' % re.escape(string.punctuation))
            line = regex.sub('', line)
            lines_.append(line)
        return lines_

    def sent2pairs_label(self, datas):
        """
        The function to load data from multinli_0.9_train.jsonl

        Args :
            datas (list): a list of dict

        Returns :
            pairs_label (list): a list of a pairs of texts in addition to a label

        """   
        pairs_label = []
        for data in datas:
            pairs_label.append([data['sentence1'], data['sentence2'], data['gold_label']])
        return pairs_label

    def tokenize(self, lines):
        """
        The function to tokenize

        Args :
            lines (list) : a list which contains many texts

        Returns :
            tokens (list) : a list which contains many tokens
        """
        tokens = []
        for line in lines:
            tokens.append(nltk.word_tokenize(line))
        return tokens



