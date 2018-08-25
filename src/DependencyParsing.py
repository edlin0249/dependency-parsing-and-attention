from stanfordcorenlp import StanfordCoreNLP

class DependencyParsing:
    """
    This is a class for dependency parsing
    """
    def __init__(self):
        """
        The function to initialize the DependencyParsing constructor
        """
        self.nlp = StanfordCoreNLP('/home/edlin0249/iis_summer_intern/stanford-corenlp-full-2018-02-27')

    def dependencyparsing(self, line):
        """
        The function to do dependency parsing
        
        Args:
            line : a string of text

        Returns:
            nlp.dependency_parsing : a list of (dependency, token_num, token_num)

        Example:
            line = 'Guangdong University of Foreign Studies is located in Guangzhou.'
            => nlp.dependency_parsing : [('ROOT', 0, 7), ('compound', 2, 1), ('nsubjpass', 7, 2), ('case', 5, 3), ('compound', 5, 4), ('nmod', 2, 5), ('auxpass', 7, 6), ('case', 9, 8), ('nmod', 7, 9), ('punct', 7, 10)]
        """
        return self.nlp.dependency_parse(line)

    def tokenize(self, line):
        """
        The function to tokenize

        Args:
            line : a string of text

        Returns :
            nlp.word_tokenize(line) : a list of string

        Example:
            line = 'Guangdong University of Foreign Studies is located in Guangzhou.'
            => nlp.word_tokenize : ['Guangdong', 'University', 'of', 'Foreign', 'Studies', 'is', 'located', 'in', 'Guangzhou', '.'] 
        """
        return self.nlp.word_tokenize(line)
