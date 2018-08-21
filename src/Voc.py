PAD_idx = 0
PAD_token = '<PAD>'

class Voc:
    """
    This is a constructor for building Voc 
    """
    def __init__(self):
        """
        This funciton to initialize constructor for Voc
        """
        self.idx_token = [PAD_token]

    def build_idx2tok(self, tokens):
        """
        This function to build a idx-to-token relationship

        Args:
            tokens : a string

        """
        for token in tokens:
            if token not in self.idx_token:
                self.idx_token.append(token)

    def build_tok2idx(self):
        """
        This function to build a token-to-idx relationship
        """ 
        self.token_idx = {token:idx for idx, token in enumerate(self.idx_token)}

    def __len__(self):
        """
        This function to get Voc size
        """
        return len(self.idx_token)

