from DependencyParsing import DependencyParsing
from Voc import Voc

class Preprocessor:
    """
    This is a class ror preprocessor
    """
    def __init__(self, dpTree, voc, max_text_length):
        """
        This function to initialize a preprocessor constructor

        Args:
           dpTree : a DependencyParsing object
           voc : a Voc object
           max_text_length : int for max text length
        """   	
        self.dpTree = dpTree
        self.voc = voc
        self.max_text_length = max_text_length

    def sent2idx(self, line):
        """
        This function to convert tokens of texts to indexes in the vocabulary

        Args:
            line : a string of text

        Returns:
            idxofsent : a list of int
        """
        tokens = self.dpTree.tokenize(line)
        idxofsent = [0]*self.max_text_length
        for idx_t, token in enumerate(tokens):
            if token in self.voc.token_idx:
                idxofsent[idx_t] = self.voc.token_idx[token]
            else:
                idxofsent[idx_t] = self.voc.token_idx[self.voc.UNK_token]
        return idxofsent

    def ordersent2idx(self, line, line_idx):
        """
        This function to convert texts to indexes of tokens in the vocabulary after preorder DFS or postorder DFS

        Args:
            line : a string of text
            line_idx : a list of int by preorder DFS or postorder DFS

        Returns:
            idxofsent : a list of int 
        """
        tokens = self.dpTree.tokenize(line)
        order_toks = []
        for idx in line_idx:
            order_toks.append(tokens[idx])
        idxofsent = [0]*self.max_text_length
        for idx_t, token in enumerate(order_toks):
            if token in self.voc.token_idx:
                idxofsent[idx_t] = self.voc.token_idx[token]
            else:
                idxofsent[idx_t] = self.voc.token_idx[self.voc.UNK_token]
        return idxofsent

    def order2sentidx(self, line_idx):
        """
        This function to convert indexes of texts after preorder DFS or postorder DFS to indexes of original texts
        
        Args:
            line_idx : a list of int by preorder DFS or postorder DFS

        Returns:
            sentidx : a list of indexes used to convert preorder sequences or postorder sequences to original sequences
        """
        sentidx = [0]*self.max_text_length
        for idx, val in enumerate(line_idx):
            sentidx[val] = idx
        for idx in range(len(line_idx), self.max_text_length):
            sentidx[idx] = idx
        return sentidx

    def labelonehots(self, labels):
        """
        This function to make labels a one-hots type

        Args:
            labels : a string

        Returns:
            onehots : a list of one-hots label

        """

        label_idx = {'entailment':0, 'neutral':1, 'contradiction':2, '-':1}
        onehots = [0]*3
        onehots[label_idx[labels]] = 1
        return onehots

def batchnize(train_datas_idx, train_datas, batch_size):
    """
    This generator function to batch training datas

    Args:
        train_datas_idx : a list of int of random indexes of original training data sequence
        train_datas : a list of training data
        batch_size : a int

    # Returns:
    #    train_datas_batch : a list of lists of random batched training datas

    """
    start = 0
    end = batch_size
    train_datas_random = []
    for idx in train_datas_idx:
        train_datas_random.append(train_datas[idx])


    train_datas_batch = []
    while start < len(train_datas):
        train_data_one_batch = train_datas_random[start:end]
        premises_normal_batch = []
        premises_preorder_batch = []
        premises_postorder_batch = []
        hypotheses_normal_batch = []
        hypotheses_preorder_batch = []
        hypotheses_postorder_batch = []
        premises_preordersentidx_batch = []
        premises_postordersentidx_batch = []
        hypotheses_preordersentidx_batch = []
        hypotheses_postordersentidx_batch = []
        premises_len_batch = []
        hypotheses_len_batch = []
        labels_batch = []
        for idx, train_data in enumerate(train_data_one_batch):
            premises_normal_batch.append(train_data[0])
            premises_preorder_batch.append(train_data[1])
            premises_postorder_batch.append(train_data[2])
            hypotheses_normal_batch.append(train_data[3])
            hypotheses_preorder_batch.append(train_data[4])
            hypotheses_postorder_batch.append(train_data[5])
            one_premise_preordersentidx = []
            one_premise_postordersentidx = []
            one_hypothese_preordersentidx = []
            one_hypothese_postordersentidx = []
            for premise_preordersentidx, premise_postordersentidx, hypothese_preordersentidx, hypothese_postordersentidx in zip(train_data[6], train_data[7], train_data[8], train_data[9]):
                one_premise_preordersentidx.append([idx, premise_preordersentidx])
                one_premise_postordersentidx.append([idx, premise_postordersentidx])
                one_hypothese_preordersentidx.append([idx, hypothese_preordersentidx])
                one_hypothese_postordersentidx.append([idx, hypothese_postordersentidx])
            premises_preordersentidx_batch.append(one_premise_preordersentidx)
            premises_postordersentidx_batch.append(one_premise_postordersentidx)
            hypotheses_preordersentidx_batch.append(one_hypothese_preordersentidx)
            hypotheses_postordersentidx_batch.append(one_hypothese_postordersentidx)
            premises_len_batch.append(train_data[10][0])
            hypotheses_len_batch.append(train_data[11][0])
            labels_batch.append(train_data[12])
        yield [premises_normal_batch, premises_preorder_batch, premises_postorder_batch, hypotheses_normal_batch, hypotheses_preorder_batch, hypotheses_postorder_batch, premises_preordersentidx_batch, premises_postordersentidx_batch, hypotheses_preordersentidx_batch, hypotheses_postordersentidx_batch, premises_len_batch, hypotheses_len_batch, labels_batch] 
        # train_datas_batch.append([premises_normal_batch, premises_preorder_batch, premises_postorder_batch, hypotheses_normal_batch, hypotheses_preorder_batch, hypotheses_postorder_batch, premises_preordersentidx_batch, premises_postordersentidx_batch, hypotheses_preordersentidx_batch, hypotheses_postordersentidx_batch, premises_len_batch, hypotheses_len_batch, labels_batch])
        start = end
        end += batch_size    
    # return train_datas_batch
