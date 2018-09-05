class CES(object):
    """
    This is a class for child enriched structure(CES)
    """
    def __call__(self, dependency):
        """child enriched structure(CES)
        Args: 
            dependency : a list of (dependency, token_num, token_num)

        Returns:
            all_preorder : a list of preorder result
       
        
        Example:
            text = 'Guangdong University of Foreign Studies is located in Guangzhou.' 
            => dependency = [('ROOT', 0, 7), ('compound', 2, 1), ('nsubjpass', 7, 2), ('case', 5, 3), ('compound', 5, 4), ('nmod', 2, 5), ('auxpass', 7, 6)    , ('case', 9, 8), ('nmod', 7, 9), ('punct', 7, 10)]
            => all_preorder = [6, 1, 0, 4, 2, 3, 5, 8, 7, 9]
        """
        n_roots = [idx for idx, ele in enumerate(dependency) if ele[1] == 0]    # one premise or hypothese may have a lot of dependency parsing trees, i.e. different roots
        all_preorder = []
        for idx_root in range(len(n_roots)):
            if idx_root == len(n_roots)-1:
                preorder_list = self.preorder(dependency[n_roots[idx_root]:], 0)
            else:
                preorder_list = self.preorder(dependency[n_roots[idx_root]:n_roots[idx_root+1]-1], 0)
            preorder_list = preorder_list[1:]
            len_all_preorder = len(all_preorder)
            for ele in preorder_list:
                all_preorder.append(len_all_preorder+(ele-1))    # revert indexes of dependency parsing trees to ones of original texts 
        return all_preorder

  
    def preorder(self, dependency, root): 
        """preorder by DFS
        Args:
            dependency : a sublist of primary dependency
            root : 0

        Returns:
            traversal : a list of int by DFS

        Example:
            dependency(as __call__ method description)
            => traversal = [0, 7, 2, 1, 5, 3, 4, 6, 9, 8, 10]
        """
        is_leaf = 1  # default each root as leaf node
        for item in dependency:  # search the list once to know whether the root still has children
            if item[1] == root:
                is_leaf = 0

        if is_leaf:    # if the root is a leaf node, return the root; otherwise, not.
            return [root]

        traversal = [root]
        for item in dependency:
            if item[1] == root:
                traversal += self.preorder(dependency, item[2])
        return traversal   # return children




        
