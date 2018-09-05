class HES(object):
    """
    This is  a class for head enriched structure(HES) 
    """
    def __call__(self, dependency):
        """head enriched structure(HES)
        Args: 
            dependency : a list of (dependency, token_num, token_num)

        Returns:
            all_preorder : a list of postorder result
       
        
        Example:
            text = 'Guangdong University of Foreign Studies is located in Guangzhou.' 
            => dependency = [('ROOT', 0, 7), ('compound', 2, 1), ('nsubjpass', 7, 2), ('case', 5, 3), ('compound', 5, 4), ('nmod', 2, 5), ('auxpass', 7, 6)    , ('case', 9, 8), ('nmod', 7, 9), ('punct', 7, 10)]
            => all_postorder = [0, 2, 3, 4, 1, 5, 7, 8, 9, 6]
        """
        n_roots = [idx for idx, ele in enumerate(dependency) if ele[1] == 0]    # one premise or hypothese may have a lot of dependency parsing trees, i.e. different roots
        all_postorder = []
        for idx_root in range(len(n_roots)):
            if idx_root == len(n_roots)-1:
                postorder_list = self.postorder(dependency[n_roots[idx_root]:], 0)
            else:
                postorder_list = self.postorder(dependency[n_roots[idx_root]:n_roots[idx_root+1]-1], 0)
            postorder_list = postorder_list[:-1]
            len_all_postorder = len(all_postorder)
            for ele in postorder_list:
                all_postorder.append(len_all_postorder+(ele-1))       # revert indexes of dependency parsing trees to ones of original texts
        return all_postorder

    def postorder(self, dependency, root):
        """postorder by DFS
        Args:
            dependency : a sublist of primary dependency
            root : 0

        Returns:
            traversal : a list of int by DFS
        
        Example:
            dependency(as __call__ method description)
            => traversal = [1, 3, 4, 5, 2, 6, 8, 9, 10, 7, 0]
        """ 
        is_leaf = 1  # default each root as leaf node
        for item in dependency:  # search the list once to know whether the root still has children
            if item[1] == root:
                is_leaf = 0

        if is_leaf:    # if the root is a leaf node, return the root; otherwise, not.
            return [root]

        traversal = []
        for item in dependency:
            if item[1] == root:
                traversal += self.postorder(dependency, item[2])
        traversal += [root]
        return traversal    # return children  




        
