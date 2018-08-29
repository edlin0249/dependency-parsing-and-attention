import tensorflow as tf 

class GRURNN(object):
    """
    This is a model class
    """
    def __init__(self, n_steps, output_size, cell_size, batch_size, voc_size, embedding_size, lr):
        """
        This function to initialize the model constructor

        Args:
            n_steps(int) : max text length
            output_size(int) : final classification
            cell_size(int) : GRU cell and others 
            batch_size(int) : batch size of training data
            voc_size(int) : how many tokens
            embedding_size(int) : a dimension for representing one token
            lr (float) : a learning for Adam 
        """
        self.n_steps = n_steps
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.voc_size = voc_size
        self.embedding_size = embedding_size
        self.lr = lr
        self.final_activation_func = tf.nn.softmax
        with tf.name_scope('inputs'):
            """
            build required placeholder
            """
            self.premises_normal = tf.placeholder(tf.int32, [None, self.n_steps], name='premises_normal')
            self.premises_preorder = tf.placeholder(tf.int32, [None, self.n_steps], name='premises_preorder')
            self.premises_postorder = tf.placeholder(tf.int32, [None, self.n_steps], name='premises_postorder')
            self.hypotheses_normal = tf.placeholder(tf.int32, [None, self.n_steps], name='hypotheses_normal')
            self.hypotheses_preorder = tf.placeholder(tf.int32, [None, self.n_steps], name='hypotheses_preorder')
            self.hypotheses_postorder = tf.placeholder(tf.int32, [None, self.n_steps], name='hypotheses_postorder')
            self.premises_preordersentidx = tf.placeholder(tf.int32, [None, self.n_steps, 2], name='premises_preordersentidx')
            self.premises_postordersentidx = tf.placeholder(tf.int32, [None, self.n_steps, 2], name='premises_postordersentidx')
            self.hypotheses_preordersentidx = tf.placeholder(tf.int32, [None, self.n_steps, 2], name='hypotheses_preordersentidx')
            self.hypotheses_postordersentidx = tf.placeholder(tf.int32, [None, self.n_steps, 2], name='hypotheses_postordersentidx')
            self.premises_len = tf.placeholder(tf.int32, [None], name='premises_len')
            self.hypotheses_len = tf.placeholder(tf.int32, [None], name='hypotheses_len')
            self.labels = tf.placeholder(tf.float32, [None, self.output_size], name='labels')

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("text-embedding"):
            """
            build embeddings for original, preorder and postorder sequences of premises and hypotheses
            """
            self.W_text = tf.Variable(tf.random_uniform([self.voc_size, self.embedding_size], -1.0, 1.0), name="W_text")
            self.premises_normal_embed = tf.nn.embedding_lookup(self.W_text, self.premises_normal)        
            self.premises_preorder_embed = tf.nn.embedding_lookup(self.W_text, self.premises_preorder)
            self.premises_postorder_embed = tf.nn.embedding_lookup(self.W_text, self.premises_postorder)

            self.hypotheses_normal_embed = tf.nn.embedding_lookup(self.W_text, self.hypotheses_normal)
            self.hypotheses_preorder_embed = tf.nn.embedding_lookup(self.W_text, self.hypotheses_preorder)
            self.hypotheses_postorder_embed = tf.nn.embedding_lookup(self.W_text, self.hypotheses_postorder)

            # self.premises_normal_embed = tf.expand_dims(self.premises_normal_embed, 0)
            # self.premises_preorder_embed = tf.expand_dims(self.premises_preorder_embed, 0)
            # self.premises_postorder_embed = tf.expand_dims(self.premises_postorder_embed, 0)
            # self.hypotheses_normal_embed = tf.expand_dims(self.hypotheses_normal_embed, 0)
            # self.hypotheses_preorder_embed = tf.expand_dims(self.hypotheses_preorder_embed, 0)
            # self.hypotheses_postorder_embed = tf.expand_dims(self.hypotheses_postorder_embed, 0)
        with tf.name_scope('premises_biGRU_cell'):
            self.add_premises_biGRU_cell()
        with tf.name_scope('premises_prepostorderGRU_cell'):
            self.add_premises_preorderGRU_cell()
        with tf.name_scope('premises_postorderGRU_cell'):
            self.add_premises_postorderGRU_cell()
        with tf.name_scope('hypotheses_biGRU_cell'):
            self.add_hypotheses_biGRU_cell()
        with tf.name_scope('hypotheses_preorder_cell'):
            self.add_hypotheses_preorderGRU_cell()
        with tf.name_scope('hypotheses_postorder_cell'):
            self.add_hypotheses_postorderGRU_cell()
        with tf.name_scope('premises_fusion_hidden'):
            self.add_premises_fusion_layer()
        with tf.name_scope('hypotheses_fusion_hidden'):
            self.add_hypotheses_fusion_layer()
        with tf.name_scope('attention'):
            self.add_attention()
        with tf.name_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('loss'):
            self.compute_loss()
        with tf.name_scope('accuracy'):
            self.compute_accuracy()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy)
            # optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            # gvs = optimizer.compute_gradients(self.cross_entropy)
            # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            # self.train_op = optimizer.apply_gradients(capped_gvs)

    def add_premises_biGRU_cell(self):
        """
        This function to apply biGRU to original sequences of premises
        """
        with tf.variable_scope('premises_forward'):
            fw_gru_cell = tf.contrib.rnn.GRUCell(self.cell_size, name='premises_fw_gru_cell')
        with tf.variable_scope('premises_backward'):
            bw_gru_cell = tf.contrib.rnn.GRUCell(self.cell_size, name='premises_bw_gru_cell')
  
        outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_gru_cell, bw_gru_cell, self.premises_normal_embed, sequence_length=self.premises_len, time_major=False, dtype=tf.float32)
        self.output_premises_fw, self.output_premises_bw = outputs[0], outputs[1]
        self.output_premises_state_fw, self.output_premises_state_bw = states[0], states[1]
        tf.summary.histogram('premises_forward/output_premises_fw', self.output_premises_fw)
        tf.summary.histogram('premises_backward/output_premises_bw', self.output_premises_bw)

    def add_premises_preorderGRU_cell(self):
        """
        This function to apply GRU to preorder sequences of premises
        """
        with tf.variable_scope('premises_preorderGRU_forward'):
            preordergru_cell = tf.contrib.rnn.GRUCell(self.cell_size, name='premises_preordergru_cell')
        self.output_premises_preorder, self.output_premises_preorder_state = tf.nn.dynamic_rnn(preordergru_cell, self.premises_preorder_embed, sequence_length=self.premises_len, time_major=False, dtype=tf.float32)
        tf.summary.histogram('premises_preorderGRU_forward/output_premises_preorder', self.output_premises_preorder)

    def add_premises_postorderGRU_cell(self):
        """
        This function to apply GRU to postorder sequences of premises
        """
        with tf.variable_scope('premises_postorderGRU_forward'):
            postordergru_cell = tf.contrib.rnn.GRUCell(self.cell_size, name='premises_postordergru_cell')
        self.output_premises_postorder, self.output_premises_postorder_state = tf.nn.dynamic_rnn(postordergru_cell, self.premises_postorder_embed, sequence_length=self.premises_len, time_major=False, dtype=tf.float32)
        tf.summary.histogram('premises_postorderGRU_forward/output_premises_postorder', self.output_premises_postorder)

    def add_hypotheses_biGRU_cell(self):
        """
        This function to apply biGRU to original sequences of hypotheses
        """
        with tf.variable_scope('hypotheses_forward'):
            fw_gru_cell = tf.contrib.rnn.GRUCell(self.cell_size, name='hypotheses_fw_gru_cell')
        with tf.variable_scope('hypotheses_backward'):
            bw_gru_cell = tf.contrib.rnn.GRUCell(self.cell_size, name='hypotheses_bw_gru_cell')
        outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_gru_cell, bw_gru_cell, self.hypotheses_normal_embed, sequence_length=self.hypotheses_len, time_major=False, dtype=tf.float32)
        self.output_hypotheses_fw, self.output_hypotheses_bw = outputs[0], outputs[1]
        self.output_hypotheses_state_fw, self.output_hypotheses_state_bw = states[0], states[1]
        tf.summary.histogram('hypotheses_forward/output_hypotheses_fw', self.output_hypotheses_fw)
        tf.summary.histogram('hypotheses_backward/output_hypotheses_bw', self.output_hypotheses_bw)

    def add_hypotheses_preorderGRU_cell(self):
        """
        This function to apply GRU to preorder sequences of hypotheses
        """
        with tf.variable_scope('hypotheses_preorderGRU_forward'):
            gru_cell = tf.contrib.rnn.GRUCell(self.cell_size, name='hypotheses_preordergru_cell')
        self.output_hypotheses_preorder, self.output_hypotheses_preorder_state = tf.nn.dynamic_rnn(gru_cell, self.hypotheses_preorder_embed, sequence_length=self.hypotheses_len, time_major=False, dtype=tf.float32)
        tf.summary.histogram('hypotheses_preorderGRU_forward/output_hypotheses_preorder', self.output_hypotheses_preorder)

    def add_hypotheses_postorderGRU_cell(self):
        """
        This function to apply GRU to postorder sequences of hypotheses
        """
        with tf.variable_scope('hypotheses_postorderGRU_forward'):
            gru_cell = tf.contrib.rnn.GRUCell(self.cell_size, name='hypotheses_postordergru_cell')
        self.output_hypotheses_postorder, self.output_hypotheses_postorder_state = tf.nn.dynamic_rnn(gru_cell, self.hypotheses_postorder_embed, sequence_length=self.hypotheses_len, time_major=False, dtype=tf.float32)
        tf.summary.histogram('hypotheses_postorderGRU_forward/output_hypotheses_postorder', self.output_hypotheses_postorder)         	

    def add_premises_fusion_layer(self):
        """
        This function to merge information of orginal, preorder and postorder sequences of premises together
        """
        Ws_normal_fw = self._weight_variable([self.cell_size, self.cell_size], name='premises_Ws_normal_fw')
        Ws_normal_bw = self._weight_variable([self.cell_size, self.cell_size], name='premises_Ws_normal_bw')
        Ws_preorder = self._weight_variable([self.cell_size, self.cell_size], name='premises_Ws_preorder')
        Ws_postorder = self._weight_variable([self.cell_size, self.cell_size], name='premises_Ws_postorder')
        # self.output_premises_fw = tf.reshape(self.output_premises_fw, [-1, self.cell_size])
        # self.output_premises_bw = tf.reshape(self.output_premises_bw, [-1, self.cell_size])
        # self.output_premises_preorder = tf.reshape(self.output_premises_preorder, [-1, self.cell_size])
        # self.output_premises_postorder = tf.reshape(self.output_premises_postorder, [-1, self.cell_size])
        # self.output_premises_preorder = tf.gather(self.output_premises_preorder, self.premises_preordersentidx)
        # self.output_premises_postorder = tf.gather(self.output_premises_postorder, self.premises_postordersentidx)
        self.output_premises_preorder = tf.gather_nd(self.output_premises_preorder, self.premises_preordersentidx)
        self.output_premises_postorder = tf.gather_nd(self.output_premises_postorder, self.premises_postordersentidx)
        h_fw = tf.tensordot(self.output_premises_fw, Ws_normal_fw, 1)
        h_bw = tf.tensordot(self.output_premises_bw, Ws_normal_bw, 1)
        h_preorder = tf.tensordot(self.output_premises_preorder, Ws_preorder, 1)
        h_postorder = tf.tensordot(self.output_premises_postorder, Ws_postorder, 1)
        self.premises_h_fusion = h_fw + h_bw + h_preorder + h_postorder
        self.premises_h_fusion = tf.nn.tanh(self.premises_h_fusion, name='premises_fusion')
        tf.summary.histogram('premises_fusion/premises_h_fusion', self.premises_h_fusion)

        
    def add_hypotheses_fusion_layer(self):
        """
        This function to merge information of original, preorder and postorder sequences of hypotheses together
        """
        Ws_normal_fw = self._weight_variable([self.cell_size, self.cell_size], name='hypotheses_Ws_normal_fw')
        Ws_normal_bw = self._weight_variable([self.cell_size, self.cell_size], name='hypotheses_Ws_normal_bw')
        Ws_preorder = self._weight_variable([self.cell_size, self.cell_size], name='hypotheses_Ws_preorder')
        Ws_postorder = self._weight_variable([self.cell_size, self.cell_size], name='hypotheses_Ws_postorder')
        # self.output_hypotheses_fw = tf.reshape(self.output_hypotheses_fw, [-1, self.cell_size])
        # self.output_hypotheses_bw = tf.reshape(self.output_hypotheses_bw, [-1, self.cell_size])
        # self.output_hypotheses_preorder = tf.reshape(self.output_hypotheses_preorder, [-1, self.cell_size])
        # self.output_hypotheses_postorder = tf.reshape(self.output_hypotheses_postorder, [-1, self.cell_size])
        # self.output_hypotheses_preorder = tf.gather(self.output_hypotheses_preorder, self.hypotheses_preordersentidx)
        # self.output_hypotheses_postorder = tf.gather(self.output_hypotheses_postorder, self.hypotheses_postordersentidx)
        self.output_hypotheses_preorder = tf.gather_nd(self.output_hypotheses_preorder, self.hypotheses_preordersentidx)
        self.output_hypotheses_postorder = tf.gather_nd(self.output_hypotheses_postorder, self.hypotheses_postordersentidx)
        h_fw = tf.tensordot(self.output_hypotheses_fw, Ws_normal_fw, 1)
        h_bw = tf.tensordot(self.output_hypotheses_bw, Ws_normal_bw, 1)
        h_preorder = tf.tensordot(self.output_hypotheses_preorder, Ws_preorder, 1)
        h_postorder = tf.tensordot(self.output_hypotheses_postorder, Ws_postorder, 1)
        self.hypotheses_h_fusion = h_fw + h_bw + h_preorder + h_postorder
        self.hypotheses_h_fusion = tf.nn.tanh(self.hypotheses_h_fusion, name='hypotheses_fusion')
        tf.summary.histogram('hypotheses_fusion/hypotheses_h_fusion', self.hypotheses_h_fusion)


    def add_attention(self):
        """
        This function to do inter-attention of premises and hypotheses
        """

        # self.premises_h_fusion = tf.reshape(self.premises_h_fusion, [-1, self.n_steps, self.cell_size])
        # self.hypotheses_h_fusion = tf.reshape(self.hypotheses_h_fusion, [-1, self.n_steps, self.cell_size])

        attention_matrix = tf.matmul(self.premises_h_fusion, tf.transpose(self.hypotheses_h_fusion, perm=[0, 2, 1]))
        attentionSoft_premises = tf.nn.softmax(attention_matrix, axis=2)
        attentionSoft_hypotheses = tf.nn.softmax(attention_matrix, axis=1)
        attentionSoft_hypotheses = tf.transpose(attentionSoft_hypotheses, perm=[0,2,1])
        self.premises_attns = tf.matmul(attentionSoft_premises, self.hypotheses_h_fusion)
        self.hypotheses_attns = tf.matmul(attentionSoft_hypotheses, self.premises_h_fusion)
        # print(self.premises_len.shape)
        # print(self.hypotheses_len.shape)
        # premises_h_fusion_slice = tf.slice(self.premises_h_fusion, [0, 0], [self.premises_len[0], self.cell_size])
        # hypotheses_h_fusion_slice = tf.slice(self.hypotheses_h_fusion, [0, 0], [self.hypotheses_len[0], self.cell_size])
        # attention_matrix = tf.slice(attention_matrix, [0, 0], [self.premises_len[0], self.hypotheses_len[0]])
        # attention_matrix = tf.exp(attention_matrix)
        # along_hypotheses_sum = tf.reduce_sum(attention_matrix, 1, keepdims=True)
        # hypotheses_normal = tf.div(attention_matrix, along_hypotheses_sum)
        # hypotheses_normal = tf.reshape(hypotheses_normal, [self.premises_len[0], self.hypotheses_len[0], -1])
        # hypotheses_h_fusion_tile = tf.tile(hypotheses_h_fusion_slice, [self.premises_len[0], 1])
        # hypotheses_h_fusion_tile = tf.reshape(hypotheses_h_fusion_tile, [self.premises_len[0], self.hypotheses_len[0], self.cell_size])

        # self.premises_attns = tf.multiply(hypotheses_h_fusion_tile, hypotheses_normal)
        # self.premises_attns = tf.reduce_sum(self.premises_attns, 1)
        
        # along_premises_sum = tf.reduce_sum(attention_matrix, 0, keepdims=True)
        # premises_normal = tf.div(attention_matrix, along_premises_sum)
        # premises_normal = tf.reshape(premises_normal, [self.premises_len[0], self.hypotheses_len[0], -1])
        # premises_h_fusion_tile = tf.tile(premises_h_fusion_slice, [1, self.hypotheses_len[0]])
        # premises_h_fusion_tile = tf.reshape(premises_h_fusion_tile, [self.premises_len[0], self.hypotheses_len[0], self.cell_size])

        # self.hypotheses_attns = tf.multiply(premises_h_fusion_tile, premises_normal)
        # self.hypotheses_attns = tf.reduce_sum(self.hypotheses_attns, 0)
        tf.summary.histogram('attention/premises_attns', self.premises_attns)
        tf.summary.histogram('attention/hypotheses_attns', self.hypotheses_attns)


    def add_output_layer(self):
        """
        This function to apply output layer after local inference block
        """
        premises_diff = tf.subtract(self.premises_h_fusion, self.premises_attns)
        premises_mul = tf.multiply(self.premises_h_fusion, self.premises_attns)
        hypotheses_diff = tf.subtract(self.hypotheses_h_fusion, self.hypotheses_attns)
        hypotheses_mul = tf.multiply(self.hypotheses_h_fusion, self.hypotheses_attns)

        premises_sum = tf.reduce_sum(self.premises_h_fusion, 1)
        premises_mean = tf.div(premises_sum, tf.cast(tf.expand_dims(self.premises_len, -1), tf.float32))
        premises_attns_sum = tf.reduce_sum(self.premises_attns, 1)
        premises_attns_mean = tf.div(premises_attns_sum, tf.cast(tf.expand_dims(self.premises_len, -1), tf.float32))
        premises_diff_sum = tf.reduce_sum(premises_diff, 1)
        premises_diff_mean = tf.div(premises_diff_sum, tf.cast(tf.expand_dims(self.premises_len, -1), tf.float32))
        premises_mul_sum = tf.reduce_sum(premises_mul, 1)
        premises_mul_mean = tf.div(premises_mul_sum, tf.cast(tf.expand_dims(self.premises_len, -1), tf.float32))
        hypotheses_sum = tf.reduce_sum(self.hypotheses_h_fusion, 1)
        hypotheses_mean = tf.div(hypotheses_sum, tf.cast(tf.expand_dims(self.hypotheses_len, -1), tf.float32))
        hypotheses_attns_sum = tf.reduce_sum(self.hypotheses_attns, 1)
        hypotheses_attns_mean = tf.div(hypotheses_attns_sum, tf.cast(tf.expand_dims(self.hypotheses_len, -1), tf.float32))
        hypotheses_diff_sum = tf.reduce_sum(hypotheses_diff, 1)
        hypotheses_diff_mean = tf.div(hypotheses_diff_sum, tf.cast(tf.expand_dims(self.hypotheses_len, -1), tf.float32))
        hypotheses_mul_sum = tf.reduce_sum(hypotheses_mul, 1)
        hypotheses_mul_mean = tf.div(hypotheses_mul_sum, tf.cast(tf.expand_dims(self.hypotheses_len, -1), tf.float32))
        # premises_diff = tf.subtract(premises_mean, premises_attns_mean)
        # hypotheses_diff = tf.subtract(hypotheses_mean, hypotheses_attns_mean)

        # premises_mul = tf.multiply(premises_mean, premises_attns_mean)
        # hypotheses_mul = tf.multiply(hypotheses_mean, hypotheses_attns_mean)
        prem_hypo = tf.concat([premises_mean, premises_attns_mean, premises_diff_mean, premises_mul_mean, hypotheses_mean, hypotheses_attns_mean, hypotheses_diff_mean, hypotheses_mul_mean], 1)
        # print(prem_hypo.shape)
        Ws_out = self._weight_variable([self.cell_size*8, self.output_size], name='Ws_out')
        bs_out = self._bias_variable([self.output_size], name='bs_out')
        # output_1 = tf.nn.relu(tf.tensordot(prem_hypo, Ws_out_1, 1) + bs_out_1)
        # Ws_out_2 = self._weight_variable([self.cell_size*2, self.output_size], name='Ws_out_2')
        # bs_out_2 = self._bias_variable([self.output_size], name='bs_out_2')
        # self.logits = self.final_activation_func(tf.tensordot(prem_hypo, Ws_out, 1) + bs_out)
        self.logits = tf.tensordot(prem_hypo, Ws_out, 1) + bs_out
        

    def compute_loss(self):
        """
        This function to compute cross entropy loss
        """
        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels))
            # self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.labels*tf.log(tf.clip_by_value(self.logits,1e-10,1.0)),1))
            # self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.labels*tf.log(self.logits),1))
            tf.summary.scalar('loss', self.cross_entropy)

    def compute_accuracy(self):
        """
        This function to compute accuracy
        """
        with tf.name_scope("accuracy"):
            self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.labels, 1), tf.argmax(self.logits,1)), tf.float32))
            tf.summary.scalar('acc', self.acc)
            

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)
