import os
import argparse
from DataLoader import DataLoader
from DependencyParsing import DependencyParsing
from CES import CES
from HES import HES
from Voc import Voc
from Preprocessor import Preprocessor, batchnize
from GRU import GRURNN
import tensorflow as tf
import tqdm
import numpy as np 
import random
import logging
import pdb
# from tensorflow.python import debug as tf_debug

def CalculateLength(line, dpTree):
    """
    This function to calculate text length
    """
    return len(dpTree.tokenize(line))


def Args():
    """
    This function to acquire required hyperparameters
    """
    parser = argparse.ArgumentParser(description='Implement Improved Neural Machine Translation with Source Syntax')
    parser.add_argument('-data_dir', default='../multinli_0.9/')
    parser.add_argument('-train_data_file_name', default='multinli_0.9_train.jsonl', help='filename for training set')
    parser.add_argument('-dev_data_file_name', default='multinli_0.9_dev_matched.jsonl', help='filename for development set')
    parser.add_argument('-test_data_file_name', default='multinli_0.9_test_matched_unlabeled.jsonl', help='filename for testing set')
    parser.add_argument('-epochs', default=30, type=int, help='epochs for training')
    parser.add_argument('-cell_size', default=128, type=int, help='cell size for GRU')
    parser.add_argument('-output_size', default=3, type=int, help='output size for output layer')
    parser.add_argument('-batch_size', default=100, type=int, help='batch size for training data')
    parser.add_argument('-embedding_size', default=128, type=int, help='embedding size for word2vec')
    parser.add_argument('-lr', default=0.0001, type=float, help='learning rate for Adam optimizer')
    parser.add_argument('-display_interval', default=500, type=int, help='per display_interval times of updating for displaying')
    parser.add_argument('-early_stopping_steps', default=3, type=int, help='bearable steps for early stopping')
    args = parser.parse_args()
    return args



def train(args):
    logging.basicConfig(filename='./log.txt', filemode='w', level=logging.WARNING, format='%(asctime)s %(levelname)-8s %(message)s')
    dataset = DataLoader(args.data_dir)
    dpTree = DependencyParsing()

    ces = CES()
    hes = HES()
    voc = Voc()

    max_text_length = 0
    for data_file_name in [args.train_data_file_name, args.dev_data_file_name, args.test_data_file_name]:
        datas = dataset.load(data_file_name)
        pairs_label = dataset.sent2pairs_label(datas)
        max_text_length = max([CalculateLength(item[0], dpTree) for item in pairs_label] + [CalculateLength(item[1], dpTree) for item in pairs_label] + [max_text_length])

    print('max_text_length = %d'%(max_text_length))
    # preparing training set

    datas = dataset.load(args.train_data_file_name)
    pairs_label = dataset.sent2pairs_label(datas)

    sent1_all_preorder = []
    sent1_all_postorder = []
    sent2_all_preorder = []
    sent2_all_postorder = []
    logging.warning("Starting dependencyparsing and tokenizing ...")
    for idx, item in enumerate(pairs_label):
        dependency1 = dpTree.dependencyparsing(item[0])
        sent1_all_preorder.append(ces(dependency1))
        sent1_all_postorder.append(hes(dependency1))

        dependency2 = dpTree.dependencyparsing(item[1])
        sent2_all_preorder.append(ces(dependency2))
        sent2_all_postorder.append(hes(dependency2))

        sent1_tokens = dpTree.tokenize(item[0])
        sent2_tokens = dpTree.tokenize(item[1])

        voc.build_idx2tok(sent1_tokens)
        voc.build_idx2tok(sent2_tokens)

        if (idx+1) % args.display_interval == 0:
            logging.warning('{}/{} is over...'.format(idx+1, len(pairs_label)))


    voc.build_tok2idx()
    print('voc size = %d'%(len(voc)))
    pdb.set_trace()
    #preprocess suitable input
    preprocessor = Preprocessor(dpTree, voc, max_text_length)
    train_datas = []

    logging.warning("Starting preprocessing ...")
    for item, sent1_preorder, sent1_postorder, sent2_preorder, sent2_postorder in zip(pairs_label, sent1_all_preorder, sent1_all_postorder, sent2_all_preorder, sent2_all_postorder):
        data = []
        data.append(preprocessor.sent2idx(item[0]))
        data.append(preprocessor.ordersent2idx(item[0], sent1_preorder))
        data.append(preprocessor.ordersent2idx(item[0], sent1_postorder))

        data.append(preprocessor.sent2idx(item[1]))
        data.append(preprocessor.ordersent2idx(item[1], sent2_preorder))
        data.append(preprocessor.ordersent2idx(item[1], sent2_postorder))

        data.append(preprocessor.order2sentidx(sent1_preorder))
        data.append(preprocessor.order2sentidx(sent1_postorder))
        data.append(preprocessor.order2sentidx(sent2_preorder))
        data.append(preprocessor.order2sentidx(sent2_postorder))

        data.append([len(sent1_preorder)])
        data.append([len(sent2_preorder)])

        data.append(preprocessor.labelonehots(item[2]))

        train_datas.append(data)


    # preparing development set
    datas = dataset.load(args.dev_data_file_name)
    pairs_label = dataset.sent2pairs_label(datas)

    sent1_all_preorder = []
    sent1_all_postorder = []
    sent2_all_preorder = []
    sent2_all_postorder = []
    logging.warning("Starting dependencyparsing and tokenizing ...")
    for idx, item in enumerate(pairs_label):
        dependency1 = dpTree.dependencyparsing(item[0])
        sent1_all_preorder.append(ces(dependency1))
        sent1_all_postorder.append(hes(dependency1))

        dependency2 = dpTree.dependencyparsing(item[1])
        sent2_all_preorder.append(ces(dependency2))
        sent2_all_postorder.append(hes(dependency2))

        if (idx+1) % args.display_interval == 0:
            logging.warning('{}/{} is over...'.format(idx+1, len(pairs_label)))

    dev_datas = []
    logging.warning("Starting preprocessing ...")
    for item, sent1_preorder, sent1_postorder, sent2_preorder, sent2_postorder in zip(pairs_label, sent1_all_preorder, sent1_all_postorder, sent2_all_preorder, sent2_all_postorder):
        data = []
        data.append(preprocessor.sent2idx(item[0]))
        data.append(preprocessor.ordersent2idx(item[0], sent1_preorder))
        data.append(preprocessor.ordersent2idx(item[0], sent1_postorder))

        data.append(preprocessor.sent2idx(item[1]))
        data.append(preprocessor.ordersent2idx(item[1], sent2_preorder))
        data.append(preprocessor.ordersent2idx(item[1], sent2_postorder))

        data.append(preprocessor.order2sentidx(sent1_preorder))
        data.append(preprocessor.order2sentidx(sent1_postorder))
        data.append(preprocessor.order2sentidx(sent2_preorder))
        data.append(preprocessor.order2sentidx(sent2_postorder))

        data.append([len(sent1_preorder)])
        data.append([len(sent2_preorder)])

        data.append(preprocessor.labelonehots(item[2]))

        dev_datas.append(data)

    # start training
    # build GRURNN

    model = GRURNN(max_text_length, args.output_size, args.cell_size, args.batch_size, len(voc), args.embedding_size, args.lr)
    saver = tf.train.Saver(max_to_keep=0)
    config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)

    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(tf.global_variables_initializer())
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    dev_total_acc = 0
    dev_total_loss = 0
    dev_total_best_acc = 0
    dev_total_best_loss = 0
    stopping_steps = 0

    logging.warning("Starting training ...")
    for epoch in range(args.epochs):
        logging.warning('epoch : {} / {}'.format(epoch+1, args.epochs))
        train_datas_idx = random.sample(range(len(train_datas)), len(train_datas))
        for idx, batch in enumerate(batchnize(train_datas_idx, train_datas, args.batch_size)):
            
            # initialize data
            feed_dict = {
                model.premises_normal: batch[0],
                model.premises_preorder: batch[1],
                model.premises_postorder: batch[2],
                model.hypotheses_normal: batch[3],
                model.hypotheses_preorder: batch[4],
                model.hypotheses_postorder: batch[5],
                model.premises_preordersentidx: batch[6],
                model.premises_postordersentidx: batch[7],
                model.hypotheses_preordersentidx: batch[8],
                model.hypotheses_postordersentidx: batch[9],
                model.premises_len: batch[10],
                model.hypotheses_len: batch[11],
                model.labels: batch[12]
            }
            # training
            train_rs, _, train_loss, train_acc, train_logits = sess.run(
                [merged, model.train_op, model.cross_entropy, model.acc, model.logits],
                feed_dict=feed_dict)
            writer.add_summary(train_rs, idx)
            
            # cal_num += 1
            # total_loss += loss

            # print("{}, {}".format(np.argmax(train_datas[data_idx][12]), np.argmax(logits)))
            # if np.argmax(train_datas[data_idx][12]) == np.argmax(logits):
            #     acc += 1

            # print(logits)
            # print loss 
            if (idx+1) % args.display_interval == 0:
                logging.warning("idx : {} , cross entropy = {} , acc = {}".format(idx+1, train_loss, train_acc))
                # cal_num = 0
                # total_loss = 0
                # acc = 0

        dev_datas_idx = list(range(len(dev_datas)))
        for idx, batch in enumerate(batchnize(dev_datas_idx, dev_datas, args.batch_size)):
            feed_dict = {
                model.premises_normal: batch[0],
                model.premises_preorder: batch[1],
                model.premises_postorder: batch[2],
                model.hypotheses_normal: batch[3],
                model.hypotheses_preorder: batch[4],
                model.hypotheses_postorder: batch[5],
                model.premises_preordersentidx: batch[6],
                model.premises_postordersentidx: batch[7],
                model.hypotheses_preordersentidx: batch[8],
                model.hypotheses_postordersentidx: batch[9],
                model.premises_len: batch[10],
                model.hypotheses_len: batch[11],
                model.labels: batch[12]    
            }
            dev_loss, dev_acc, dev_logits = sess.run(
                [model.cross_entropy, model.acc, model.logits],
                feed_dict=feed_dict)

            dev_total_acc += dev_acc*len(batch[0])
            # logging.warning('dev_total_acc = {} , dev_acc = {} , len(batch[0]) = {}'.format(dev_total_acc, dev_acc, len(batch[0])))
        dev_total_acc /= len(dev_datas)
        logging.warning('epoch : {}, dev_total_acc = {}'.format(epoch+1, dev_total_acc))
        if dev_total_acc > dev_total_best_acc:
            stopping_steps = 0
            dev_total_best_acc = dev_total_acc  
            save_path = saver.save(sess, "model/net.ckpt", global_step=epoch+1)
        else:
            stopping_steps += 1

        if stopping_steps >= args.early_stopping_steps:
            logging.warning('Early stopping is trigger at epoch : {}, best development set accuracy : {}'.format(epoch+1, dev_total_best_acc))
            break

        dev_total_acc = 0


if __name__ == '__main__':
    args = Args()
    train(args)
