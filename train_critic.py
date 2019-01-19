from anser import Answerer
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from os import path
import json
import time
import string
import logging
import socket
from collections import Iterable, Counter

def padding(seqs, max_len=None, pos='post', default=0, dtype=np.int32):
    '''
        make a list of (maybe nested) sequence a numpy n-D array
        args:
            seqs: inconsistent sequences, rank of which >= 2
            max_len: expected length on the first dimension, max of all sequence lengths by default
            pos: padding/truncating mode, at the head or the tail
            default: the default minimum element, 0 by default
            dtype: the type of minimum element
    '''
    if not isinstance(seqs, Iterable):
        raise ValueError('seqs must be iterable.')
    if any([not isinstance(seq, Iterable) for seq in seqs]):
        raise ValueError('seqs must have at least rank of 2_')
    if all([all([isinstance(elem, Iterable) for elem in seq]) for seq in seqs]):
        max_sub_len = max([max([len(elem) for elem in seq], default=0) for seq in seqs], default=0)
        seqs = [padding(seq, max_sub_len, pos, default, dtype) for seq in seqs]
    if max_len is None:
        max_len = max([len(seq) for seq in seqs], default=0)
    seq_mat = []
    for seq in seqs:
        l = len(seq)
        if l > max_len:
            seq_mat.append(seq[:max_len] if pos=='post' else seq[(l-max_len):])
        elif l < max_len:
            try:
                sub_shape = np.shape(seq[0])
            except IndexError:
                sub_shape = ()
            d = np.zeros((max_len-l,)+sub_shape, dtype=dtype)+default
            comps = (seq, d) if pos=='post' else (d, seq)
            seq_mat.append(np.concatenate(comps))
        else:
            seq_mat.append(seq)
    return np.array(seq_mat, dtype=dtype)

def load_file(filepath):
    with open(filepath, 'r') as fp:
        data = json.load(fp)
    return data

def save_summary(summary_writer, data, global_step=None):
    '''
        helper function to write non-tensor statistics to tensorboard
        args:
            summary_writer: tf.summary.FileWriter object
            data: a dict that contains the (scalar) data we want to write into log
            global_step: optional global step
        Note:
            keep the key value of data consistent
    '''
    for key, value in data.items():
        summary = tf.Summary()
        summary.value.add(tag=key, simple_value=value)
        summary_writer.add_summary(summary, global_step)

class DataLoader(object):
    def __init__(self, data_folder, data_file, doc_file, que_file, doc_emb_file, que_emb_file):
        self.data_table = load_file(path.join(data_folder, data_file))
        self.doc = load_file(path.join(data_folder, doc_file))
        self.dlen = np.array([len(line) for line in self.doc])
        self.doc = padding(self.doc)
        self.que = load_file(path.join(data_folder, que_file))
        self.qlen = np.array([len(line) for line in self.que])
        self.que = padding(self.que)
        self.doc_len = self.doc.shape[1]
        self.que_len = self.que.shape[1]
        self.doc_embedding = np.load(path.join(data_folder, doc_emb_file))
        self.que_embedding = np.load(path.join(data_folder, que_emb_file))

    def create_datasets(self, batch_size):
        data_size = len(self.data_table)
        num_batch = data_size * 4 // (5*batch_size)
        train_size = num_batch * batch_size
        valid_size = (data_size-train_size) // 2
        test_size = data_size - train_size - valid_size
        table = np.array(self.data_table)
        idx = np.arange(data_size)
        np.random.shuffle(idx)
        train_idx = idx[:train_size]
        valid_idx = idx[train_size:(train_size+valid_size)]
        test_idx = idx[-test_size:]
        self.train_set = [table[idx] for idx in np.split(train_idx, num_batch)]
        self.valid_set = table[valid_idx]
        self.test_set = table[test_idx]
        return num_batch

    def reshuffle_trainset(self):
        train_data = np.concatenate(self.train_set)
        np.random.shuffle(train_data)
        self.train_set = np.split(train_data, num_batch)

    def batch(self):
        for batch in self.train_set:
            doc_ids, que_ids, ans_starts, ans_lens = [ary.flatten() for ary in np.split(batch, 4, axis=1)]
            docs = self.doc[doc_ids]
            dlen = self.dlen[doc_ids]
            ques = self.que[que_ids]
            qlen = self.qlen[que_ids]
            ans_ends = ans_starts + ans_lens
            yield docs, dlen, ques, qlen, ans_starts, ans_ends

    @property
    def valid_size(self):
        return self.valid_set.shape[0]

    def get_valid(self):
        doc_ids, que_ids, ans_starts, ans_lens = [ary.flatten() for ary in np.split(self.valid_set, 4, axis=1)]
        docs = self.doc[doc_ids]
        dlen = self.dlen[doc_ids]
        ques = self.que[que_ids]
        qlen = self.qlen[que_ids]
        ans_ends = ans_starts + ans_lens
        return docs, dlen, ques, qlen, ans_starts, ans_ends

    @property
    def test_size(self):
        return self.test_set.shape[0]

    def get_test(self):
        doc_ids, que_ids, ans_starts, ans_lens = [ary.flatten() for ary in np.split(self.test_set, 4, axis=1)]
        docs = self.doc[doc_ids]
        dlen = self.dlen[doc_ids]
        ques = self.que[que_ids]
        qlen = self.qlen[que_ids]
        ans_ends = ans_starts + ans_lens
        return docs, dlen, ques, qlen, ans_starts, ans_ends

def f1_score(pred, truth):
    common_num = sum((Counter(pred) & Counter(truth)).values())
    if common_num == 0:
        return 0, 0, 0
    else:
        precision = 1.0*common_num / len(pred)
        recall = 1.0*common_num / len(truth)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1, precision, recall

def main(_):
    loader = DataLoader(DATA_FOLDER, DATA_FILE, SOURCE_FILE, TARGET_FILE, SOURCE_EMB_FILE, TARGET_EMB_FILE)
    with graph.as_default():
        answerer = Answerer(loader.doc_len, loader.que_len, SOURCE_VOCAB_SIZE, TARGET_VOCAB_SIZE,\
                FLAGS.attention_dim, FLAGS.hidden_dim, FLAGS.learning_rate, FLAGS.max_grad_norm, \
                loader.doc_embedding, loader.que_embedding)
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),\
            log_device_placement=True, allow_soft_placement=True)
    sess = tf.Session(config=config, graph=graph)
    writer = tf.summary.FileWriter(LOG_FOLDER)
    saver = tf.train.Saver()

    logger = logging.getLogger('TrainLogger')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(path.join(LOG_FOLDER, 'train'), mode='w')
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s\t%(message)s'))
    logger.addHandler(handler)

    start_time = time.time()
    if FLAGS.cont:
        saver.restore(sess, path.join(CKPT_FOLDER, 'best_model'))
    else:
        sess.run(tf.global_variables_initializer())
    graph.finalize()
    l_avg = 0.0
    num_batch = loader.create_datasets(FLAGS.batch_size)
    doc_valid, dlen_valid, que_valid, qlen_valid, ans_s_valid, ans_e_valid = loader.get_valid()
    doc_test, dlen_test, que_test, qlen_test, ans_s_test, ans_e_test = loader.get_test()
    ignore_set = set(string.punctuation) + set(('an', 'a', 'the'))
    for epoch in range(FLAGS.epoch_num):
        logger.info('Start epoch %d' % (epoch+1))
        for data in tqdm(loader.batch(), desc='Epoch %d' % (epoch+1), total=num_batch):
            doc, dlen, que, qlen, ans_s, ans_e = data
            feed_dict = dict(document=doc, question=que, dlen=dlen, qlen=qlen, ans_s=ans_s, ans_e=ans_e)
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            feed_dict['options'] = run_options
            feed_dict['metadata'] = run_metadata
            loss, gs, _, summ = answerer.train(sess, **feed_dict)
            logger.info('Step %d, loss = %.4f' % (gs, loss))
            l_avg += loss
            if (gs>0) and (gs+1 % FLAGS.valid_step == 0):
                print('Average loss in previous %6d steps:\t%.4f' % (FLAGS.valid_step, l_avg/FLAGS.valid_step))
                l_avg = 0.0
                writer.add_summary(summ, gs)
                writer.add_run_metadata(run_metadata, 'step%d'%gs, gs)
                fd = dict(document=doc_valid, question=que_valid, dlen=dlen_valid, qlen=qlen_valid)
                span_s_valid, span_e_valid = answerer.infer(sess, **fd)
                f1_avg, precision_avg, recall_avg = 0., 0., 0.
                for doc, p_s, p_e, t_s, t_e in zip(doc_valid, span_s_valid, span_e_valid, ans_s_valid, ans_e_valid):
                    pred = [SOURCE_VOCAB[str(idx)] for idx in doc[p_s:p_e] if SOURCE_VOCAB[str(idx)] not in ignore_set]
                    truth = [SOURCE_VOCAB[str(idx)] for idx in doc[t_s:t_e] if SOURCE_VOCAB[str(idx)] not in ignore_set]
                    f1, prec, rec = f1_score(pred, truth)
                    f1_avg += f1/loader.valid_size
                    precision_avg += precision/loader.valid_size
                    recall_avg += recall/loader.valid_size
                save_data = dict(f1_score=f1_avg, precision=precision_avg, recall=recall_avg)
                print('Validation F1 score at step %6d steps:\t%.4f' % (gs, f1_avg))
                logger.info('Validation F1 score at step %6d steps:\t%.4f' % (gs, f1_avg))
                save_summary(writer, save_data, gs)
        loader.reshuffle_trainset()
    print('Training Done! It takes %.2f seconds' % time.time()-start_time)
    logger.info('Training Done! It takes %.2f seconds' % time.time()-start_time)



if __name__ == '__main__':
    FLAGS = tf.flags.FLAGS
    tf.flags.DEFINE_boolean('train', True, 'train mode')
    tf.flags.DEFINE_boolean('cont', False, 'continue training from checkpoint')
    tf.flags.DEFINE_integer('epoch_num', 5, 'number of training epochs')
    tf.flags.DEFINE_integer('batch_size', 16, 'batch size')
    tf.flags.DEFINE_integer('hidden_dim', 250, 'hidden dimensions of the encoder')
    tf.flags.DEFINE_integer('attention_dim', 100, 'dimensions of attention matrix')
    tf.flags.DEFINE_integer('summary_step', 1000, 'record summary every ... steps')
    tf.flags.DEFINE_integer('valid_step', 2000, 'validation after so many steps')
    tf.flags.DEFINE_float('learning_rate', 1e-3, 'initial learning rate')
    tf.flags.DEFINE_float('max_grad_norm', 5.0, 'max-tolerated gradient norm')
    tf.flags.DEFINE_string('pos_file', 'pos.txt', 'positive file')
    tf.flags.DEFINE_string('neg_file', 'neg.txt', 'negative file')

    hostname = socket.gethostname()
    if hostname == 'DeepLearning':
        ROOT = 'home/xiaoyong/nlg'
    else:
        ROOT = 'home/x_jin/workspace/nlg'
    graph = tf.Graph()
    DATA_FOLDER = path.join(ROOT, 'data')
    CKPT_FOLDER = path.join(ROOT, 'checkpoints')
    LOG_FOLDER = path.join(ROOT, 'log')
    DATA_FILE = 'data.json'
    SOURCE_FILE = 'source_index.json'
    SOURCE_VOCAB_FILE = 'source_vocab.json'
    SOURCE_EMB_FILE = 'source_embed.npy'
    TARGET_FILE = 'target_index.json'
    TARGET_VOCAB_FILE = 'target_vocab.json'
    TARGET_EMB_FILE = 'target_embed.npy'

    SOURCE_VOCAB = load_file(path.join(DATA_FOLDER, SOURCE_VOCAB_FILE))
    TARGET_VOCAB = load_file(path.join(DATA_FOLDER, TARGET_VOCAB_FILE))
    SOURCE_VOCAB_SIZE = len(SOURCE_VOCAB)
    TARGET_VOCAB_SIZE = len(TARGET_VOCAB)

