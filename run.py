import generator
import critic
import tensorflow as tf
from tensorflow.python import debug
from collections import Counter, Iterable
import numpy as np
import os
import string
from os import path
from nltk.translate import bleu_score
from tqdm import tqdm
import json
import time
import logging
import socket
import random

def load_file(filepath):
    with open(filepath, 'r') as fp:
        data = json.load(fp)
    return data

def padding(seqs, max_len=None, pos='post', default=0, dtype=np.int32):
    '''
        make a list of (maybe nested) setargetnce a numpy n-D array
        args:
            seqs: inconsistent setargetnces, rank of which >= 2
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

class DataLoader(object):
    def __init__(self, data_folder, train_file, test_file, source_file, target_file, source_emb_file, target_emb_file):
        self.train_data = np.array(load_file(path.join(data_folder, train_file)))
        self.test_data = np.array(load_file(path.join(data_folder, test_file)))
        self.source = load_file(path.join(data_folder, source_file))
        self.source_len = np.array([len(line) for line in self.source])
        self.target = load_file(path.join(data_folder, target_file))
        self.target_len = np.array([len(line) for line in self.target])
        self.source_embedding = np.load(path.join(data_folder, source_emb_file))
        self.target_embedding = np.load(path.join(data_folder, target_emb_file))

    def split_train(self, batch_size):
        data_size = len(self.train_data)
        num_batch = data_size * 9 // (10*batch_size)
        train_size = num_batch * batch_size
        valid_size = data_size - train_size
        valid_idx = np.random.choice(data_size, valid_size, replace=False)
        train_idx = np.setdiff1d(np.arange(data_size), valid_idx)
        self._train_set = sorted(train_idx, key=lambda x: len(self.source[self.train_data[x][0]]))
        self._train_set = [self._train_set[batch_size*i:batch_size*(i+1)] for i in range(num_batch)]
        self._valid_set = sorted(valid_idx, key=lambda x: len(self.source[self.train_data[x][0]]))
        return num_batch

    def train_batch(self):
        train_size = self.train_size
        np.random.shuffle(self._train_set)
        for batch in self._train_set:
            np.random.shuffle(batch)
            batch_data = self.train_data[batch]
            source_id, target_id, ans_start, ans_len = batch_data.T
            source = padding([self.source[id] for id in source_id])
            source_len = self.source_len[source_id]
            target = padding([self.target[id] for id in target_id])
            target_len = self.target_len[target_id]
            yield source, source_len, target, target_len, ans_start, ans_len

    def valid_batch(self, batch_size=None):
        valid_size = self.valid_size
        if batch_size is None:
            valid_set = [self._valid_set]
        else:
            num_batch = valid_size // batch_size + 1
            valid_set = [self._valid_set[batch_size*i:min(batch_size*(i+1), valid_size)] for i in range(num_batch)]
        for batch in valid_set:
            batch_data = self.train_data[batch]
            source_id, target_id, ans_start, ans_len = batch_data.T
            source = padding([self.source[id] for id in source_id])
            source_len = self.source_len[source_id]
            target = padding([self.target[id] for id in target_id])
            target_len = self.target_len[target_id]
            yield source, source_len, target, target_len, ans_start, ans_len

    def test_batch(self, batch_size=None):
        test_set = sorted(np.arange(self.test_size), key=lambda x: len(self.source[self.test_data[x][0]]))
        if batch_size is None:
            test_set = [test_set]
        else:
            num_batch = self.test_size // batch_size + 1
            test_set = [test_set[batch_size*i:min(batch_size*(i+1), self.test_size)] for i in range(num_batch)]
        for batch in test_set:
            batch_data = self.test_data[batch]
            source_id, target_id, ans_start, ans_len = batch_data.T
            source = padding([self.source[id] for id in source_id])
            source_len = self.source_len[source_id]
            target = padding([self.target[id] for id in target_id])
            target_len = self.target_len[target_id]
            yield source, source_len, target, target_len, ans_start, ans_len

    @property
    def train_size(self):
        if not hasattr(self, '_train_set'):
            raise AttributeError('Please first call `self.split_train(batch_size)` for cross-validation.')
        return sum(len(batch) for batch in self._train_set)

    @property
    def valid_size(self):
        if not hasattr(self, '_valid_set'):
            raise AttributeError('Please first call `self.split_train(batch_size)` for cross-validation.')
        return len(self._valid_set)

    @property
    def test_size(self):
        return self.test_data.shape[0]

def compute_f1(source, pred, truth):
    ignore_set = set(string.punctuation) | set(['an', 'a', 'the'])
    score = 0.0
    for s, p, t in zip(source, pred, truth):
        pr = [SOURCE_VOCAB[idx] for idx in s[p[0]:p[1]] if SOURCE_VOCAB[idx] not in ignore_set]
        tr = [SOURCE_VOCAB[idx] for idx in s[t[0]:t[1]] if SOURCE_VOCAB[idx] not in ignore_set]
        common_num = sum((Counter(pr) & Counter(tr)).values())
        if common_num != 0:
            precision = 1.0*common_num / len(pred)
            recall = 1.0*common_num / len(truth)
            f1 = (2 * precision * recall) / (precision + recall)
            score += f1
    return score * 100 / len(source)

def compute_bleu(generation, target, source, dump=False):
    generated_str = []
    target_str = []
    score = 0.0
    trans = lambda index, ref: TARGET_VOCAB[index] if index < TARGET_VOCAB_SIZE\
            else SOURCE_VOCAB[ref[index-TARGET_VOCAB_SIZE]]
    for gen_row, tgt_row, src_row in zip(generation, target, source):
        gen = [trans(index, src_row) for index in gen_row if index > 0]
        tgt = [trans(index, src_row) for index in tgt_row if index > 0]
        generated_str.append(' '.join(gen))
        target_str.append(' '.join(tgt))
        score += bleu_score.sentence_bleu([tgt], gen, smoothing_function=bleu_score.SmoothingFunction().method1)
    if dump:
        with open(FLAGS.neg_file, 'w') as fp:
            fp.write('\n'.join(generated_str))
        with open(FLAGS.pos_file, 'w') as fp:
            fp.write('\n'.join(target_str))
    return score*100/len(source)

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

def main(_):
    loader = DataLoader(DATA_FOLDER, TRAIN_FILE, TEST_FILE, SOURCE_FILE, TARGET_FILE, SOURCE_EMB_FILE, TARGET_EMB_FILE)
    graph = tf.Graph()
    with graph.as_default():
        #with tf.variable_scope('Generator'):
        #    gtr = generator.G_net(SOURCE_VOCAB_SIZE, TARGET_VOCAB_SIZE, FLAGS.enc_dim, FLAGS.dec_dim, FLAGS.att_dim,\
        #            FLAGS.beam, FLAGS.lr, FLAGS.max_gnorm, loader.source_embedding, loader.target_embedding)
        with tf.variable_scope('Critic'):
            ctc = critic.C_net(SOURCE_VOCAB_SIZE, TARGET_VOCAB_SIZE, FLAGS.enc_dim, FLAGS.match_dim, FLAGS.lr,\
                    FLAGS.max_gnorm, loader.source_embedding, loader.target_embedding)
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
                log_device_placement=True, allow_soft_placement=True)
        sess = tf.Session(graph=graph, config=config)
        saver = tf.train.Saver()
        if FLAGS.train:
            train_writer = tf.summary.FileWriter(path.join(LOG_FOLDER, 'critic'), sess.graph)
            logger = logging.getLogger('TrainLogger')
            logger.setLevel(logging.INFO)
            handler = logging.FileHandler(path.join(LOG_FOLDER, 'train.log'), mode='w', encoding='utf-8')
            handler.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter('%(asctime)s:\t%(message)s'))
            logger.addHandler(handler)

            loss = best_score = 0.0
            #kws = ['source', 'source_length', 'target', 'target_length', 'answer_start', 'answer_length']
            kws = ['document', 'doc_length', 'question', 'que_length', 'ans_start', 'ans_length']
            start_time = time.time()
            if FLAGS.cont:
                saver.restore(sess, path.join(CKPT_FOLDER, 'critic', 'best_model'))
            else:
                sess.run(tf.global_variables_initializer())
            graph.finalize()
            num_batch = loader.split_train(FLAGS.batch)
            global_step = 0
            for epoch in range(FLAGS.epoch_num):
                logger.info('Start training epoch %d.' % (epoch+1))
                for data in tqdm(loader.train_batch(), desc='Training Epoch {}'.format(epoch+1), total=num_batch):
                    feed_dict = dict(zip(kws, data))
                    if (global_step+1) % FLAGS.summary_step == 0:
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        feed_dict['options'] = run_options
                        feed_dict['metadata'] = run_metadata
                    #summaries, step_loss, global_step = gtr.pretrain(sess, **feed_dict)
                    step_loss, global_step, _, summaries = ctc.train(sess, **feed_dict)
                    loss += step_loss
                    logger.info('Loss at step %d: %.6f' % (global_step, step_loss))
                    if global_step % FLAGS.summary_step == 0:
                        loss_avg = loss / FLAGS.summary_step
                        print('At previous %d global steps, loss = %.6f' % (global_step, loss_avg))
                        print('Training time %.2f'%(time.time()-start_time))
                        save_summary(train_writer, dict(Loss=loss_avg), global_step)
                        train_writer.add_summary(summaries, global_step)
                        train_writer.add_run_metadata(run_metadata, 'step%d'%global_step, global_step)
                        loss = 0
                    if global_step % FLAGS.valid_step == 0:
                        total_batch = loader.valid_size // FLAGS.batch + 1
                        generated_seqs = []
                        ground_truth = []
                        ground_source = []
                        for data in tqdm(loader.valid_batch(FLAGS.batch), desc='Validation:', total=total_batch):
                            feed_dict = dict(zip(kws, data))
                            span_s, span_e = ctc.infer(sess, **feed_dict)
                            generated_seqs.extend(zip(span_s, span_e))
                            ground_truth.extend(zip(data[4], data[4]+data[5]))
                            ground_source.extend(data[0])
                        score = compute_f1(source, span, ans)

                        #for data in tqdm(loader.valid_batch(FLAGS.batch), desc='Validation', total=total_batch):
                        #    feed_dict = dict(zip(kws, data))
                        #    generated = gtr.generate(sess, **feed_dict)
                        #    generated_seqs.extend(generated)
                        #    ground_truth.extend(data[2])
                        #    ground_source.extend(data[0])
                        #score = compute_bleu(generated_seqs, ground_truth, ground_source, dump=False)
                        if score > best_score:
                            saver.save(sess, path.join(CKPT_FOLDER, 'critic', 'best_model'))
                            logger.info('Validation score %.2f' % score)
                            print('New model with BLEU score %.2f saved' % score)
                            best_score = score
            print('Training finished. It takes %.2fs' % (time.time()-start_time))
            train_writer.close()
        else:
            if not FLAGS.cont:
                raise ValueError('Must start from checkpoint in evaluation mode')
            saver.restore(sess, path.join(CKPT_FOLDER, 'best_model'))
            kws = ['source', 'source_length', 'target', 'target_length', 'answer_start', 'answer_length']
            total_batch = loader.test_size // FLAGS.batch + 1
            generated_seqs = []
            ground_truth = []
            ground_source = []
            for data in tqdm(loader.test_batch(FLAGS.batch), desc='Test', total=total_batch):
                feed_dict = dict(zip(kws, data))
                generated = gtr.generate(sess, **feed_dict)
                generated_seqs.extend(generated)
                ground_truth.extend(data[2])
                ground_source.extend(data[0])
            score = compute_bleu(generated_seqs, ground_truth, ground_source, dump=True)
            print('Test BLEU Score: %.2f' % score)
        sess.close()

if __name__ == '__main__':
    FLAGS = tf.flags.FLAGS
    tf.flags.DEFINE_boolean('train', True, 'train mode')
    tf.flags.DEFINE_boolean('cont', False, 'continue training from checkpoint')
    tf.flags.DEFINE_integer('epoch_num', 5, 'number of training epochs')
    tf.flags.DEFINE_integer('batch', 16, 'batch size')
    tf.flags.DEFINE_integer('enc_dim', 250, 'hidden dimensions of the encoder')
    tf.flags.DEFINE_integer('dec_dim', 500, 'hidden dimensions of the decoder')
    tf.flags.DEFINE_integer('att_dim', 100, 'dimensions of attention matrix')
    tf.flags.DEFINE_integer('match_dim', 20, 'dimensions of multi-lateral similarity')
    tf.flags.DEFINE_integer('beam', 32, 'beam size of decoder')
    tf.flags.DEFINE_integer('summary_step', 500, 'record summary every ... steps')
    tf.flags.DEFINE_integer('valid_step', 1000, 'do validation every ... steps')
    tf.flags.DEFINE_float('lr', 1e-3, 'initial learning rate')
    tf.flags.DEFINE_float('max_gnorm', 5.0, 'max tolerable global gradient norm')
    tf.flags.DEFINE_string('pos_file', 'pos.txt', 'positive file')
    tf.flags.DEFINE_string('neg_file', 'neg.txt', 'negative file')

    hostname = socket.gethostname()
    if hostname == 'xiaoyong-XPS-8500':
        ROOT = '/home/x_jin/workspace/nlg'
    elif hostname == 'DeepLearning':
        ROOT = '/home/xiaoyong/nlg'
    DATA_FOLDER = path.join(ROOT, 'data')
    CKPT_FOLDER = path.join(ROOT, 'checkpoints')
    LOG_FOLDER = path.join(ROOT, 'log')
    TRAIN_FILE = 'train.json'
    TEST_FILE = 'test.json'
    SOURCE_FILE = 'source_index.json'
    SOURCE_VOCAB_FILE = 'source_vocab.json'
    SOURCE_EMB_FILE = 'source_emb.npy'
    TARGET_FILE = 'target_index.json'
    TARGET_VOCAB_FILE = 'target_vocab.json'
    TARGET_EMB_FILE = 'target_emb.npy'

    reverse = lambda d: dict((v, k) for k, v in d.items())
    SOURCE_VOCAB = reverse(load_file(path.join(DATA_FOLDER, SOURCE_VOCAB_FILE)))
    TARGET_VOCAB = reverse(load_file(path.join(DATA_FOLDER, TARGET_VOCAB_FILE)))
    SOURCE_VOCAB_SIZE = len(SOURCE_VOCAB)
    TARGET_VOCAB_SIZE = len(TARGET_VOCAB)

    tf.app.run()
