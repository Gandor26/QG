import math
import tensorflow as tf
rnn = tf.nn.rnn_cell

def leaky_relu(alpha=0.0, max_value=None):
    def relu(x):
        with tf.name_scope('LeakyReLU', [x]) as scope:
            neg = tf.nn.relu(-x)
            x = tf.nn.relu(x)
            if max_value is not None:
                x = tf.clip_by_value(x, tf.constant(0, x.dtype), tf.constant(max_value, x.dtype))
            x = tf.subtract(x, neg*alpha, name=scope)
        return x
    return relu

def compute_MP_sim(vec1, vec2, weights=None, name=None):
    '''
        args:
            vec1, vec2: two (sequences of) representations
            weights: multi-perspective weights
            shape(vec1) = batch_size, seq1_len, hidden_dim
            shape(vec2) = batch_size, seq2_len, hidden_dim
            shape(weights) = match_dim, hidden_dim
        return:
            sim: element-wise cosine similarity between v1 and v2 in each group
            shape(sim) = seq1_len, [seq2_len, match_dim]
    '''
    with tf.name_scope(name, 'MultiPerspective_Cosine_Similarity', [vec1, vec2, weights]) as scope:
        if weights:
            expanded_vec1 = tf.expand_dims(vec1, axis=-2, name='expanded_vec1')
            weighted_vec1 = tf.multiply(expanded_vec1, weights, name='weighted_vec1')
            normed_vec1 = tf.nn.l2_normalize(weighted_vec1, dim=-1, name='normed_vec1')
            expanded_vec2 = tf.expand_dims(vec2, axis=-2, name='expanded_vec2')
            weighted_vec2 = tf.multiply(expanded_vec2, weights, name='weighted_vec2')
            normed_vec2 = tf.nn.l2_normalize(weighted_vec2, dim=-1, name='normed_vec2')
            sim = tf.einsum('bpmh,bqmh->bpqm', normed_vec1, normed_vec2)
            sim = tf.identity(sim, name=scope)
        else:
            normed_vec1 = tf.nn.l2_normalize(vec1, dim=-1, name='normalized_vec1')
            normed_vec2 = tf.nn.l2_normalize(vec2, dim=-1, name='normalized_vec2')
            sim = tf.matmul(normed_vec1, normed_vec2, transpose_b=True, name=scope)
    return sim

def inspect(tensor, *targets):
    tensor = tf.Print(tensor, [tensor]+list(targets), message='the value of %s and others' % tensor.name, summarize=100)
    tensor = tf.Print(tensor, [tf.shape(tensor)], message='the shape of %s' % tensor.name, summarize=5)
    return tensor

def leaky_relu(alpha=0.0, max_value=None):
    def relu(x):
        with tf.name_scope('LeakyReLU', [x]) as scope:
            neg = tf.nn.relu(-x)
            x = tf.nn.relu(x)
            if max_value is not None:
                x = tf.clip_by_value(x, tf.constant(0, x.dtype), tf.constant(max_value, x.dtype))
            x = tf.subtract(x, neg*alpha, name=scope)
        return x
    return relu

def compute_MP_sim(vec1, vec2, weights=None, name=None):
    with tf.name_scope(name, 'MultiPerspective_Cosine_Similarity', [vec1, vec2, weights]) as scope:
        if weights:
            expanded_vec1 = tf.expand_dims(vec1, axis=-2, name='expanded_vec1')
            weighted_vec1 = tf.multiply(expanded_vec1, weights, name='weighted_vec1')
            normed_vec1 = tf.nn.l2_normalize(weighted_vec1, dim=-1, name='normed_vec1')
            expanded_vec2 = tf.expand_dims(vec2, axis=-2, name='expanded_vec2')
            weighted_vec2 = tf.multiply(expanded_vec2, weights, name='weighted_vec2')
            normed_vec2 = tf.nn.l2_normalize(weighted_vec2, dim=-1, name='normed_vec2')
            sim = tf.einsum('bpmh,bqmh->bpqm', normed_vec1, normed_vec2)
            sim = tf.identity(sim, name=scope)
        else:
            normed_vec1 = tf.nn.l2_normalize(vec1, dim=-1, name='normalized_vec1')
            normed_vec2 = tf.nn.l2_normalize(vec2, dim=-1, name='normalized_vec2')
            sim = tf.matmul(normed_vec1, normed_vec2, transpose_b=True, name=scope)
    return sim

class C_net(object):
    def __init__(self, doc_vocab_size, que_vocab_size, hidden_dim, match_dim,\
            learning_rate=1e-3, max_grad_norm=None, doc_pretrained_emb=None, que_pretrained_emb=None):
        self.doc_vocab_size = doc_vocab_size
        self.que_vocab_size = que_vocab_size
        self.match_dim = match_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm or 5.0
        self.doc = tf.placeholder(tf.int32, [None, None], name='document')
        self.que = tf.placeholder(tf.int32, [None, None], name='question')
        self.doc_len = tf.placeholder(tf.int32, [None], name='document_length')
        self.doc_maxlen = tf.reduce_max(self.doc_len, name='document_max_length')
        self.que_len = tf.placeholder(tf.int32, [None], name='question_length')
        self.que_maxlen = tf.reduce_max(self.que_len, name='question_max_length')
        self.ans_start = tf.placeholder(tf.int32, [None], name='answer_start')
        self.ans_len = tf.placeholder(tf.int32, [None], name='answer_length')
        self.ans_end = tf.add(self.ans_start, self.ans_len, name='answer_end')
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')

        with tf.variable_scope('Embedding'):
            assert doc_pretrained_emb.shape[1] == que_pretrained_emb.shape[1]
            self.emb_dim = doc_pretrained_emb.shape[1]
            doc_emb = self._build_emb_mat(self.doc_vocab_size, doc_pretrained_emb, name='document')
            que_emb = self._build_emb_mat(self.que_vocab_size, que_pretrained_emb, name='question')
            self.emb = tf.concat([doc_emb, que_emb], 0, name='embeddings')
        #--------------- REPRESENTATION LAYER ----------------#
        cond = tf.less(self.que, self.que_vocab_size)
        x = tf.add(self.que, self.doc_vocab_size)
        idx = tf.clip_by_value(self.que-self.que_vocab_size, 0, self.doc_maxlen-1)
        batches = tf.transpose(tf.range(tf.shape(self.que)[0]) + tf.transpose(tf.zeros_like(self.que)))
        y = tf.gather_nd(self.doc, tf.stack((batches, idx), -1))
        real_que = tf.where(cond, x, y)
        doc_emb = tf.nn.embedding_lookup(self.emb, self.doc)
        que_emb = tf.nn.embedding_lookup(self.emb, real_que)
        #------------------- FILTER LAYER --------------------#
        r_mat = compute_MP_sim(doc_emb, que_emb, name='relevance_matrix')
        filtered_doc_emb = tf.multiply(tf.reduce_max(r_mat, axis=2, keep_dims=True), doc_emb, name='filtered_doc')
        #------------------- CONTEXT LAYER -------------------#
        with tf.variable_scope('Context') as vs:
            cell_fw = rnn.BasicLSTMCell(self.hidden_dim)
            cell_bw = rnn.BasicLSTMCell(self.hidden_dim)
            (que_repr_fw, que_repr_bw), (que_hidden_fw, que_hidden_bw) = tf.nn.bidirectional_dynamic_rnn(\
                cell_fw, cell_bw, que_emb, self.que_len, dtype=tf.float32, scope=vs)
            vs.reuse_variables()
            (doc_repr_fw, doc_repr_bw), (doc_hidden_fw, doc_hidden_bw) = tf.nn.bidirectional_dynamic_rnn(\
                cell_fw, cell_bw, filtered_doc_emb, self.doc_len, dtype=tf.float32, scope=vs)
        que_hidden_fw = que_hidden_fw.h
        que_hidden_bw = que_hidden_bw.h
        doc_hidden_fw = doc_hidden_fw.h
        doc_hidden_bw = doc_hidden_bw.h
        #------------------- MATCH LAYER ---------------------#
        match = list()
        with tf.variable_scope('Matching'):
            #----------------- FUll MATCH PHASE  -----------------#
            full_weights_fw = tf.get_variable('full_weights_fw', [self.match_dim, self.hidden_dim], tf.float32)
            full_weights_bw = tf.get_variable('full_weights_bw', [self.match_dim, self.hidden_dim], tf.float32)
            que_hidden_fw = tf.expand_dims(que_hidden_fw, axis=1, name='question_last_hidden_fw')
            que_hidden_bw = tf.expand_dims(que_hidden_bw, axis=1, name='question_last_hidden_bw')
            full_match_fw = compute_MP_sim(doc_repr_fw, que_hidden_fw, full_weights_fw, name='full_MPCM_fw')
            full_match_bw = compute_MP_sim(doc_repr_bw, que_hidden_bw, full_weights_bw, name='full_MPCM_bw')
            full_match_fw = tf.squeeze(full_match_fw, axis=2, name='full_match_fw')
            full_match_bw = tf.squeeze(full_match_bw, axis=2, name='full_match_bw')
            match.append(full_match_fw)
            match.append(full_match_bw)
            #-------------- MAXPOOLING MATCH PHASE ---------------#
            max_weights_fw = tf.get_variable('max_weights_fw', [self.match_dim, self.hidden_dim], tf.float32)
            max_match_fw = compute_MP_sim(doc_repr_fw, que_repr_fw, max_weights_fw, name='max_MPCM_fw')
            max_match_fw = tf.reduce_max(max_match_fw, axis=2, name='max_match_fw')
            max_weights_bw = tf.get_variable('max_weights_bw', [self.match_dim, self.hidden_dim], tf.float32)
            max_match_bw = compute_MP_sim(doc_repr_bw, que_repr_bw, max_weights_bw, name='max_MPCM_bw')
            max_match_bw = tf.reduce_max(max_match_bw, axis=2, name='max_match_bw')
            match.append(max_match_fw)
            match.append(max_match_bw)
            #--------------- ATTENTION MATCH PHASE ---------------#
            mean_weights_fw = tf.get_variable('mean_weights_fw', [self.match_dim, self.hidden_dim], tf.float32)
            mean_match_fw = compute_MP_sim(doc_repr_fw, que_repr_fw, mean_weights_fw, name='mean_MPCM_fw')
            mean_match_fw = tf.reduce_mean(mean_match_fw, axis=2, name='mean_match_fw')
            mean_weights_bw = tf.get_variable('mean_weights_bw', [self.match_dim, self.hidden_dim], tf.float32)
            mean_match_bw = compute_MP_sim(doc_repr_bw, que_repr_bw, mean_weights_bw, name='mean_MPCM_bw')
            mean_match_bw = tf.reduce_mean(mean_match_bw, axis=2, name='mean_match_bw')
            match.append(mean_match_fw)
            match.append(mean_match_bw)
        match = tf.concat(match, axis=-1, name='matching')
        #---------------- AGGREGATION LAYER ------------------#
        with tf.variable_scope('Aggregate') as vs:
            cell_fw = rnn.BasicLSTMCell(self.hidden_dim)
            cell_bw = rnn.BasicLSTMCell(self.hidden_dim)
            aggr_repr_fb, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, match,\
                        self.doc_len, dtype=tf.float32, scope=vs)
        aggr_repr = tf.concat(aggr_repr_fb, axis=-1, name='aggregation')
        #----------------- PREDICTION LAYER ------------------#
        len_mask = tf.sequence_mask(self.doc_len, self.doc_maxlen, name='length_mask')
        with tf.variable_scope('Predict_start'):
            pred_weight_s = tf.get_variable('prediction_weight', [self.hidden_dim*2], tf.float32)
            logits_s = tf.reduce_sum(tf.multiply(aggr_repr, pred_weight_s), axis=-1,\
                    name='logits')
            pred_s = tf.where(len_mask, tf.nn.log_softmax(logits_s), -1e4*tf.ones_like(logits_s), name='prediction')
            loss_s = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_s, labels=self.ans_start,\
                    name='loss')
        with tf.variable_scope('Predict_end'):
            pred_weight_e = tf.get_variable('prediction_weight', [self.hidden_dim*2], tf.float32)
            logits_e = tf.reduce_sum(tf.multiply(aggr_repr, pred_weight_e), axis=-1,\
                    name='logits')
            pred_e = tf.where(len_mask, tf.nn.log_softmax(logits_e), -1e4*tf.ones_like(logits_e), name='prediction')
            loss_e = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_e, labels=self.ans_end,\
                    name='loss')
        self.loss = tf.reduce_mean(loss_s+loss_e, name='loss')
        tf.summary.scalar('Loss', self.loss)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads, vars = zip(*optimizer.compute_gradients(self.loss))
        if self.max_grad_norm is None:
            clipped_grads = grads
        else:
            clipped_grads, self.grads_norm = tf.clip_by_global_norm(grads, self.max_grad_norm, name='grads_norm')
        self.opt = optimizer.apply_gradients(zip(clipped_grads, vars), self.global_step, name='optimization')

        pred = tf.add(tf.expand_dims(pred_s, axis=2), tf.expand_dims(pred_e, axis=1), name='joint_likelihood')
        reg_mask = lambda pred: tf.where(tf.sequence_mask(tf.range(self.doc_maxlen)+1, self.doc_maxlen),\
                -1e4*tf.ones_like(pred), pred)
        masked_pred = tf.map_fn(reg_mask, pred)
        self.max_logit, idx = tf.nn.top_k(tf.reshape(pred, [-1, self.doc_maxlen**2]))
        idx = tf.squeeze(idx, axis=1)
        self.start, self.end = tf.div(idx, self.doc_maxlen, name='s_idx'), tf.mod(idx, self.doc_maxlen, name='e_idx')

        self.summaries = tf.summary.merge_all()


    def _build_emb_mat(self, vocab_size, emb_mat, name):
        iv_num = emb_mat.shape[0]
        oov_num = vocab_size - iv_num
        with tf.variable_scope('Embedding') as vs:
            iv_emb = tf.get_variable(name+'_iv_emb', emb_mat.shape, tf.float32,\
                                     initializer=tf.constant_initializer(emb_mat), trainable=False)
            oov_emb = tf.get_variable(name+'_oov_emb', (oov_num, self.emb_dim), tf.float32,\
                                     initializer=tf.random_uniform_initializer(-.5, .5))
            emb = tf.concat([iv_emb, oov_emb], axis=0, name=name+'_emb_matrix')
        return emb

    def train(self, sess, **kwargs):
        '''
            Train the model.
            Required variables:
                document:   [batch_size, doc_maxlen] of int32, document tokens
                question:   [batch_size, que_maxlen] of int32, question tokens
                doc_length: [batch_size] of int32, actual length of document
                que_length: [batch_size] of int32, actual length of question
                ans_start:  [batch_size] of int32, answer starting position
                ans_length: [batch_size] of int32, answer starting position
                options:    (OPTIONAL) tf.RunOptions
                metadata:   (OPTIONAL) tf.RunMetaData
            Return:
                loss:       float32, current loss
                gs:         int32, global step
                gn:         float32, gloabl gradient norm
                summ:       runtime statistics
        '''
        fd = {
                self.doc: kwargs['document'],
                self.que: kwargs['question'],
                self.doc_len: kwargs['doc_length'],
                self.que_len: kwargs['que_length'],
                self.ans_start: kwargs['ans_start'],
                self.ans_len: kwargs['ans_length']
            }
        options = kwargs.get('options', None)
        metadata = kwargs.get('metadata', None)
        _, l, gs, gn, summ = sess.run([self.opt, self.loss, self.global_step, self.grads_norm, self.summaries],\
                feed_dict=fd, options=options, run_metadata=metadata)
        return l, gs, gn, summ

    def infer(self, sess, **kwargs):
        '''
            Apply the model.
            Required variables:
                document:   [batch_size, doc_maxlen] of int32, document tokens
                question:   [batch_size, que_maxlen] of int32, question tokens
                doc_length: [batch_size] of int32, actual length of document
                que_length: [batch_size] of int32, actual length of question
            Return:
                ans_s:      int32, answer starting position
                ans_e:      int32, answer ending position
        '''
        fd = {
                self.doc: kwargs['document'],
                self.que: kwargs['question'],
                self.doc_len: kwargs['doc_length'],
                self.que_len: kwargs['que_length'],
            }
        ans_s, ans_e = sess.run([self.start, self.end], feed_dict=fd)
        return ans_s, ans_e
