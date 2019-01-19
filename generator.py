import tensorflow as tf
rnn = tf.nn.rnn_cell

class NormedLSTMCell(rnn.LSTMCell):
    def __init__(self, num_units, num_outputs=None, use_peepholes=False, reuse=None):
        super().__init__(num_units, use_peepholes, reuse=reuse)
        self._output_size_copy = self._output_size
        if num_outputs is not None:
            self._output_size = num_outputs
            self._output_layer = lambda inputs: tf.layers.dense(inputs, num_outputs, tf.nn.softmax)
        else:
            self._output_layer = None

    def _reshape_state(self, state, shape_wo_last_dim=None):
        if shape_wo_last_dim is None:
            shape_wo_last_dim = [-1]
        c_shape = shape_wo_last_dim+[self._state_size.c]
        h_shape = shape_wo_last_dim+[self._state_size.h]
        return rnn.LSTMStateTuple(c=tf.reshape(state.c, c_shape), h=tf.reshape(state.h, h_shape))

    def call(self, inputs, state):
        high_dim = inputs.shape.ndims > 2
        if high_dim:
            dynamic_shape = tf.shape(inputs)
            static_shape = inputs.shape.as_list()
            shape_wo_last_dim = [d or dynamic_shape[i] for i, d in enumerate(static_shape[:-1])]
            inputs = tf.reshape(inputs, [-1, static_shape[-1]])
            state = self._reshape_state(state)
        output, new_state = super().call(inputs, state)
        if high_dim:
            out_shape = shape_wo_last_dim+[self._output_size_copy]
            output = tf.reshape(output, out_shape)
            new_state = self._reshape_state(new_state, shape_wo_last_dim)
        if self._output_layer is not None:
            output = self._output_layer(output)
        return output, new_state

class G_net(object):
    def __init__(self, source_vocab_size, target_vocab_size, enc_hidden_dim, dec_hidden_dim, attention_dim, beam_size,\
            learning_rate, max_grad_norm, source_pretrained_emb, target_pretrained_emb):
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.attention_dim = attention_dim
        self.beam_size = beam_size
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.end_token = self.target_vocab_size-1

        self.source = tf.placeholder(tf.int32, [None, None], 'source')
        self.target = tf.placeholder(tf.int32, [None, None], 'target')
        self.source_len = tf.placeholder(tf.int32, [None], 'source_length')
        self.source_maxlen = tf.reduce_max(self.source_len, name='source_max_length')
        self.target_len = tf.placeholder(tf.int32, [None], 'target_length')
        self.target_maxlen = tf.reduce_max(self.target_len, name='target_max_length')
        self.ans_start = tf.placeholder(tf.int32, [None], 'answer_start')
        self.ans_len = tf.placeholder(tf.int32, [None], 'answer_length')
        self.pre_global_steps = tf.Variable(0, trainable=False, name='pre_global_steps', dtype=tf.int32)
        self.gen_global_steps = tf.Variable(0, trainable=False, name='gen_global_steps', dtype=tf.int32)

        assert source_pretrained_emb.shape[1] == target_pretrained_emb.shape[1]
        self.emb_dim = source_pretrained_emb.shape[1]
        source_emb = self._build_emb_mat(self.source_vocab_size, source_pretrained_emb, name='source_embedding')
        target_emb = self._build_emb_mat(self.target_vocab_size, target_pretrained_emb, name='target_embedding')
        self.emb = tf.concat([source_emb, target_emb], axis=0, name='embedding_matrix')

        with tf.variable_scope('Encoder') as vs:
            self.doc = tf.nn.embedding_lookup(self.emb, self.source, name='document')
            self.ans = self._extract_answer_span(self.doc)

            doc_hidden, _ = self._context_BiLSTM(self.enc_hidden_dim, self.doc, self.source_len, name='document_context')
            ans_hidden = self._extract_answer_span(doc_hidden, name='answer_context')
            ans_repr = tf.concat([ans_hidden, self.ans], axis=-1, name='answer_representation')
            _, ans_state = self._context_BiLSTM(self.enc_hidden_dim, ans_repr, self.ans_len, name='answer')
            memory = tf.concat([doc_hidden, tf.tile(tf.expand_dims(ans_state, 1), [1, self.source_maxlen, 1])], axis=-1, name='memory')

        with tf.variable_scope('Decoder') as vs:
            # prepare initial state of decoder
            doc_hidden_avg = tf.reduce_sum(doc_hidden, 1) / tf.cast(tf.expand_dims(self.source_len, -1), tf.float32)
            h = tf.layers.dense(tf.concat([ans_state, doc_hidden_avg], -1), self.dec_hidden_dim, tf.nn.tanh, name='initial_state')
            batch_size = tf.shape(h)[0]
            init_state = rnn.LSTMStateTuple(c=h, h=h)
            init_score = self._attention_score(h, memory, 'init_score')
            init_attention = tf.squeeze(tf.matmul(tf.expand_dims(init_score, 1), memory), 1, name='init_atttention')
            init_time = tf.constant(0, tf.int32, name='init_time')
            init_finished = tf.zeros([batch_size], tf.bool, name='init_finished')
            # prepare training data (target sequence)
            self.que = tf.nn.embedding_lookup(self.emb, self._extract_doc_words(self.target), name='question')
            target_ta = tf.TensorArray(tf.float32, self.target_maxlen).unstack(tf.transpose(self.que, [1,0,2]))
            cell = NormedLSTMCell(self.dec_hidden_dim, self.target_vocab_size)

            with tf.name_scope('Training') as ns:
                init_inputs = tf.random_uniform([batch_size, self.emb_dim], -.5, .5, seed=2017, name='init_inputs')
                preds_ta = tf.TensorArray(tf.float32, 0, True)
                sampling_prob = tf.train.exponential_decay(1.0, self.pre_global_steps, 1000, .9, name='sampling_prob')

                def teacher_forcing(time, inputs, attention, state, finished, preds_ta):
                    ''' loop body for training'''
                    # Compute output
                    cell_output, cell_state = cell(tf.concat([inputs, attention], -1), state)
                    score = self._attention_score(cell_state.h, memory, name='copy_score')
                    next_attention = tf.squeeze(tf.matmul(tf.expand_dims(score, 1), memory), 1, name='next_attention')
                    switch = tf.layers.dense(next_attention, 1, activation=tf.sigmoid, name='switch_weight')
                    preds = tf.log(tf.concat([switch*cell_output, (1-switch)*score], axis=-1), name='current_prediction')
                    # scheduled sampling
                    using_sample = tf.greater(tf.random_uniform([batch_size], dtype=tf.float32, seed=2017), sampling_prob)
                    samples = tf.nn.embedding_lookup(self.emb, self._extract_doc_words(tf.argmax(preds, -1)))
                    next_inputs = tf.where(using_sample, samples, target_ta.read(time), name='next_inputs')
                    # dynamically accomodate variable length sequences in one batch
                    next_time = time + 1
                    next_finished = tf.greater_equal(next_time, self.target_len, name='next_finished')
                    next_state = self._copy_state(finished, cell_state, state, name='next_state')
                    trimmed_preds = tf.where(finished, tf.zeros_like(preds)-32.0, preds)
                    preds_ta = preds_ta.write(time, trimmed_preds)
                    return next_time, next_inputs, next_attention, next_state, next_finished, preds_ta

                _, _, _, _, _, preds_ta = tf.while_loop(
                        cond=lambda time, _1, _2, _3, _4, _5:  tf.less(time, self.target_maxlen),
                        body=teacher_forcing,
                        loop_vars=[init_time, init_inputs, init_attention, init_state, init_finished, preds_ta],
                        parallel_iterations=32)
                self.pretrain_logits = tf.transpose(preds_ta.stack(), [1,0,2], name='logits')
                tf.summary.histogram('pretrain_logits', self.pretrain_logits)
                cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pretrain_logits, labels=self.target)
                target_mask = tf.sequence_mask(self.target_len, self.target_maxlen, name='target_mask')
                self.pretrain_loss = tf.reduce_mean(tf.boolean_mask(cross_ent, target_mask), name='loss')
                tf.summary.scalar('loss', self.pretrain_loss)

            with tf.name_scope('Inference') as ns:
                vs.reuse_variables()
                init_inputs_beams = tf.random_uniform([batch_size, self.beam_size, self.emb_dim], -.5, .5, seed=2017, name='init_inputs_beams')
                init_h = tf.stack([init_state.h]*self.beam_size, axis=1, name='init_h_beams')
                init_c = tf.stack([init_state.c]*self.beam_size, axis=1, name='init_c_beams')
                init_state_beams = rnn.LSTMStateTuple(c=init_c, h=init_h)
                init_attention_beams = tf.stack([init_attention]*self.beam_size, axis=1, name='init_attention_beams')
                init_top_preds = tf.zeros([batch_size, self.beam_size], tf.float32, name='init_top_preds')
                paths_ta = tf.TensorArray(tf.int32, 0, dynamic_size=True, element_shape=tf.TensorShape([None, self.beam_size]))
                samples_ta = tf.TensorArray(tf.int32, 0, dynamic_size=True, element_shape=tf.TensorShape([None, self.beam_size]))
                real_target_vocab_size = self.source_maxlen + self.target_vocab_size

                def decode_recurrence(time, inputs, attention, state, finished, top_preds, paths_ta, samples_ta):
                    cell_output, cell_state = cell(tf.concat([inputs, attention], -1), state)
                    score = self._attention_score(cell_state.h, memory)
                    next_attention = tf.matmul(score, memory, name='next_attention')
                    switch = tf.layers.dense(next_attention, 1, activation=tf.sigmoid, name='switch_weight')
                    preds = tf.log(tf.concat([switch*cell_output, (1-switch)*score], axis=-1), name='current_prediction')
                    # beam search
                    total_preds = tf.reshape(tf.expand_dims(top_preds, -1) + preds, [batch_size, -1])
                    next_top_preds, indices = tf.nn.top_k(total_preds, self.beam_size, name='next_beam')
                    paths = tf.div(indices, real_target_vocab_size, name='paths')
                    samples = tf.mod(indices, real_target_vocab_size, name='samples')

                    # dynamically stop generation when <EOS> is encountered
                    next_time = time+1
                    next_finished = tf.logical_or(tf.equal(samples[:,0], self.end_token), finished, name='next_finished')
                    next_samples = tf.where(finished, tf.zeros_like(samples), samples, name='next_samples')
                    next_inputs = tf.nn.embedding_lookup(self.emb, self._extract_doc_words(next_samples), name='next_inputs')
                    next_state = self._copy_state(finished, cell_state, state, name='next_state')
                    trimmed_paths = tf.where(finished, tf.zeros_like(paths), paths)
                    paths_ta = paths_ta.write(time, trimmed_paths)
                    samples_ta = samples_ta.write(time, next_samples)
                    return next_time, next_inputs, next_attention, next_state, next_finished, next_top_preds, paths_ta, samples_ta

                size, _, _, _, _, _, paths_ta, samples_ta = tf.while_loop(
                        cond=lambda time, _2, _3, _4, finished, _6, _7, _8: tf.logical_and(tf.logical_not(tf.reduce_all(finished)),\
                                tf.less(time, self.target_maxlen)),
                        body=decode_recurrence,
                        loop_vars=[init_time, init_inputs_beams, init_attention_beams, init_state_beams, init_finished, init_top_preds, paths_ta, samples_ta],
                        parallel_iterations=32)

                seq = tf.TensorArray(tf.int32, size, name='obtained_seq')
                last_paths = tf.zeros([batch_size], tf.int32, name='init_paths')
                batches = tf.range(batch_size, dtype=tf.int32, name='batches')

                def body(time, paths, seq):
                    indices = tf.stack([batches, paths], -1, name='indices')
                    cur_beams = samples_ta.read(time, name='current_beams')
                    cur_samples = tf.gather_nd(cur_beams, indices, name='current_samples')
                    seq = seq.write(time, cur_samples, name='update_sequence')
                    prev_record = paths_ta.read(time, name='prev_record')
                    prev_paths = tf.gather_nd(prev_record, indices, name='prev_path')
                    return time-1, prev_paths, seq

                _, _, seq = tf.while_loop(
                        cond=lambda time, _1, _2: tf.greater_equal(time, 0),
                        body=body,
                        loop_vars=[size-1, last_paths, seq],
                        parallel_iterations=32)
                self.generated_seq = tf.transpose(seq.stack(), [1,0], name='reconstructed_sequence')

        self.opt = tf.train.AdamOptimizer(self.learning_rate)
        pretrain_grads, pretrain_vars = zip(*self.optimizer.compute_gradients(self.pretrain_loss))
        clipped_pretrain_grads, _ = tf.clip_by_global_norm(pretrain_grads, self.max_grad_norm, name='clipping_grads')
        self.pretrain_step = self.optimizer.apply_gradients(zip(clipped_pretrain_grads, pretrain_vars), self.pre_global_steps, name='pretrain')
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

    def _context_BiLSTM(self, hidden_dim, inputs, lens, name=None):
        with tf.variable_scope(name, 'Context') as vs:
            cell_fw = rnn.BasicLSTMCell(hidden_dim)
            cell_bw = rnn.BasicLSTMCell(hidden_dim)
            output, (s_fw, s_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, lens,\
                                                                   dtype=tf.float32, scope=vs)
            output = tf.concat(output, axis=-1, name='output')
            hidden = tf.concat([s_fw.h, s_bw.h], axis=-1, name='hidden')
        return output, hidden

    def _copy_state(self, finished, new_state, old_state, name=None):
        with tf.name_scope(name, 'CopyState', [finished, new_state, old_state]) as scope:
            state_h = tf.where(finished, old_state.h, new_state.h, name='selected_h')
            state_c = tf.where(finished, old_state.c, new_state.c, name='selected_c')
            state = rnn.LSTMStateTuple(c=state_c, h=state_h)
        return state

    def _extract_answer_span(self, source, name=None):
        with tf.name_scope(name, 'ExtractAnswerSpan', [source, self.ans_start, self.ans_len]) as scope:
            reversed_seqs = tf.reverse(source, axis=[1])
            truncated_seqs = tf.reverse_sequence(reversed_seqs, self.source_maxlen-self.ans_start, seq_axis=1)
            mask = tf.expand_dims(tf.sequence_mask(self.ans_len, self.source_maxlen, dtype=tf.float32), -1)
            max_length = tf.reduce_max(self.ans_len)
            span = tf.identity((truncated_seqs*mask)[:, :max_length], name=scope)
        return span

    def _extract_doc_words(self, target, name=None):
        with tf.name_scope(name, 'ExtractDocWords', [target]) as scope:
            if target.dtype != tf.int32:
                target = tf.cast(target, tf.int32)
            cond = tf.less(target, self.target_vocab_size)
            x = tf.add(target, self.source_vocab_size)
            indices = tf.clip_by_value(tf.subtract(target, self.target_vocab_size), 0, self.source_maxlen-1)
            batches = tf.transpose(tf.range(tf.shape(target)[0]) + tf.transpose(tf.zeros_like(target)))
            y = tf.gather_nd(self.source, tf.stack([batches, indices], axis=-1))
            real_target = tf.where(cond, x, y, name=scope)
        return real_target

    def _attention_score(self, query, memory, name=None):
        with tf.variable_scope('Attention') as vs:
            try:
                v = tf.get_variable('attention_weight', [self.attention_dim], tf.float32)
            except ValueError:
                vs.reuse_variables()
                v = tf.get_variable('attention_weight', [self.attention_dim], tf.float32)
            processed_query = tf.expand_dims(tf.layers.dense(query, self.attention_dim, use_bias=False), -2)
            processed_memory = tf.layers.dense(memory, self.attention_dim, use_bias=False)
            if processed_query.shape.ndims > 3:
                processed_memory = tf.expand_dims(processed_memory, 1)
            score = tf.nn.softmax(tf.reduce_sum(v*tf.tanh(processed_query+processed_memory), -1), name=name or 'attention_score')
        return score

    def pretrain(self, sess, **kwargs):
        feed_dict = {
                self.source: kwargs['source'],
                self.source_len: kwargs['source_length'],
                self.target: kwargs['target'],
                self.target_len: kwargs['target_length'],
                self.ans_start: kwargs['answer_start'],
                self.ans_len: kwargs['answer_length']
                }
        options = kwargs.get('options', None)
        metadata = kwargs.get('metadata', None)
        summaries, loss, gs, _ = sess.run([self.summaries, self.pretrain_loss, self.pre_global_steps, self.pretrain_step],\
                feed_dict=feed_dict, options=options, run_metadata=metadata)
        return summaries, loss, gs

    def generate(self, sess, **kwargs):
        feed_dict = {
                self.source: kwargs['source'],
                self.source_len: kwargs['source_length'],
                self.target_len: kwargs['target_length'],
                self.ans_start: kwargs['answer_start'],
                self.ans_len: kwargs['answer_length'],
                }
        seq = sess.run(self.generated_seq, feed_dict=feed_dict)
        return seq
