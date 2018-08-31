import tensorflow as tf


class Seq2Seq(object):

    def __init__(self, hparams, mode):
        self.mode = mode
        self.embeddings = tf.Variable(tf.random_uniform([hparams.voc_size, hparams.embedding_size], -1.0, 1.0))

        self.source = tf.placeholder(tf.int32, shape=[None, None], name='source')
        self.target_input = tf.placeholder(tf.int32, shape=[None, None], name='target_input')
        self.target_output = tf.placeholder(tf.int32, shape=[None, None], name='target_output')

        self.source_seq_length = tf.placeholder(tf.int32, shape=[None], name='source_seq_length')
        self.target_seq_length = tf.placeholder(tf.int32, shape=[None], name='target_seq_length')

        self.num_utterance = tf.placeholder(tf.int32, shape=[], name='num_utterance')

        self.num_units = hparams.num_units
        self.voc_size = hparams.voc_size

        self.loss, self.sample_id = self.build_graph(hparams)

        if mode == 'train':
            self.train_op = tf.train.AdamOptimizer(hparams.learning_rate).minimize(self.loss)

        self.saver = tf.train.Saver()

    def build_graph(self, hparams):
        enc_state = self.build_encoder(hparams)

        logits, sample_id = self.build_decoder(hparams, enc_state)

        loss = self.compute_loss(logits)

        return loss, sample_id

    def build_cell(self, hparams):
        cell = tf.nn.rnn_cell.GRUCell(hparams.num_units)
        return cell

    def build_encoder(self, hparams):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            enc_emb_input = tf.nn.embedding_lookup(self.embeddings, self.source)
            cell = self.build_cell(hparams)
            _, state = tf.nn.dynamic_rnn(cell, enc_emb_input, self.source_seq_length, dtype=tf.float32)
        return state

    def build_decoder(self, hparams, initial_state):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                dec_emb_input = tf.nn.embedding_lookup(self.embeddings, self.target_input)
                helper = tf.contrib.seq2seq.TrainingHelper(dec_emb_input, self.target_seq_length)
                cell = self.build_cell(hparams)
                decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, initial_state,
                                                          output_layer=tf.layers.Dense(hparams.voc_size,
                                                                                       use_bias=False))
                outputs, states, length = tf.contrib.seq2seq.dynamic_decode(decoder)
                logits = outputs.rnn_output
                sample_id = None
            else:
                cell = self.build_cell(hparams)

                start_tokens = tf.fill([self.num_utterance], 1)
                end_token = 2
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embeddings, start_tokens, end_token)
                decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, initial_state,
                                                          output_layer=tf.layers.Dense(hparams.voc_size,
                                                                                       use_bias=False))
                outputs, states, length = tf.contrib.seq2seq.dynamic_decode(decoder)
                logits = outputs.rnn_output
                sample_id = outputs.sample_id

        return logits, sample_id

    def compute_loss(self, logits):
        max_time = self.get_max_time(self.target_output)
        weights = tf.sequence_mask(self.target_seq_length, max_time, dtype=logits.dtype)
        loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(logits, self.target_output, weights))
        return loss

    def get_max_time(self, tensor):

        return tensor.shape[1].value


