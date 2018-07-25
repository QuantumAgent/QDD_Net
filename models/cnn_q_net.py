import tensorflow as tf
from models.layers import add_dense_layer, dot_attention, biGRUs, add_CNNs
from models.base_model import BaseModel

lmap = lambda func, it: list(map(lambda x: func(x), it))


class CnnQNet(BaseModel):
    def __init__(self, word_embedding, char_embedding, log_dir='./logs', learning_rate=0.0001):
        super().__init__(word_embedding, char_embedding, 'Cnn_QNet', log_dir=log_dir)

        self._build_model(word_embedding, char_embedding, cnn_encoder_units_number=[256, 128], attention_size=[128],
                          attention_cnn_size=[128, 64], kernel_size=2, learning_rate=learning_rate)
        self.init_op = tf.global_variables_initializer()
        self.merge_op = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.writer = tf.summary.FileWriter(log_dir,
                                            graph=self.session.graph
                                            )
        self.session.run(self.init_op)

    def _build_model(self, word_embedding, char_embedding, cnn_encoder_units_number, attention_size,
                     attention_cnn_size, kernel_size, learning_rate=0.0001):
        with tf.variable_scope('word_embedding', initializer=tf.contrib.layers.xavier_initializer()):
            Ww = tf.Variable(word_embedding, trainable=True, dtype=tf.float32)
            self.q1w = tf.nn.embedding_lookup(ids=self.q1_words, params=Ww)
            self.q2w = tf.nn.embedding_lookup(ids=self.q2_words, params=Ww)

        with tf.variable_scope('char_embedding', initializer=tf.contrib.layers.xavier_initializer()):
            Wc = tf.Variable(char_embedding, trainable=True, dtype=tf.float32)
            self.q1c = tf.nn.embedding_lookup(ids=self.q1_chars, params=Wc)
            self.q2c = tf.nn.embedding_lookup(ids=self.q2_chars, params=Wc)

        with tf.variable_scope('word_encoder', initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                               reuse=tf.AUTO_REUSE) as scope:
            # fcell, bcell = biGRUs(cnn_encoder_units_number, activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)
            # self.q1we, self._ = tf.nn.bidirectional_dynamic_rnn(inputs=self.q1w, cell_fw=fcell, cell_bw=bcell,
            #                                                     dtype=tf.float32, scope='q1w_encoder')
            # self.q1we = tf.concat(self.q1we, axis=-1)
            # self.q1we = tf.contrib.layers.layer_norm(self.q1we, scope=scope)
            self.q1we = add_CNNs(inputs=self.q1w, hidden_units=cnn_encoder_units_number, kernel_size=kernel_size,
                                 activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)
            #
            # tf.get_variable_scope().reuse_variables()
            # self.q2we, self._ = tf.nn.bidirectional_dynamic_rnn(inputs=self.q2w, cell_fw=fcell, cell_bw=bcell,
            #                                                     dtype=tf.float32, scope='q2w_encoder')
            # self.q2we = tf.concat(self.q2we, axis=-1)
            #
            # self.q2we = tf.contrib.layers.layer_norm(self.q2we, scope=scope)
            self.q2we = add_CNNs(inputs=self.q2w, hidden_units=cnn_encoder_units_number, kernel_size=kernel_size,
                                 activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)

            tf.summary.histogram('q1we', self.q1we)
            tf.summary.histogram('q2we', self.q2we)

        with tf.variable_scope('char_encoder', initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                               reuse=tf.AUTO_REUSE) as scope:
            # fcell, bcell = biGRUs(cnn_encoder_units_number, activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)
            # self.q1ce, self._ = tf.nn.bidirectional_dynamic_rnn(inputs=self.q1c, cell_fw=fcell, cell_bw=bcell,
            #                                                     dtype=tf.float32, scope='q1c_encoder')
            # self.q1ce = tf.concat(self.q1ce, axis=-1)
            # self.q1ce = tf.contrib.layers.layer_norm(self.q1ce, scope=scope)
            #
            # tf.get_variable_scope().reuse_variables()
            # self.q2ce, self._ = tf.nn.bidirectional_dynamic_rnn(inputs=self.q2c, cell_fw=fcell, cell_bw=bcell,
            #                                                     dtype=tf.float32, scope='q2c_encoder')
            # self.q2ce = tf.concat(self.q2ce, axis=-1)
            #
            # self.q2ce = tf.contrib.layers.layer_norm(self.q2ce, scope=scope)
            self.q1ce = add_CNNs(inputs=self.q1c, hidden_units=cnn_encoder_units_number, kernel_size=kernel_size * 2,
                                 activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)
            self.q2ce = add_CNNs(inputs=self.q2c, hidden_units=cnn_encoder_units_number, kernel_size=kernel_size * 2,
                                 activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)
            tf.summary.histogram('q1ce', self.q1ce)
            tf.summary.histogram('q2ce', self.q2ce)

        with tf.variable_scope('word_attention', initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                               reuse=tf.AUTO_REUSE) as scope:
            fcell, bcell = biGRUs(attention_cnn_size, activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)

            self.q1q2w_att = dot_attention(self.q1we, self.q2we, hidden=attention_size, scope='q1q2w_attention')
            self.q1q2w_att = add_CNNs(inputs=self.q1q2w_att, hidden_units=attention_cnn_size, kernel_size=kernel_size,
                                      activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)
            # self.q1q2w_att, _ = tf.nn.bidirectional_dynamic_rnn(inputs=self.q1q2w_att, cell_fw=fcell, cell_bw=bcell,
            #                                                     dtype=tf.float32, scope='q1q2w_attention_rnn')
            # self.q1q2w_att = tf.concat(self.q1q2w_att, axis=-1)
            # self.q1q2w_att = tf.contrib.layers.layer_norm(self.q1q2w_att, scope=scope)
            #
            self.q2q1w_att = dot_attention(self.q2we, self.q1we, hidden=attention_size, scope='q2q1w_attention')
            self.q2q1w_att = add_CNNs(inputs=self.q2q1w_att, hidden_units=attention_cnn_size, kernel_size=kernel_size,
                                      activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)
            # tf.get_variable_scope().reuse_variables()
            # self.q2q1w_att, _ = tf.nn.bidirectional_dynamic_rnn(inputs=self.q2q1w_att, cell_fw=fcell, cell_bw=bcell,
            #                                                     dtype=tf.float32, scope='q2q1w_attention_rnn')
            # self.q2q1w_att = tf.concat(self.q2q1w_att, axis=-1)
            # self.q2q1w_att = tf.contrib.layers.layer_norm(self.q2q1w_att, scope=scope)
            tf.summary.histogram('q1q2_att', self.q1q2w_att)
            tf.summary.histogram('q2q1w_att', self.q2q1w_att)

        with tf.variable_scope('char_attention', initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                               reuse=tf.AUTO_REUSE) as scope:
            fcell, bcell = biGRUs(attention_cnn_size, activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)

            self.q1q2c_att = dot_attention(self.q1ce, self.q2ce, hidden=attention_size, scope='q1q2c_attention')
            self.q1q2c_att = add_CNNs(inputs=self.q1q2c_att, hidden_units=attention_cnn_size,
                                      kernel_size=kernel_size * 2, activation=tf.nn.relu,
                                      keep_prob=self.dropout_keep_prob)
            # self.q1q2c_att, _ = tf.nn.bidirectional_dynamic_rnn(inputs=self.q1q2c_att, cell_fw=fcell, cell_bw=bcell,
            #                                                     dtype=tf.float32, scope='q1q2c_attention_rnn')
            # self.q1q2c_att = tf.concat(self.q1q2c_att, axis=-1)
            # self.q1q2c_att = tf.contrib.layers.layer_norm(self.q1q2c_att, scope=scope)

            self.q2q1c_att = dot_attention(self.q2ce, self.q1ce, hidden=attention_size, scope='q2q1c_attention')
            self.q2q1c_att = add_CNNs(inputs=self.q2q1c_att, hidden_units=attention_cnn_size,
                                      kernel_size=kernel_size * 2, activation=tf.nn.relu,
                                      keep_prob=self.dropout_keep_prob)
            # tf.get_variable_scope().reuse_variables()
            # self.q2q1c_att, _ = tf.nn.bidirectional_dynamic_rnn(inputs=self.q2q1c_att, cell_fw=fcell, cell_bw=bcell,
            #                                                     dtype=tf.float32, scope='q2q1c_attention_rnn')
            # self.q2q1c_att = tf.concat(self.q2q1c_att, axis=-1)
            # self.q2q1c_att = tf.contrib.layers.layer_norm(self.q2q1c_att, scope=scope)
            tf.summary.histogram('q1q2c_att', self.q1q2c_att)
            tf.summary.histogram('q2q1c_att', self.q2q1c_att)

        with tf.variable_scope('word_output_layer',
                               initializer=tf.contrib.layers.xavier_initializer(uniform=True)) as scope:
            # self.s1=tf.layers.max_pooling1d(self.q1q2w_att,pool_size=kernel_size,strides=1,padding='valid')
            # self.s2=tf.layers.max_pooling1d(self.q2q1w_att,pool_size=kernel_size,strides=1,padding='valid')

            # wgru = add_GRU(attention_cnn_size[-1] // 2, activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)
            # _, self.s1 = tf.nn.dynamic_rnn(inputs=self.q1q2w_att, cell=wgru, dtype=tf.float32, scope='s1')
            # tf.get_variable_scope().reuse_variables()
            # _, self.s2 = tf.nn.dynamic_rnn(inputs=self.q2q1w_att, cell=wgru, dtype=tf.float32, scope='s2')
            # # s1_number=tf.reduce_sum(tf.abs(tf.sign(self.s1)),axis=1)
            # # s2_number=tf.reduce_sum(tf.abs(tf.sign(self.s2)),axis=1)
            # s1_number=tf.count_nonzero(self.s1,axis=1,dtype=tf.float32)
            # s2_number=tf.count_nonzero(self.s2,axis=1,dtype=tf.float32)
            # self.s1=tf.reduce_sum(self.s1,axis=1)/s1_number
            # self.s2=tf.reduce_sum(self.s2,axis=1)/s2_number
            # self.s1=tf.reduce_mean(self.s1,axis=1)
            # self.s2=tf.reduce_mean(self.s2,axis=1)
            self.s1 = add_CNNs(inputs=self.q1q2w_att, hidden_units=lmap(lambda x: x // 2, attention_cnn_size),
                               kernel_size=kernel_size, activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)
            self.s1 = tf.reduce_mean(self.s1, axis=1)
            self.s2 = add_CNNs(inputs=self.q2q1w_att, hidden_units=lmap(lambda x: x // 2, attention_cnn_size),
                               kernel_size=kernel_size, activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)
            self.s2 = tf.reduce_mean(self.s2, axis=1)

        with tf.variable_scope('char_output_layer',
                               initializer=tf.contrib.layers.xavier_initializer(uniform=True)) as scope:
            # self.s3 = tf.layers.max_pooling1d(self.q1q2c_att, pool_size=kernel_size*2, strides=1, padding='valid')
            # self.s4 = tf.layers.max_pooling1d(self.q2q1c_att, pool_size=kernel_size*2, strides=1, padding='valid')
            # cgru = add_GRU(attention_cnn_size[-1] // 2, activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)
            # _, self.s3 = tf.nn.dynamic_rnn(inputs=self.q1q2c_att, cell=cgru, dtype=tf.float32, scope='s3')
            # tf.get_variable_scope().reuse_variables()
            # _, self.s4 = tf.nn.dynamic_rnn(inputs=self.q2q1c_att, cell=cgru, dtype=tf.float32, scope='s4')
            # # s3_number=tf.reduce_sum(tf.abs(tf.sign(self.s3)),axis=1)
            # # s4_number=tf.reduce_sum(tf.abs(tf.sign(self.s4)),axis=1)
            # s3_number=tf.count_nonzero(self.s3,axis=1,dtype=tf.float32)
            # s4_number=tf.count_nonzero(self.s4,axis=1,dtype=tf.float32)
            # self.s3=tf.reduce_sum(self.s3,axis=1)/s3_number
            # self.s4=tf.reduce_sum(self.s4,axis=1)/s4_number
            # self.s3=tf.reduce_mean(self.s3,axis=1)
            # self.s4=tf.reduce_mean(self.s4,axis=1)
            self.s3 = add_CNNs(inputs=self.q1q2c_att, hidden_units=lmap(lambda x: x // 2, attention_cnn_size),
                               kernel_size=kernel_size * 2, activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)
            self.s3 = tf.reduce_mean(self.s3, axis=1)
            self.s4 = add_CNNs(inputs=self.q2q1c_att, hidden_units=lmap(lambda x: x // 2, attention_cnn_size),
                               kernel_size=kernel_size * 2, activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)
            self.s4 = tf.reduce_mean(self.s4, axis=1)

        with tf.variable_scope('output_layer', initializer=tf.contrib.layers.xavier_initializer(uniform=True)) as scope:
            self.o = tf.concat([self.s1, self.s2, self.s3, self.s4], axis=-1)
            # self.o=tf.concat([self.s3,self.s4],axis=-1)
            self.o = tf.contrib.layers.layer_norm(self.o, scope=scope)
            self.o = add_dense_layer(self.o, [128, 64, 2], self.dropout_keep_prob, activation=tf.nn.relu, use_bias=True)
            self.output = tf.nn.softmax(self.o)
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.o, labels=self.y))

        with tf.variable_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            tf.summary.scalar('loss', self.loss)
            self.train_op = self.optimizer.minimize(self.loss)
