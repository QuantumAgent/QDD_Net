import tensorflow as tf
import os


class BaseModel(object):
    def __init__(self, word_embedding, char_embedding, model_name, log_dir):
        self.word_embedding = word_embedding
        self.char_embedding = char_embedding
        self._name = model_name
        tf.reset_default_graph()
        self.q1_words = tf.placeholder(shape=[None, None], dtype=tf.int32, name='q1_words')
        self.q2_words = tf.placeholder(shape=[None, None], dtype=tf.int32, name='q2_words')
        self.q1_chars = tf.placeholder(shape=[None, None], dtype=tf.int32, name='q1_chars')
        self.q2_chars = tf.placeholder(shape=[None, None], dtype=tf.int32, name='q2_chars')
        self.y = tf.placeholder(shape=[None], dtype=tf.int32, name='y_start')
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_keep_prob')
        self.global_step = 0

    def train(self, q1w, q1c, q2w, q2c, y, drop_keep_prob=0.85, record_interval=10):
        feed_dict = {
            self.q1_words: q1w,
            self.q2_words: q2w,
            self.q1_chars: q1c,
            self.q2_chars: q2c,
            self.y: y,
            self.dropout_keep_prob: drop_keep_prob
        }
        if self.global_step % record_interval == 0:
            _, loss, summaries = self.session.run([self.train_op, self.loss, self.merge_op], feed_dict=feed_dict)
            self.writer.add_summary(summaries, self.global_step)
        else:
            _, loss = self.session.run([self.train_op, self.loss], feed_dict=feed_dict)
        self.global_step += 1
        return loss

    def evaluate(self, q1w, q1c, q2w, q2c, y, drop_keep_prob=1.0):
        feed_dict = {
            self.q1_words: q1w,
            self.q2_words: q2w,
            self.q1_chars: q1c,
            self.q2_chars: q2c,
            self.y: y,
            self.dropout_keep_prob: drop_keep_prob
        }
        loss = self.session.run([self.loss], feed_dict=feed_dict)[0]
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = loss
        summary_value.tag = 'evaluate_loss'
        self.writer.add_summary(summary, self.global_step)
        return loss

    def predict(self, q1w, q1c, q2w, q2c):
        feed_dict = {
            self.q1_words: q1w,
            self.q2_words: q2w,
            self.q1_chars: q1c,
            self.q2_chars: q2c,
            self.dropout_keep_prob: 1.0
        }
        y_hat = self.session.run([self.output], feed_dict=feed_dict)[0]
        return y_hat[:, 1]

    def load_model(self, model_path='./QModel'):
        self.saver.restore(self.session, model_path + '/' + self._name)

    def save_model(self, model_path='./QModel'):
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_file = model_path + '/' + self._name
        self.saver.save(self.session, model_file)
