import numpy as np
from models import rnn_q_net

lmap = lambda func, it: list(map(lambda x: func(x), it))


def main():
    train_corpus = np.load('train_corpus.npy')
    test_corpus = np.load('test_corpus.npy')

    word_matrix = np.load('word_matrix.npy')
    char_matrix = np.load('char_matrix.npy')

    np.random.shuffle(train_corpus)

    dev_corpus = train_corpus[:1024]
    train_corpus = train_corpus[1024:]

    net = rnn_q_net.RnnQNet(word_embedding=word_matrix, char_embedding=char_matrix, log_dir='./logs/model4')

    global_step = 0
    previous_save_loss = np.inf
    max_batch_size = 256
    max_dev_batch_size = 256
    checkpoint_interval = 30
    evaluate_interval = 10
    q1w_col = 39
    q1c_col = q1w_col + 58
    q2w_col = q1c_col + 39
    q2c_col = q2w_col + 58
    for e in range(10):
        np.random.shuffle(train_corpus)
        epoch_loss = []
        it = 0
        while it < train_corpus.shape[0]:
            max_q1w_length = max(np.sum(train_corpus[it:it + max_batch_size, :q1w_col] > 0, axis=1))
            b_q1w = train_corpus[it:it + max_batch_size, :max_q1w_length]
            max_q1c_length = max(np.sum(train_corpus[it:it + max_batch_size, q1w_col:q1c_col] > 0, axis=1))
            b_q1c = train_corpus[it:it + max_batch_size, q1w_col:q1w_col + max_q1c_length]
            max_q2w_length = max(np.sum(train_corpus[it:it + max_batch_size, q1c_col:q2w_col] > 0, axis=1))
            b_q2w = train_corpus[it:it + max_batch_size, q1c_col:q1c_col + max_q2w_length]
            max_q2c_length = max(np.sum(train_corpus[it:it + max_batch_size, q2w_col:q2c_col] > 0, axis=1))
            b_q2c = train_corpus[it:it + max_batch_size, q2w_col:q2w_col + max_q2c_length]
            b_y = train_corpus[it:it + max_batch_size, -1]
            loss = net.train(q1w=b_q1w, q1c=b_q1c, q2w=b_q2w, q2c=b_q2c, y=b_y, record_interval=1)
            epoch_loss.append(loss)
            print('epoch', e, 'step', global_step, 'iteration', it, 'iteration loss', loss, 'epoch mean loss',
                  np.mean(epoch_loss))
            it += max_batch_size
            global_step += 1
            if global_step % evaluate_interval == 0:
                np.random.shuffle(dev_corpus)
                max_q1w_length = max(np.sum(dev_corpus[:max_dev_batch_size, :q1w_col] > 0, axis=1))
                b_q1w = dev_corpus[:max_dev_batch_size, :max_q1w_length]
                max_q1c_length = max(np.sum(dev_corpus[:max_dev_batch_size, q1w_col:q1c_col] > 0, axis=1))
                b_q1c = dev_corpus[:max_dev_batch_size, q1w_col:q1w_col + max_q1c_length]
                max_q2w_length = max(np.sum(dev_corpus[:max_dev_batch_size, q1c_col:q2w_col] > 0, axis=1))
                b_q2w = dev_corpus[:max_dev_batch_size, q1c_col:q1c_col + max_q2w_length]
                max_q2c_length = max(np.sum(dev_corpus[:max_dev_batch_size, q2w_col:q2c_col] > 0, axis=1))
                b_q2c = dev_corpus[:max_dev_batch_size, q2w_col:q2w_col + max_q2c_length]
                b_y = dev_corpus[:max_dev_batch_size, -1]
                dev_loss = net.evaluate(q1w=b_q1w, q1c=b_q1c, q2w=b_q2w, q2c=b_q2c, y=b_y)
                if global_step % checkpoint_interval == 0 and previous_save_loss > dev_loss:
                    net.save_model(model_path='./QModel4')
                    previous_save_loss = dev_loss
                    print(global_step, 'save model @ val loss', dev_loss)
                print(global_step, 'evaluate loss', dev_loss)

    net.load_model(model_path='./QModel4')

    it = 0
    result = []
    while it < test_corpus.shape[0]:
        max_q1w_length = max(np.sum(test_corpus[it:it + max_batch_size, :q1w_col] > 0, axis=1))
        b_q1w = test_corpus[it:it + max_batch_size, :max_q1w_length]
        max_q1c_length = max(np.sum(test_corpus[it:it + max_batch_size, q1w_col:q1c_col] > 0, axis=1))
        b_q1c = test_corpus[it:it + max_batch_size, q1w_col:q1w_col + max_q1c_length]
        max_q2w_length = max(np.sum(test_corpus[it:it + max_batch_size, q1c_col:q2w_col] > 0, axis=1))
        b_q2w = test_corpus[it:it + max_batch_size, q1c_col:q1c_col + max_q2w_length]
        max_q2c_length = max(np.sum(test_corpus[it:it + max_batch_size, q2w_col:q2c_col] > 0, axis=1))
        b_q2c = test_corpus[it:it + max_batch_size, q2w_col:q2w_col + max_q2c_length]
        y_hat = net.predict(q1w=b_q1w, q1c=b_q1c, q2w=b_q2w, q2c=b_q2c)
        result.extend(y_hat)
        it += max_batch_size
        print(it / test_corpus.shape[0])

    result_str = lmap(lambda x: str(x) + '\n', result)

    with open('result_5.csv', 'w+') as f:
        f.writelines(result_str)


if __name__ == '__main__':
    main()
