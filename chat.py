import tensorflow as tf
import numpy as np
import sys
from models import Seq2Seq
from dialogue import Dialogue


def reply(model, sess, sentence):

    tokens= dialogue.tokenizer(sentence)



    ids= dialogue.tokens_to_ids(tokens)

    result = sess.run(model.sample_id,
                      feed_dict={model.source: [ids],
                                 model.source_seq_length: [len(ids)],
                                 model.num_utterance: 1})
    end = np.where(result == 2)

    r = ' '.join(dialogue.ids_to_tokens(result[:end[0][0]]))
    if r == '':
        r = '.....'
    return r


def chat(model):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state('./model')
    model.saver.restore(sess, ckpt.model_checkpoint_path)

    sys.stdout.write("> ")
    sys.stdout.flush()
    line = sys.stdin.readline()

    while line:
        response = reply(model, sess, line.strip())
        sys.stdout.write(response+"\n> ")
        sys.stdout.flush()
        line = sys.stdin.readline()


path = './data/conversation.txt'
dialogue = Dialogue(path)

hparams = tf.contrib.training.HParams(total_epochs=1000,
                                      num_units=128,
                                      learning_rate=0.0001,
                                      voc_size=dialogue.voc_size,
                                      embedding_size=100,
                                      batch_size=100,
                                      total_batch=len(dialogue.seq_data) // 100 + 1)


model = Seq2Seq(hparams, 'infer')
chat(model)


