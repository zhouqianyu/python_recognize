from deep_learning import *
AUDIO_PATH = 'train/'
LABEL_PATH = 'data/'
model_save_path = 'saver/'


def rec(path, train_model, words):
    mfcc = get_audio_mfcc(path, 26, 9)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_save_path)
    with tf.Session() as sess:
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        dict_map = {train_model.inputs: [mfcc],
                    train_model.seq_length: [len(mfcc)],
                    train_model.keep_dropout: 1}
        result = train_model.run(sess, dict_map=dict_map)
        result = sparse_tuple_to_labels(result, words)
        print(result)
        return result[0]
