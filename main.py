from flask import Flask
from flask import request
from recognize import *
from file_import import *
from VrModel2 import *
import os

app = Flask(__name__)
labels, audios = get_audios_and_labels(AUDIO_PATH, LABEL_PATH)
words, word_num_map = generate_words_table(labels)


@app.route('/')
def index():
    return 'Hello World!'


@app.route('/user/<name>')
def user(name):
    return '<h1>Hello, %s!</h1>' % name


@app.route('/recognize', methods=['POST'])
def recognize():
    if request.method == 'POST':
        file = request.files['file']
        path = './audios/' + file.filename
        file.save(path)
        train_model = VrModel(False, 1, len(words) + 1)
        result = rec(path, train_model, words)
        os.remove(path)
        tf.reset_default_graph()
        return result


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
