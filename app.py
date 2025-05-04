from flask import Flask, request, jsonify, render_template, session
from flask_session import Session
from numpy.linalg import norm
import numpy as np
from scipy.spatial.distance import cosine
import json
from datetime import datetime
import os
from flask import make_response
from functools import wraps, update_wrapper
import re
from sklearn import svm
import sys
import pickle
from scipy.stats import entropy
import spacy
import subprocess
import socket

# ========== Ensure spaCy Model ==========
def ensure_spacy_model(model_name="en_core_web_sm"):
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"Model {model_name} not found. Attempting to download...")
        subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
        return spacy.load(model_name)

# ========== App Initialization ==========
app = Flask(__name__, static_url_path='', static_folder='', template_folder='templates')
app.config["SESSION_PERMANENT"] = False
app.config['SESSION_TYPE'] = 'filesystem'
app.secret_key = os.urandom(24)
Session(app)

# Global Variables
lookup = None
nlp = ensure_spacy_model()

# ========== Helper Functions ==========
def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
    return update_wrapper(no_cache, view)

def load_phonetic_embedding():
    global lookup
    path = "./data/"
    with open(path+'phonetic_embd.pickle', 'rb') as handle:
        lookup = pickle.load(handle)
    print("Phonetic embedding loaded!")
    return "success"

# ========== Routes ==========
@app.route('/')
def index():
    load_phonetic_embedding()
    return render_template('index.html')

@app.route('/get_default_content')
def get_default_content():
    path = "./data/"
    with open(path+'default_content.txt', 'r', encoding="utf8") as f:
        data = f.read()
    return data

@app.route('/update')
def update():
    text = request.args.get("text")
    easy = request.args.get("easy")
    diff = request.args.get("diff")
    thresh = float(request.args.get("thresh")) / 100

    if not text:
        print("Empty Text")
        return jsonify([])

    words, tags = parseString(text)
    res = get_hard_words(easy, diff, thresh, words, tags)
    next_word = next_uncertain_word(easy, diff)
    return jsonify({"hard_words": res, "next_word": next_word})

def uncertainity_sampling():
    clf = pickle.loads(session['model'])
    X = list(lookup.values())
    prob = clf.predict_proba(X)
    ent = entropy(prob.T)
    sorted_ind = (-ent).argsort()
    return sorted_ind

def next_uncertain_word(easy, diff):
    easy_words = easy.split(",")
    diff_words = diff.split(",")
    label_words = easy_words + diff_words
    all_words = list(lookup.keys())
    sorted_ind = uncertainity_sampling()
    for i in sorted_ind:
        word = all_words[i]
        if word not in label_words:
            break
    return all_words[i]

def get_hard_words(easy, diff, thresh, text_words, tags):
    easy = easy.replace(' ', '').split(",")
    difficult = diff.replace(' ', '').split(",")

    X, y = [], []
    for w in easy:
        word = w.upper()
        if word in lookup:
            X.append(lookup[word])
            y.append(0)
    for w in difficult:
        word = w.upper()
        if word in lookup:
            X.append(lookup[word])
            y.append(1)

    clf = svm.SVC(probability=True, random_state=0)
    clf.fit(X, y)
    session['model'] = pickle.dumps(clf)

    res = []
    word_list = []
    for w, t in zip(text_words, tags):
        w = w.upper()
        if w not in lookup:
            continue
        vec = lookup[w]
        p = round(clf.predict_proba([vec])[0][1], 2)
        if p >= thresh and w not in word_list:
            res.append((w, p, t))
            word_list.append(w)
    return res

def parseString(sentences):
    doc = nlp(sentences)
    tokens = []
    tags = []
    for i in range(len(doc)):
        tokens.append(doc[i].text)
        tags.append(doc[i].ent_type_)
    return (tokens, tags)

@app.route('/check_if_word_difficult')
def check_if_word_difficult():
    if 'model' in session:
        clf = pickle.loads(session['model'])
    else:
        print("Couldn't access session model")
        return jsonify([])

    synonyms = request.args.getlist("synonyms[]")
    thresh = float(request.args.get("thresh")) / 100

    res = []
    for w in synonyms:
        w = w.upper()
        if w not in lookup:
            continue
        vec = lookup[w]
        p = round(clf.predict_proba([vec])[0][1], 2)
        if p <= thresh:
            res.append((w, p))
    return jsonify(res)

@app.route('/getFileNames/')
def getFileNames():
    tar_path = './data/wordList/target'
    gp_path = './data/wordList/groups'
    target = os.listdir(tar_path)
    group = os.listdir(gp_path)
    return jsonify([group, target])

# ========== Run App ==========
if __name__ == '__main__':
    hostname = socket.gethostname()
    if hostname == 'ubuntuedge1':
        app.run(host='0.0.0.0', port=3999, debug=True)
    else:
        app.run(port=3999, debug=True)
