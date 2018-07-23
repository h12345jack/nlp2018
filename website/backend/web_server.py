# -*- coding=utf-8 -*-


import time

from flask import Flask
from flask import render_template
from flask import jsonify
from flask_cors import CORS
from flask import request


app = Flask(__name__, template_folder = 'website/build', static_folder='website/build/static') 
CORS(app)

def cut(sentence, model, dataset):
	if model == 'mm':
		pass
	elif model == 'perceptron':
		pass
	elif model == 'hmm':
		pass
	elif model == 'crf':
		pass
	elif model == 'bilstm':
		pass

@app.route('/api/cws', methods=['POST', 'GET'])
def index():
	if request.method == 'POST':
		sentence = request.form.get('sentence', '')
		model = request.form.get('model', None)
		data = request.form.get('data', '')
		cutted = cut(sentence, model, data)
		return jsonify(result=cutted)
	else:
		return render_template('index.html')

@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)