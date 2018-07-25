# -*- coding=utf-8 -*-


import time

from flask import Flask
from flask import render_template
from flask import jsonify
from flask_cors import CORS
from flask import request


from models.maxmatch.max_match import MaxMatch

app = Flask(__name__, template_folder = 'website/build', static_folder='website/build/static') 
CORS(app)

def cut(sentence, model, dataset):
	if model == 'mm':
		print(sentence, 20)
		tmp = MaxMatch(dataset)
		rs = tmp.cut(sentence)
		return rs
	elif model == 'perceptron':
		pass
	elif model == 'hmm':
		pass
	elif model == 'crf':
		pass
	elif model == 'bilstm':
		pass

@app.route('/api/cws', methods=['POST', 'GET'])
def cws():
	if request.method == 'POST':
		print(request, 36)
		req = request.json
		print(req, 37)
		sentence = req.get('sentence', '')
		model = req.get('model', 'mm')
		dataset = req.get('dataset', 'pku')
		cutted = cut(sentence, model, dataset)
		print(cutted, 39)
		return jsonify(result=cutted)
	else:
		return render_template('index.html')

@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")



if __name__ == '__main__':
    app.run(debug=True)