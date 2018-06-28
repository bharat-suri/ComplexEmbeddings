import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.datasets.base import Bunch
import json
from gensim.models import FastText
import logging
import sys
import argparse

torch.manual_seed(1)

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO

wv = None

def load_model(savedModel):
	global wv
	model = FastText.load(savedModel)
	wv = model.wv
	del model

def load_mappings(descriptions):
	global wv
	count = 0
	shape = (500, 300)
	entities = []
	abstracts = []
	with open(descriptions, 'r') as file:
		logging.info('mapping the resources and their descriptions')
		for line in file:
			if count % 5000 == 0:
				logging.info('reading resource # {0}'.format(count))
			jsonline = json.loads(line)		
			ent = [_ for _ in jsonline.keys()][0]
			desc = jsonline[ent].split()
			try:
				t = np.zeros(shape, dtype=np.float32)
				v = np.array(wv.get_vector(ent), dtype=np.float32)
				r = np.array(list(map(lambda x: wv.get_vector(x), desc)), dtype=np.float32)
				t[:r.shape[0], :r.shape[1]] = r
				entities.append(v)
				abstracts.append(t)
				count += 1
			except:
				continue
	del wv
	logging.info('resources read into np stack of length : {0}'.format(count))
	entities = np.stack(entities)
	abstracts = np.stack(abstracts)
	logging.info('saving entity embeddings to data/entity.npy')
	np.save('../data/entity.npy', entities)
	logging.info('saving description embeddings to data/description.npy')
	np.save('../data/description.npy', abstracts)
	# return Bunch(X=np.vstack(abstracts).astype("object"),
	# 			 y=np.hstack(entities).astype("object"))

def create_tensors():
	pass

def main(savedModel, descriptions):
	global wv
	load_model(savedModel)
	mappings = load_mappings(descriptions)

if __name__ == '__main__':
	mode = sys.argv[1]
	if mode == 'save':
		savedModel, descriptions = sys.argv[2], sys.argv[3]
		main(savedModel, descriptions)
	elif mode == 'load':
		create_tensors()