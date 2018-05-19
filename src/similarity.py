import os
import sys
import numpy as np
import math
from ranking import *

from gensim.models import FastText

def main(file, data):
	word_vecs_norm = {}
	word_vecs = {}
	model = FastText.load(file)
	wv = model.wv
	del model
	for word in wv.index2word:
		word_vecs_norm[word] = np.zeros(200, dtype=float)
		word_vecs[word] = np.zeros(200, dtype=float)
		for index,val in enumerate(wv.get_vector(word)):
			word_vecs_norm[word][index] = float(val)
			word_vecs[word][index] = float(val)
		word_vecs_norm[word] /= math.sqrt((word_vecs_norm[word]**2).sum() + 1e-6)
	train, test, train_norm = ({}, {}, {})
	not_found, total_size = (0, 0)
	with open(data, 'r') as file:
		for line in file:
			line = line.strip().lower()
			word1, word2, val = line.split()
			if word1 in word_vecs and word2 in word_vecs:
				test[(word1, word2)] = float(val)
				train[(word1, word2)] = cosine_sim(word_vecs[word1], word_vecs[word2])
				train_norm[(word1, word2)] = cosine_sim(word_vecs_norm[word1], word_vecs_norm[word2])
			else:
				not_found += 1
			total_size += 1
	print("Total Size : ", total_size, "Not Found : ", not_found)
	print("Vectors : ", spearmans_rho(assign_ranks(test), assign_ranks(train)))
	print("Normalized vectors : ", spearmans_rho(assign_ranks(test), assign_ranks(train_norm)))

if __name__ == '__main__':
	model, data = sys.argv[1], sys.argv[2]
	main(model, data)