import argparse
import logging
from gensim.models.word2vec import LineSentence
from gensim.models import FastText

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO

def train(infile, outfile, skipgram, loss, size, epochs):
	sentence = LineSentence(infile)

	model = FastText(sentence, sg=skipgram, hs=loss, size=size, alpha=0.05, window=5,
					min_count=5, min_n=2, max_n=5, workers=3, iter=epochs)

	model.save(outfile)

def main(args):
	infile, outfile = args.input, args.output
	skipgram, loss, size, epochs = int(args.skipgram), int(args.loss), int(args.size), int(args.epochs)
	train(infile, outfile, skipgram, loss, size, epochs)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	groupF = parser.add_argument_group('Files')
	groupF.add_argument("-i", "--input", default="data/file9",
							help="Clean wiki dump file with text")
	groupF.add_argument("-o", "--output", default="model/pre_wiki",
							help="Output directory to save the model")
	groupM = parser.add_argument_group('Model')
	groupM.add_argument("-s", "--size", default="200",
							help="Embedding size")
	groupM.add_argument("-sg", "--skipgram", default="1",
							help="Model uses skipgram or CBOW")
	groupM.add_argument("-hs", "--loss", default="1",
							help="Hierarichal softmax or Negative Sampling as loss function")
	groupM.add_argument("-e", "--epochs", default="3",
							help="Number of epochs")

	args = parser.parse_args()

	main(args)