import argparse

from sklearn.datasets.base import Bunch
import numpy as np

from package.utils import standardize_string
from package.embedding import Embedding

def fetch_dataset(analogy):
    category = []
    questions = []
    answers = []
    with open(analogy) as question:
        for line in question:
            if line.startswith(":"):
                c = line.lower().split()[1]
            else:
                words = standardize_string(line).split()
                questions.append(words[0:3])
                answers.append(words[3])
                category.append(c)

    return Bunch(X=np.vstack(questions).astype("object"),
                 y=np.hstack(answers).astype("object"),
                 category=np.hstack(category).astype("object"))

def load_embedding(model, normalize=True, lower=False, clean_words=True):
    w = Embedding.from_fasttext(model)

    if normalize:
        w.normalize_words(inplace=True)
    if lower or clean_words:
        w.standardize_words(lower=lower, clean_words=clean_words, inplace=True)

    return w

def test_analogy(data, w):
    subset = [50, 1000, 4000, 10000, 14000]
    for id in subset:
        w1, w2, w3 = data.X[id][0], data.X[id][1], data.X[id][2]
        print("Question: {} is to {} as {} is to ?".format(w1, w2, w3))
        print("Answer: " + data.y[id])
        print("Predicted: " + " ".join(w.nearest_neighbors(w[w2] - w[w1] + w[w3], exclude=[w1, w2, w3])))

def main(args):
    analogy, outfile, model = args.input, args.output, args.model
    data = fetch_dataset(analogy)
    embeddings = load_embedding(model, normalize=True, lower=False, clean_words=True)
    test_analogy(data, embeddings)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    groupF = parser.add_argument_group('Files')
    groupF.add_argument("-i", "--input", default="data/questions-words.txt",
                            help="Analogy dataset")
    groupF.add_argument("-o", "--output", default="result/analogy/benchmark-results.txt",
                            help="Output directory to save the analogy results")
    groupF.add_argument("-m", "--model", default="result/pre_wiki",
                            help="Saved model with pre-trained embeddings")

    args = parser.parse_args()

    main(args)
