import argparse
from datetime import datetime
import random

from sklearn.datasets.base import Bunch
import numpy as np

from package.utils import standardize_string
from package.embedding import Embedding

def fetch_dataset(analogy):
    """
    fetch_dataset(analogy) -> Takes the analogy dataset as input. Make sure the dataset is clean and a simple text file.
    Arguments
    ---------
    analogy : Input analogy dataset

    Returns
    -------
    X : matrix of word questions
    y : vector of answers
    category : name of category
    """
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
    """
    load_embedding(args**) -> It calls the embedding base class to load the embeddings from the saved model.
    Arguments
    ---------
    model : Saved model with the pre-trained embeddings.
    normalize : Whether the embeddings need to be normalized or not.
    lower, clean_words : Clean the data by applying lower case and preserving '_' and '-'

    Returns
    -------
    w : Object of class Embedding
    """
    w = Embedding.from_fasttext(model)

    if normalize:
        w.normalize_words(inplace=True)
    if lower or clean_words:
        w.standardize_words(lower=lower, clean_words=clean_words, inplace=True)

    return w

def test_analogy(data, w):
    random.seed(datetime.now())
    subset = [random.randint(0, 19000) for _ in range(6)]
    # subset = [50, 1000, 4000, 10000, 14000]
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
