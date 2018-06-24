import argparse
from datetime import datetime
import random
import os
import sys

sys.path.append(os.getcwd())

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
    with open(analogy, "r") as question:
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

def load_embedding(model, fasttext, normalize=True, lower=False, clean_words=True):
    """
    load_embedding(args**) -> It calls the embedding base class to load the embeddings from the saved model.
    Arguments
    ---------
    model : Saved model with the pre-trained embeddings.
    fasttext : FastText or Word2Vec
    normalize : Whether the embeddings need to be normalized or not.
    lower, clean_words : Clean the data by applying lower case and preserving '_' and '-'

    Returns
    -------
    w : Object of class Embedding
    """
    if fasttext == "fasttext":
        w = Embedding.from_fasttext(model)
    else:
        w = Embedding.from_gensim_word2vec(model)

    if normalize:
        w.normalize_words(inplace=True)
    if lower or clean_words:
        w.standardize_words(lower=lower, clean_words=clean_words, inplace=True)

    return w

def test_analogy(data, w, outfile):
    # random.seed(datetime.now())
    # subset = [random.randint(0, 19000) for _ in range(6)]
    # subset = [50, 1000, 4000, 10000, 14000]
    # for id in subset:
    #     w1, w2, w3 = data.X[id][0], data.X[id][1], data.X[id][2]
    #     print("Question: {} is to {} as {} is to ?".format(w1, w2, w3))
    #     print("Answer: " + data.y[id])
    #     print("Predicted: " + " ".join(w.nearest_neighbors(w[w2] - w[w1] + w[w3], exclude=[w1, w2, w3])))
    score, count = 0, 0
    semantic, syntactic = 0, 0
    for idx in range(0, 19544):
        w1, w2, w3 = data.X[idx][0], data.X[idx][1], data.X[idx][2]
        # print("Question: {} is to {} as {} is to ?".format(w1, w2, w3))
        # print("Answer: " + data.y[id])
        # print("Predicted: " + " ".join(w.nearest_neighbors(w[w2] - w[w1] + w[w3], exclude=[w1, w2, w3])))
        if " ".join(w.nearest_neighbors(w[w2] - w[w1] + w[w3], exclude=[w1, w2, w3])) == str(data.y[idx]):
            score += 1
            if idx < 8869:
                semantic += 1
        count += 1
        print("Total Questions : " + str(count) + " Benchmark : " + str((score/count)*100), end='\r')
    syntactic = score - semantic
    print("Total Questions : " + str(count) + " Benchmark : " + str((score/count)*100))
    with open(outfile, "w") as f:
        f.write("Total Questions: " + str(count) + " Benchmark: " + str((score/count)*100), end='\n')
        f.write("Hits @1 : " + str(score) + " Semantic: " + str(semantic) + " Syntactic: " + str(syntactic), end='\n')
    # print("\n\nTotal Questions : ", count)
    # print("Benchmark : ", (score/count)*100)

def main(args):
    analogy, outfile, model, fasttext = args.input, args.output, args.model, args.fasttext
    data = fetch_dataset(analogy)
    embeddings = load_embedding(model, fasttext, normalize=True, lower=False, clean_words=True)
    test_analogy(data, embeddings, outfile)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    groupF = parser.add_argument_group('Files')
    groupF.add_argument("-i", "--input", default="data/questions-words.txt",
                            help="Analogy dataset")
    groupF.add_argument("-o", "--output", default="result/analogy/benchmark-results.txt",
                            help="Output directory to save the analogy results")
    groupF.add_argument("-m", "--model", default="result/pre_wiki",
                            help="Saved model with pre-trained embeddings")
    groupF.add_argument("-f", "--fasttext", default="fasttext",
                            help="FastText or Word2Vec")

    args = parser.parse_args()

    main(args)
