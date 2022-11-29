#!/usr/bin/env python3
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec
import networkx as nx
import matplotlib.pyplot as plt
from queue import Queue
import datetime
import argparse
import os
import getpass
import random



class AnimateSemanticTree:
    def __init__(self, nviz, sim, model, max_size=50):
        self.nviz = nviz
        self.sim = sim
        self.max_size = max_size
        self.Q = Queue()
        self.tree = nx.DiGraph()
        self.scanned = []
        self.root = None
        self.model = model
        self.model_type = None



    def get_neighbors(self, nviz, sim):
        word = self.Q.get(timeout=1)
        if word in self.scanned:
            viz = []
        else:
            viz = [(v, s) for v, s in self.model.wv.most_similar(word, topn=nviz) if s >= sim]
            self.scanned.append(word)
            self.make_edges(word, viz)
        # print("Neighbors of {}: ".format(word), viz)
        return viz

    def make_edges(self, wo, viz):
        # print('adding', viz)
        self.tree.add_weighted_edges_from([(wo, v, w) for v, w in viz])
        # print('Edges: ', self.tree.edges())

    def build_document_neighbors(self, docid):
        pass

    def build_neighbors(self, word):
        try:
            assert word in self.model.wv.key_to_index
        except AssertionError:
            newword = list(self.model.wv.key_to_index.keys())[random.randint(0,len(self.model.wv.key_to_index)-1)]
            print(f"Word {word} not in vocabulary trying {newword}")
            word = newword
        print("Building graph...")
        self.root = word
        self.tree = nx.Graph()
        self.tree.add_node(word, **{'color': 'blue'})
        self.Q.put(word)
        while self.tree.size() < self.max_size:
            viz = self.get_neighbors(self.nviz, self.sim)
            for w in viz:
                if w[0] not in self.scanned:
                    self.Q.put(w[0])
            if self.Q.empty():
                break
        self.scanned += list(set(self.tree.nodes())-set(self.scanned))
        return self.tree

def draw_tree(word, tree, nviz, sim, size):
    print("Drawing graph...")
    cols = ['r'] * len(tree.nodes());
    cols[tree.nodes().index(word)] = 'b'
    pos = nx.spring_layout(tree, iterations=500)
    nx.draw_networkx(tree, pos=pos, node_color=cols, node_size=1000, alpha=0.5, font_size=14)
    plt.title("Semantic Neighborhood of '{}'\n max. neighbors: {}, min. similarity: {}, size: {}".format(word, nviz, sim, size))
    plt.show()
    nx.write_gml(tree, "{}_neighborhood.gml".format(word))

def send_to_gource(tree):
    user = getpass.getuser()
    now = int(datetime.datetime.now().timestamp())
    with open('gource_log.log', 'w') as f:
        f.write('{}|{}|A|{}\n'.format(now, user, tree.root))
        for w in tree.scanned:
            now += 1
            path = nx.shortest_path(tree.tree, tree.root, w)
            f.write('{}|{}|A|{}\n'.format(now, user, '/'.join(path)))

def load_model(model_fname):
    print("Loading Data...")
    try:
        model = Word2Vec.load(model_fname)
        print("Detected Word2Vec model")
        model_type = 'w2v'
    except:
        model = Doc2Vec.load(model_fname)
        print("Detected Doc2Vec model")
        model_type = 'd2v'
    return model

def demo():
    gcommand = "gource --realtime --title \"SemanticTree\" --hide date --log-format custom --auto-skip-seconds 1 -"
    import gensim.downloader as api
    print("Loading Google News 300 w2v model. It may take a few minutes depending on your connection.")
    wv = api.load('word2vec-google-news-300')
    AT = AnimateSemanticTree(nviz=15, sim=0.3, model=wv, max_size=50)
    g = AT.build_neighbors('king')

    send_to_gource(AT)
    os.system('cat gource_log.log |{}'.format(gcommand))

def run():
    gcommand = "gource --realtime --title \"Semtree\" --hide date --log-format custom --auto-skip-seconds 0.3 -"
    parser = argparse.ArgumentParser(
        description="Visualize the semantic neighborhood of a term. given a Gensim's Word2vec trained model")
    parser.add_argument('model', type=str, help='file name of a Gensim\'s Word2vec model.')
    parser.add_argument('-w', help='the word or bi_gram to analyze')
    parser.add_argument('-n', type=int, default=15, help='Max number of neighbors to scan.')
    parser.add_argument('-s', '--sim', type=float, default=0.3, help='minimum similarity to be a neighbor')
    parser.add_argument('-S', '--size', type=int, default=500, help='Maximum size of the three')
    parser.add_argument('-o', '--output', type=str, choices=['image', 'animation'], default='animation',
                        help='Type of visualization')

    args = parser.parse_args()
    model = load_model(args.model)
    AT = AnimateSemanticTree(args.n, args.sim, model, args.size)
    word = 'israel'

    g = AT.build_neighbors(args.w)

    if args.output == 'image':
        draw_tree(args.w, g, args.n, args.sim, args.size)
    else:
        send_to_gource(AT)
        os.system('cat gource_log.log |{}'.format(gcommand))

if __name__ == "__main__":
    run()
