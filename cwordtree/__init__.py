from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict

import pandas as pd
from tqdm.auto import tqdm

#from graphviz import Digraph
from streamlit_agraph import agraph, Node, Edge, Config

test_svg = """
    <svg viewBox="0 0 240 80" xmlns="http://www.w3.org/2000/svg">
    <style>
        .small { font: italic 13px sans-serif; }
        .heavy { font: bold 30px sans-serif; }
    </style>

    <text x="20" y="35" class="small">My</text>
    <text x="40" y="35" class="heavy">cat</text>
    <text x="55" y="55" class="small">is</text>
    <text x="65" y="55" class="Rrrrr">Grumpy!</text>
    </svg>
"""

class Direction(Enum):
    Forward = 1
    Backward = 2


@dataclass
class FreqNode:
    freq: int
    children: Dict[str, "FreqNode"]


def build_tree(ngrams, frequencies):
    tree = FreqNode(freq=0, children={})
    for ngram, freq in zip(ngrams, frequencies):
        subtree = tree
        for gram in ngram:
            if gram not in subtree.children:
                subtree.children[gram] = FreqNode(children={}, freq=freq)
            subtree = subtree.children[gram]
        subtree.freq = freq
    return tree


def build_both_trees(keyword, ngrams, frequencies):
    fwd_ngrams, fwd_frequencies = [], []
    bwd_ngrams, bwd_frequencies = [], []

    for ngram, freq in zip(ngrams, frequencies):
        fwd = ngram[0] == keyword
        bwd = ngram[-1] == keyword
        assert fwd or bwd, "ngram does not have keyword at beginning or end: {}".format(ngram)

        if fwd:
            fwd_ngrams.append(ngram[1:])
            fwd_frequencies.append(freq)
        if bwd:
            bwd_ngrams.append(reversed(ngram[:-1]))
            bwd_frequencies.append(freq)

    fwd_tree = build_tree(fwd_ngrams, fwd_frequencies)
    bwd_tree = build_tree(bwd_ngrams, bwd_frequencies)

    return fwd_tree, bwd_tree


class TreeDrawer:
    def __init__(self, keyword, fwd_tree, bwd_tree, max_font_size=30, min_font_size=12, font_interp=None):
        self.max_font_size = max_font_size
        self.min_font_size = min_font_size
        self.keyword = keyword
        self.fwd_tree = fwd_tree
        self.bwd_tree = bwd_tree
        self.max_freq = max([t.freq for t in fwd_tree.children.values()] +
                            [t.freq for t in bwd_tree.children.values()])
        self.font_interp = None

        self.nodes = []
        self.edges = []

        self.config = Config(width=1000, 
                height=1000, 
                directed=True,
                nodeHighlightBehavior=True, 
                highlightColor="#ffffff", # or "blue"
                strokeColor="#ffffff",
                collapsible=False,
                staticGraph=False,
                node={'labelProperty':'label', 'highlightColor': "#ffffff", 'color': "#ffffff", 'labelPosition': 'center', 'fontSize': 12},
                link={'labelProperty': 'label', 'renderLabel': True}
                # **kwargs e.g. node_size=1000 or node_color="blue"
                ) 
        #self.graph = Digraph(keyword, format='svg')
        #self.graph.attr('graph', rankdir='LR')
        #self.graph.attr('node', shape='plaintext', margin='0')

    def interpolate_fontsize(self, freq):
        lower = self.min_font_size
        upper = self.max_font_size
        t = freq / self.max_freq

        def quad(t):
            return t ** (1. / 3)

        font_interp = quad if self.font_interp is None else self.font_interp

        return int(font_interp(t) * (upper - lower) + lower)


    def draw_subtree(self, tree, direction, root, suffix, depth):
        if depth > 0:
            fontsize = self.interpolate_fontsize(tree.freq)
            self.nodes.append(Node(id=root + suffix, label=root, labelPosition='center', color='#FFFFFF', fontSize=fontsize))
            #self.graph.node(root + suffix, label=root, fontsize=str(fontsize))

        for word, subtree in tree.children.items():
            new_suffix = '{}-{}'.format(suffix, word)
            self.draw_subtree(subtree, direction, word, new_suffix, depth + 1)
            src = root if depth == 0 else root + suffix
            dst = word + new_suffix
            if direction == Direction.Forward:
                self.edges.append(Edge(source=src, target=dst))
                #self.graph.edge(src, dst)
            else:
                self.edges.append(Edge(source=dst, target=src))
                #self.graph.edge(dst, src)

    def draw(self):
        self.nodes.append(Node(id=self.keyword, label=self.keyword, labelPosition='center', color='#FFFFFF', fontSize=str(self.max_font_size)))
        self.draw_subtree(self.bwd_tree, Direction.Backward, self.keyword, "-bwd", 0)
        self.draw_subtree(self.fwd_tree, Direction.Forward, self.keyword, "-fwd", 0)
        return agraph(nodes=self.nodes, edges=self.edges, config=self.config)


def draw(keyword, ngrams, frequencies, max_per_n=8, **kwargs):
    df = pd.DataFrame([
        {'ngram': ngram, 'n': len(ngram), 'forward': ngram[0] == keyword,
         'freq': freq}
        for ngram, freq in zip(ngrams, frequencies)
    ])

    filtered_df = df.sort_values('freq', ascending=False) \
                    .groupby(['forward', 'n']) \
                    .head(max_per_n) \
                    .reset_index()

    ngrams = filtered_df.ngram.tolist()
    frequencies = filtered_df.freq.tolist()

    fwd_tree, bwd_tree = build_both_trees(keyword, ngrams, frequencies)
    t = TreeDrawer(keyword, fwd_tree, bwd_tree, **kwargs)
    return t.draw()


def search(corpus, keyword, max_n=5, tokenizer=None):
    if tokenizer is None:
        tokenizer = lambda s: [t.strip() for t in s.lower().split(' ') if t.strip() is not None]

    frequencies_dict = defaultdict(int)
    for doc in tqdm(corpus):
        tokens = tokenizer(doc)
        for n in range(2, max_n + 1):
            for i in range(0, len(tokens) - n + 1):
                ngram = tokens[i:i + n]
                if ngram[0] == keyword or ngram[-1] == keyword:
                    frequencies_dict[tuple(ngram)] += 1

    ngrams = []
    frequencies = []
    for ngram, freq in frequencies_dict.items():
        ngrams.append(ngram)
        frequencies.append(freq)

    return ngrams, frequencies

def search_and_draw(corpus, keyword, max_n=5, tokenizer=None, **kwargs):
    ngrams, frequencies = search(corpus, keyword, max_n=max_n, tokenizer=tokenizer)
    return draw(keyword, ngrams, frequencies, **kwargs)
