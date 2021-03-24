import streamlit as st
import spacy
from spacy.matcher import Matcher
from wordcloud import WordCloud, ImageColorGenerator
from collections import defaultdict
from spacy.lang.en.stop_words import STOP_WORDS
import matplotlib.pyplot as plt
from nltk import ngrams
import textacy.vsm
from consultimi.preprocessor import TextPreprocessor
import pandas as pd
import graphviz as gv

import streamlit.components.v1 as components
from gchart import gchart

TOP_N_TO_SHOW_WC_SUGGESTIONS = 10

@st.cache()
def process_texts(preprocessor, raw_text):
    texts = [preprocessor.preprocess(str(openend)) for openend in raw_text.split("\n")]
    return(texts)

#@st.cache()
def get_terms_for_wordcloud(pos, corpus):
    frames = defaultdict(lambda: 0)  
    for doc in corpus:
        if ("Noun phrases" in pos):
            with doc.retokenize() as retokenizer:
                for noun_chunk in doc.noun_chunks:
                    retokenizer.merge(noun_chunk)

        matches = matcher(doc)
        spans = [doc[start:end] for _, start, end in matches]
        for span in spacy.util.filter_spans(spans):
            #print(span.text)
            if (span.text not in stop_words):
                frames[span.text] += 1 

    reduced_d = defaultdict(frames.default_factory, filter(lambda i: i[1] >= freq_cutoff, frames.items()))
    #reduced_d = {k: v for k, v in frames.items() if v >= freq_cutoff}
    return(reduced_d)

default_pos = ["Noun phrases","Nouns","Verbs","Adjectives"]

ngramdict = defaultdict(lambda: 0)  

ngrams = None
keep = None
stop_words = STOP_WORDS

st.set_page_config(layout="wide")
st.sidebar.header('Short Text Analysis Toolkit')

pos = st.sidebar.multiselect('Keep', default_pos, default=default_pos)

load_demo = st.sidebar.button("Load demo data")
st.sidebar.write("or")

if not load_demo:
    raw_text = st.sidebar.text_area('Paste opens here', height=500)
else:
    with open("example_data/sapphire_promo.txt") as f:
        raw_text = f.read()
        print(raw_text)
# st.button('Hit me')
# st.checkbox('Check me out')
#tok_method = st.sidebar.radio('Tokenisation Method', ["n-grams", "Spacy"])

#if (tok_method  == "n-grams"):
#    ngrams = st.sidebar.number_input('n', value=3)
#else:
freq_cutoff = st.sidebar.number_input('Minimum Freq', min_value=0, value=2)

color_option = st.sidebar.radio('Wordcloud colour', ["Default", "Choose one"])

if (color_option == "Choose one"):
    colors = st.sidebar.color_picker('Colour')


en = textacy.load_spacy_lang("en_core_web_sm", disable=())
matcher = Matcher(en.vocab)
if ("Noun phrases" in pos or "Nouns" in pos):
    matcher.add("Noun", [[{ "POS": "NOUN" }]])
if ("Verbs" in pos):
    matcher.add("Verbs", [[{ "POS": "VERB" }]])
if ("Adjectives" in pos):
    matcher.add("Adjectives", [[{ "POS": "ADJ" }]])

#print(keep)

# Load the texts, preprocess and store in corpus
texts = process_texts(TextPreprocessor(), raw_text)
corpus = textacy.Corpus(en, data=texts)

terms = get_terms_for_wordcloud(pos, corpus) 
if (len(terms) == 0):
    st.stop()

with st.beta_expander("Word Cloud"):
    #print(terms)
    wordcloud = WordCloud(width=1024, height=768, background_color="white", contour_width=3)
    wordcloud.generate_from_frequencies(terms)
    #print(wordcloud)
    out = plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(out)

with st.beta_expander("Word Tree"):
    #terms = get_terms_for_wordcloud(pos, corpus) 
    #candidates = sorted(terms.items(), key=lambda kv: kv[1], reverse=True)[:TOP_N_TO_SHOW_WC_SUGGESTIONS]
    #print(terms.items())
    candidates = sorted(terms.items(), key=lambda kv: kv[1], reverse=True)[:TOP_N_TO_SHOW_WC_SUGGESTIONS]
    print(candidates)

    chosen_keyword = st.selectbox("Top Keywords", options=[k for k,v in candidates])
    #vectorizer = textacy.vsm.Vectorizer(tf_type="linear", apply_idf=False, min_df=1, max_df=0.95)

    #terms_list = (doc._.to_terms_list(ngrams=2, entities=True, as_strings=True) for doc in corpus)

    #doc_term_matrix = vectorizer.fit_transform(terms_list)
    #frequencies = textacy.vsm.matrix_utils.get_term_freqs(doc_term_matrix)
    #terms_freqs = sorted(zip(frequencies,vectorizer.terms_list),reverse=True)

    #reduced_ngrams = {k: v for k, v in ngramdict.items() if "like" in k}

    #rint(reduced_ngrams.keys())
    textsout = [["Phrases"]] + [[a] for a in texts]
    print(textsout)
    gchart(key="cat_chart", data=textsout, chartType="WordTree", width='500px', height='300px', wordtree={"format": "implicit", "word": "sapphires", "type": "double"})

    #g = wordtree.draw(ngrams = reduced_ngrams.keys(), frequencies = reduced_ngrams.values(), keyword = "like")
    #st.graphviz_chart(g.source, True)

    #st.write(hide_st_style, unsafe_allow_html=True)

