import streamlit as st
import spacy
from spacy.matcher import Matcher
from collections import defaultdict
import pandas as pd
import streamlit.components.v1 as components
import streamlit_gchart as gchart
import streamlit_wordcloud as wordcloud
from consultimi.preprocessor import TextPreprocessor
import textacy
TOP_N_TO_SHOW_WC_SUGGESTIONS = 10

@st.cache()
def process_texts(preprocessor, raw_text):
    texts = [preprocessor.preprocess(str(openend)) for openend in raw_text.split("\n")]
    return(texts)

@st.cache()
def load_demo_data():
    raw_text = ''
    with open("example_data/sapphire_promo.txt") as f:
        raw_text = f.read()
    return(raw_text)

#@st.cache()
def get_terms_for_wordcloud(matcher, corpus, stop_words=[]):
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

def get_matcher(en, pos):
    matcher = Matcher(en.vocab)
    if ("Noun phrases" in pos or "Nouns" in pos):
        matcher.add("Noun", [[{ "POS": "NOUN" }]])
    if ("Verbs" in pos):
        matcher.add("Verbs", [[{ "POS": "VERB" }]])
    if ("Adjectives" in pos):
        matcher.add("Adjectives", [[{ "POS": "ADJ" }]])
    return(matcher)

default_pos = ["Noun phrases","Nouns","Verbs","Adjectives"]


#st.set_page_config(layout="wide")
st.sidebar.header('Short Text Analysis Toolkit')

data_source = st.sidebar.radio('Data source', ['Paste my own data', 'Use demo data'])

if (data_source == 'Paste my own data'):
    raw_text = st.sidebar.text_area('Paste texts here', height=500, key="textin")
    process = st.sidebar.button("Process Texts")
else:
    raw_text = load_demo_data()

if (raw_text is None or raw_text == ''):
    st.stop()

texts = process_texts(TextPreprocessor(), raw_text)

focus_word = 'the'
with st.beta_expander("Word Cloud"):
    left_column, right_column = st.beta_columns(2)
    pos = None
    with left_column:
        pos = st.multiselect('Keep', default_pos, default=default_pos)

    with right_column:      
        freq_cutoff = st.number_input('Minimum Token Freq', min_value=0, value=2)
  
        #color_option = st.radio('Wordcloud colour', ["Default", "Choose one"])

        #if (color_option == "Choose one"):
        #    colors = st.color_picker('Colour')

    en = textacy.load_spacy_lang("en_core_web_sm", disable=())
    matcher = get_matcher(en, pos)
    corpus = textacy.Corpus(en, data=texts)
    terms = get_terms_for_wordcloud(matcher, corpus) 

    if (len(terms) > 0):
        wordcloud_terms = [ dict(text=k, value=v) for k,v in terms.items() ]
        return_obj = wordcloud.visualize(wordcloud_terms, tooltip_data_fields={
            'text':'Term', 'value':'Frequency'
        }, per_word_coloring=False)
        #print(clicked['clicked'])
        if (return_obj != None and 'clicked' in return_obj):
            if (return_obj['clicked'] != None):
                focus_word = return_obj['clicked']['text']


with st.beta_expander("Word Tree"):

    textsout = [["Phrases"]] + [[a] for a in texts]
    #print(textsout)
    gchart.gchart(key="cat_chart", data=textsout, chartType="WordTree", width='500px', height='300px', wordtree={"format": "implicit", "word": focus_word, "type": "double"})

