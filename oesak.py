import streamlit as st
import spacy
from spacy.matcher import Matcher
from wordcloud import WordCloud, ImageColorGenerator
from collections import defaultdict
from spacy.lang.en.stop_words import STOP_WORDS
import matplotlib.pyplot as plt
from nltk import ngrams
import textacy

default_pos = ["Noun phrases","Nouns","Verbs","Adjectives"]

frames = defaultdict(lambda: 0)  
ngrams = None
keep = None
stop_words = STOP_WORDS

st.sidebar.header('Open-End Swiss Army Knife')

pos = st.sidebar.multiselect('Keep', default_pos, default=default_pos)

raw_text = st.sidebar.text_area('Paste opens here', height=500)

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
corpus = textacy.Corpus(en, data=[str(openend).strip().lower() for openend in raw_text.split("\n")])
for doc in corpus:
    #openend = 
    #doc = nlp(openend)
    if ("Noun phrases" in pos):
        with doc.retokenize() as retokenizer:
            for noun_chunk in doc.noun_chunks:
                retokenizer.merge(noun_chunk)
    # make ngrams
    #ngram_out = ngrams([t.text for t in doc])
    #ngram_out = ngrams("blah", 3)
    matches = matcher(doc)
    spans = [doc[start:end] for _, start, end in matches]
    for span in spacy.util.filter_spans(spans):
        print(span.text)
        if (span.text not in stop_words):
            frames[span.text] += 1 

reduced_d = {k: v for k, v in frames.items() if v >= freq_cutoff}

if (len(reduced_d) > 0):
    wordcloud = WordCloud(width=1600, height=800,collocations=False, 
        background_color="white", 
        contour_width=3)

    wordcloud.generate_from_frequencies(reduced_d)
    #print(wordcloud)
    out = plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(out)
else:
    st.write("No tokens created. Please check the options and try again.")