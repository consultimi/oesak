import streamlit as st
import spacy
from spacy.matcher import Matcher
from wordcloud import WordCloud, ImageColorGenerator
from collections import defaultdict
from spacy.lang.en.stop_words import STOP_WORDS
import matplotlib.pyplot as plt
from nltk import ngrams
import textacy
import textacy.vsm
import itertoolz
import wordtree

default_pos = ["Noun phrases","Nouns","Verbs","Adjectives"]

frames = defaultdict(lambda: 0)  
ngramdict = defaultdict(lambda: 0)  

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
rawdocs = [str(openend).strip().lower() for openend in raw_text.split("\n")]
corpus = textacy.Corpus(en, data=rawdocs)
for doc in corpus:
    #openend = 
    #doc = nlp(openend)
    for span in textacy.extract.ngrams(doc, 2, min_freq=1):
        out = tuple(token.text for token in span)
        ngramdict[out] += 1


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
        #print(span.text)
        if (span.text not in stop_words):
            frames[span.text] += 1 

reduced_d = {k: v for k, v in frames.items() if v >= freq_cutoff}

print()
vectorizer = textacy.vsm.Vectorizer(tf_type="linear", apply_idf=False, min_df=1, max_df=0.95)

terms_list = (doc._.to_terms_list(ngrams=2, entities=True, as_strings=True) for doc in corpus)

doc_term_matrix = vectorizer.fit_transform(terms_list)
frequencies = textacy.vsm.matrix_utils.get_term_freqs(doc_term_matrix)
terms_freqs = sorted(zip(frequencies,vectorizer.terms_list),reverse=True)
#print(terms_freqs)
#print(ngramdict.keys())

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

    #reduced_ngrams = {k: v for k, v in ngramdict.items() if "like" in k}

    #rint(reduced_ngrams.keys())
    g = wordtree.search_and_draw(corpus = rawdocs, keyword = "coffee", max_n=6)

    #g = wordtree.draw(ngrams = reduced_ngrams.keys(), frequencies = reduced_ngrams.values(), keyword = "like")
    st.graphviz_chart(g, True)

else:
    st.write("No tokens created. Please check the options and try again.")