import streamlit as st
import spacy
from spacy.matcher import Matcher
from wordcloud import WordCloud, ImageColorGenerator
from collections import defaultdict
from spacy.lang.en.stop_words import STOP_WORDS
import matplotlib.pyplot as plt

frames = defaultdict(lambda: 0)  
ngrams = None
keep = None
stop_words = STOP_WORDS

st.sidebar.header('Open-End Swiss Army Knife')

raw_text = st.sidebar.text_area('Paste opens here', height=500)

spacy_options = ["Noun phrases","Nouns","Verbs","Adjectives"]
# st.button('Hit me')
# st.checkbox('Check me out')
#tok_method = st.sidebar.radio('Tokenisation Method', ["n-grams", "Spacy"])

#if (tok_method  == "n-grams"):
#    ngrams = st.sidebar.number_input('n', value=3)
#else:
keep = st.sidebar.multiselect('Keep', spacy_options, default=spacy_options)

freq_cutoff = st.sidebar.number_input('Minimum Freq', min_value=0, value=2)

nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)
if ("Noun phrases" in keep or "Nouns" in keep):
    matcher.add("Noun", [[{ "POS": "NOUN" }]])
if ("Verbs" in keep):
    matcher.add("Verbs", [[{ "POS": "VERB" }]])
if ("Adjectives" in keep):
    matcher.add("Verbs", [[{ "POS": "ADJ" }]])

print(keep)
for openend in raw_text.split("\n"):
    openend = str(openend).strip().lower()
    doc = nlp(openend)
    if ("Noun phrases" in keep):
        with doc.retokenize() as retokenizer:
            for noun_chunk in doc.noun_chunks:
                retokenizer.merge(noun_chunk)
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
    print(wordcloud)
    out = plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(out)
else:
    st.write("No tokens created. Please check the options and try again.")