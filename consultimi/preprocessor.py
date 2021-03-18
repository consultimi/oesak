from textacy import preprocessing.normalize, preprocessing.remove
replacements = {


}

# "Heavily inspired" https://github.com/cbaziotis/ekphrasis/blob/master/ekphrasis/classes/preprocessor.py

class TextPreprocessor:
    def __init__(self, **kwargs):
        return

    def preprocess(self, text):
        out_text = text.strip().lower()
        out_text = textacy.preprocessing.normalize.normalize_unicode(out_text, "NFKC")
        out_text = textacy.preprocessing.remove.remove_punctuation(out_text)
        return(out_text)


    def unpack_contractions(self, text):
        # Taken from https://github.com/cbaziotis/ekphrasis/blob/master/ekphrasis/utils/nlp.py (in turn taken from an older version of textacy!)

        text = re.sub(
            r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]ould)n't",
            r"\1\2 not", text)
        text = re.sub(
            r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'ll",
            r"\1\2 will", text)
        text = re.sub(r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'re", r"\1\2 are",
                    text)
        text = re.sub(
            r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou)'ve",
            r"\1\2 have", text)
        # non-standard
        text = re.sub(r"(\b)([Cc]a)n't", r"\1\2n not", text)
        text = re.sub(r"(\b)([Ii])'m", r"\1\2 am", text)
        text = re.sub(r"(\b)([Ll]et)'s", r"\1\2 us", text)
        text = re.sub(r"(\b)([Ww])on't", r"\1\2ill not", text)
        text = re.sub(r"(\b)([Ss])han't", r"\1\2hall not", text)
        text = re.sub(r"(\b)([Yy])(?:'all|a'll)", r"\1\2ou all", text)
        return text