import json
import spacy
nlp = spacy.load('en_core_web_sm')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
vader_model = SentimentIntensityAnalyzer()
def run_vader(nlp,
              textual_unit, 
              lemmatize=False, 
              parts_of_speech_to_consider=set(),
              verbose=0):
    """
    Run VADER on a sentence from spacy
    
    :param str textual unit: a textual unit, e.g., sentence, sentences (one string)
    (by looping over doc.sents)
    :param bool lemmatize: If True, provide lemmas to VADER instead of words
    :param set parts_of_speech_to_consider:
    -empty set -> all parts of speech are provided
    -non-empty set: only these parts of speech are considered
    :param int verbose: if set to 1, information is printed
    about input and output
    
    :rtype: dict
    :return: vader output dict
    """
    doc = nlp(textual_unit)
        
    input_to_vader = []

    for sent in doc.sents:
        for token in sent:
            
            if verbose >= 2:
                print(token, token.pos_)

            to_add = token.text

            if lemmatize:
                to_add = token.lemma_

                if to_add == '-PRON-': 
                    to_add = token.text

            if parts_of_speech_to_consider:
                if token.pos_ in parts_of_speech_to_consider:
                    input_to_vader.append(to_add) 
            else:
                input_to_vader.append(to_add)

    scores = vader_model.polarity_scores(' '.join(input_to_vader))
    
    if verbose >= 1:
        print()
        print('INPUT SENTENCE', sent)
        print('INPUT TO VADER', input_to_vader)
        print('VADER OUTPUT', scores)

    return scores

def vader_output_to_label(vader_output):
    """
    map vader output e.g.,
    {'neg': 0.0, 'neu': 0.0, 'pos': 1.0, 'compound': 0.4215}
    to one of the following values:
    a) positive float -> 'positive'
    b) 0.0 -> 'neutral'
    c) negative float -> 'negative'
    
    :param dict vader_output: output dict from vader
    
    :rtype: str
    :return: 'negative' | 'neutral' | 'positive'
    """
    compound = vader_output['compound']
    
    if compound < 0:
        return 'negative'
    elif compound == 0.0:
        return 'neutral'
    elif compound > 0.0:
        return 'positive'
    
assert vader_output_to_label( {'neg': 0.0, 'neu': 0.0, 'pos': 1.0, 'compound': 0.0}) == 'neutral'
assert vader_output_to_label( {'neg': 0.0, 'neu': 0.0, 'pos': 1.0, 'compound': 0.01}) == 'positive'
assert vader_output_to_label( {'neg': 0.0, 'neu': 0.0, 'pos': 1.0, 'compound': -0.01}) == 'negative'



my_tweets = json.load(open('my_tweets.json'))
tweets = []
all_vader_output = []
manual_annotation = []

for id_, tweet_info in enumerate(my_tweets):
    the_tweet = tweet_info['text_of_tweet']
    sentence = tweet_info['text_of_tweet']
    vader_output = run_vader(nlp, sentence, parts_of_speech_to_consider=set(["love","hate"]),)
    vader_label = vader_output_to_label(vader_output)
    
    tweets.append(the_tweet)
    all_vader_output.append(vader_label)
    manual_annotation.append(tweet_info['sentiment_label'])
    


correct_instances = 0
#https://stackoverflow.com/questions/522563/accessing-the-index-in-for-loops
for index, item in enumerate(all_vader_output):
    if item == manual_annotation[index]:
        correct_instances += 1

accuracy = correct_instances / len(my_tweets)
print("Vader accuracy = " + str(int(100 * accuracy)) + "%")

