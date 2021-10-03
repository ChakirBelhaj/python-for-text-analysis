import glob
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
# https://datatofish.com/txt-files-directory-python/
def get_paths(input_folder):
    return glob.glob(input_folder + "/*.txt")
    
def get_basic_stats(txt_path):
    with open(txt_path, 'r', encoding="utf8") as file:
        data = file.read()
        wordTokenizedData = word_tokenize(data)
        saved = {'num_chapters_or_acts' : 0}
        saved['num_sents'] = len(sent_tokenize(data))
        saved['num_tokens'] = len(wordTokenizedData)
        saved['vocab_size'] = len(set(wordTokenizedData))
        if txt_path == "../Data/books\HuckFinn.txt" : saved['num_chapters_or_acts'] = wordTokenizedData.count("CHAPTER")
        if txt_path == "../Data/books\AnnaKarenina.txt" : saved['num_chapters_or_acts'] = wordTokenizedData.count("Chapter")
        if txt_path == "../Data/books\Macbeth.txt" :  saved['num_chapters_or_acts'] = wordTokenizedData.count("ACT")
        return saved




