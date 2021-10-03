from utils import get_paths
from utils import get_basic_stats

input_folder = "../Data/books"


def printTextFromFolder(folder):
    for file in get_paths(folder):
        saved = get_basic_stats(file)
        print("fileName = " + file)
        print("sentences = " + str(saved['num_sents']))
        print("words = " + str(saved['num_tokens']))
        print("unique = " + str(saved['vocab_size']))
        print("num_chapters_or_acts = " + str(saved['num_chapters_or_acts']))
        print("\n")


printTextFromFolder("../Data/books")