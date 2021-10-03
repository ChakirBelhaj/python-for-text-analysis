from utils_3a import pre_process

# source https://www.geeksforgeeks.org/find-frequency-of-each-word-in-a-string-in-python/
def count(text) :
    text = pre_process(text)
    str_list = str.split(text)
    unique_words = set(str_list)
      
    for words in unique_words :
        print('Frequency of ', words , 'is :', str_list.count(words))

text = input("Type in a text that has to be filtered and counted \n")
count(text)