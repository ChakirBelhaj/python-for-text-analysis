def pre_process(text, chars_to_remove={'\n', ',', '.', '"', "(" ,")"}) :
    for item in chars_to_remove:
        text = text.replace(item, "")
    return text
