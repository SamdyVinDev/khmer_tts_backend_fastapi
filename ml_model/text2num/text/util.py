import re


def get_whole_word_regx(regx: str):
    return r' %s |^%s|%s$' % (regx, regx, regx)


def remove_zero_space(text: str):
    return re.sub(r'\u200b', '', text)


def remove_words_w_bracket(text: str):
    return re.sub(r'\(.*\)', '', text)


def remove_url(text: str):
    return re.sub(r'https?:\/\/.*[\r\n]*', '', text)


def replace_words(text: str, words: [str], replaces: [str]):
    for word, replace in zip(words, replaces):
        word = re.escape(word)
        text = re.sub(get_whole_word_regx(word), '%s' % replace, text)


def lekto2text(text: str):
    lek_tos = re.findall(r'(([ក-៩]+) ?ៗ)', text)
    for lek_to, word in lek_tos:
        text = re.sub(get_whole_word_regx(lek_to), '%s %s' % (word, word), text, 1)
