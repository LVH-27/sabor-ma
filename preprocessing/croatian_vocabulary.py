#!/usr/bin/python2
import wikipedia
import re
from croatian_stemmer import stem_token
# from sys import stderr


def get_stop_words(stop_file):
    stop_words = []
    with open(stop_file, "r") as f:
        content = f.read().decode('iso8859_2').encode('utf-8').split('\r\n')
        for line in content:
            if len(line) == 0:
                continue
            if line[0] != ';':
                stop_words.append(line.split(';')[0].strip())
    return stop_words


stop_file = 'hrvatski_stoprijeci.txt'

wikipedia.set_lang("hr")
titles_seed = ["Hrvatski sabor",
               "Domovinski rat",
               "Hrvatska demokratska zajednica",
               "Socijaldemokratska partija Hrvatske",
               "Europska unija",
               "Recesija",
               "Inflacija"
               ]

titles_extra = []
pages = []
for title in titles_seed:
    try:
        page = wikipedia.page(title)
    except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError) as e:
        print(e)
        continue
    pages.append(page)

    links = []
    for link in page.links:
        if not re.search('[0-9]+', link):  # drop all the pages about years and dates
            if link not in titles_seed:
                links.append(link)
        if len(links) >= 20:
            break
    titles_extra.extend(links)

titles_extra = list(set(titles_extra))  # deduplication

for title in titles_extra:
    try:
        page = wikipedia.page(title)
    except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError) as e:
        print(e)
        continue
    pages.append(page)

stop_words = get_stop_words(stop_file)
# for stop_word in stop_words:
#     stderr.write(stop_word + '\n')

with open("cro_vocab.txt", "w") as f:
    for page in pages:
        content = page.content.encode('utf-8').split()
        tokens = []
        for term in content:
            if "http" not in term:
                token = term.strip('.').strip(',').strip(')').strip(':').strip('\'').strip('"').strip().lower()
                if '(' in token:
                    token = token.split('(')[1]
                if token not in tokens and token not in stop_words and not re.search('[0-9]+', token) and not re.search('\W', token):
                    tokens.append(stem_token(token))

        for token in tokens:
            token_out = token + '\n'
            f.write(token_out)
