#!/usr/bin/python
import wikipedia


wikipedia.set_lang("hr")
titles = ["Hrvatski sabor",
          "Domovinski rat",
          "Hrvatska demokratska zajednica",
          "Socijaldemokratska partija Hrvatske",
          "Europska unija",
          "Recesija",
          "Inflacija"
          ]

pages = [wikipedia.page(title) for title in titles]

with open("cro_vocab.txt", "w") as f:
    for page in pages:
        summary = page.summary.split()
        tokens = []
        for term in summary:
            if "http" not in term:
                tokens.append(term.split('.')[0])

        for token in tokens:
            f.write(token + '\n')
