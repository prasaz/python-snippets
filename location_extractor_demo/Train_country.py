# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 19:00:33 2022

@author: wijepra
"""

import spacy
from spacy.tokens import DocBin
import geonamescache

geodata = geonamescache.GeonamesCache()

nlp = spacy.blank("en")
country_list = [x for x in geodata.get_countries_by_names()]

# training_data = [(c, [(0, len(c), 'COUNTRY')]) for c in country_list]

training_data = [
  ("Tokyo Tower is 333m tall.", [(0, 11, "BUILDING")]),
]
# training_data2 = [
#   ("Tokyo Tower is 333m tall.", [(0, 11, "BUILDING")]),
# ]
# the DocBin will store the example documents
db = DocBin()
for text, annotations in training_data:
    doc = nlp(text)
    ents = []
    for start, end, label in annotations:
        span = doc.char_span(start, end, label=label)
        ents.append(span)
    doc.ents = ents
    db.add(doc)
db.to_disk("./train.spacy")



