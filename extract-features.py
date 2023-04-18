#! /usr/bin/python3

import sys
import re
from os import listdir

from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize
import string
import nltk


## --------- tokenize sentence -----------
## -- Tokenize sentence, returning tokens and span offsets

def tokenize(txt):
    offset = 0
    tks = []
    ## word_tokenize splits words, taking into account punctuations, numbers, etc.
    for t in word_tokenize(txt):
        ## keep track of the position where each token should appear, and
        ## store that information with the token
        offset = txt.find(t, offset)
        tks.append((t, offset, offset + len(t) - 1))
        offset += len(t)

    ## tks is a list of triples (word,start,end)
    return tks


## --------- get tag -----------
##  Find out whether given token is marked as part of an entity in the XML

def get_tag(token, spans):
    (form, start, end) = token
    for (spanS, spanE, spanT) in spans:
        if start == spanS and end <= spanE:
            return "B-" + spanT
        elif start >= spanS and end <= spanE:
            return "I-" + spanT

    return "O"


## --------- Feature extractor -----------
## -- Extract features for each token in given sentence

def extract_features(tokens):
    drug_names = set()
    brand_names = set()
    group_names = set()

    with open('../DDI/resources/HSDB.txt', 'r') as f:
        for line in f:
            drug_names.add(line.strip().lower())

    with open('../DDI/resources/DrugBank.txt', 'r') as f:
        for line in f:
            if line.split("|")[-1] == "drug":
                drug_names.add(line.strip().lower())
            elif line.split("|")[-1] == "brand":
                brand_names.add(line.strip().lower())
            elif line.split("|")[-1] == "group":
                group_names.add(line.strip().lower())
    with open('../DDI/resources/HSDB.txt', 'r') as f:
        for line in f:
            drug_names.add(line.strip().lower())

    with open('./drug.txt', 'r') as f:
        for line in f:
            drug_names.add(line.strip().lower())
    with open('./brand.txt', 'r') as f:
        for line in f:
            brand_names.add(line.strip().lower())
    with open('./group.txt', 'r') as f:
        for line in f:
            group_names.add(line.strip().lower())
    drug_pattern = re.compile(
        r'\b(?:\w+[,-])*([A-Z][a-z]{2,}(?:\s[A-Z][a-z]+)*(?:\s\w+)?)(?:,[A-Z][a-z]{2,}(?:\s[A-Z][a-z]+)*(?:\s\w+)*)*\b')
    result = []
    for k in range(0, len(tokens)):  # len(tokens) is the number of words
        tokenFeatures = []
        t = tokens[k][0]
        tokenFeatures.append("form=" + t)
        tokenFeatures.append("suf3=" + t[-3:])
        tokenFeatures.append("suf2=" + t[-2:])
        tokenFeatures.append("suf1=" + t[-1:])
        tokenFeatures.append("prefix2=" + t[:2])
        tokenFeatures.append("prefix3=" + t[:3])
        tokenFeatures.append("length=" + str(len(t)))
        tokenFeatures.append("uppercase=" + str(t.isalpha() and t.isupper()))
        tokenFeatures.append("startwithdash=" + str(t[0] == '-'))
        tokenFeatures.append("nucleotidesequence=" + str(t in ['a', 'c', 'g', 't']))
        tokenFeatures.append("pos=" + nltk.pos_tag([t])[0][1])
        tokenFeatures.append("isDrugName=" + str(int(t.lower() in drug_names)))
        tokenFeatures.append("isBrandName=" + str(int(t.lower() in brand_names)))
        tokenFeatures.append("isGroupName=" + str(int(t.lower() in group_names)))
        drug_match = drug_pattern.match(t)
        tokenFeatures.append("drugPattern=" + str(int(bool(drug_match))))

        # tokenFeatures.append("prefix1=" + t[:1])
        # tokenFeatures.append("prefix4=" + t[:4])
        # tokenFeatures.append("capitalized=" + str(t.isupper()))
        # tokenFeatures.append("numeric=" + str(t.isnumeric()))
        # tokenFeatures.append("lowercase=" + str(t.isalpha() and t.islower()))
        # tokenFeatures.append("startwithpunctuation=" + str(t[0] in string.punctuation))
        # tokenFeatures.append("quote=" + str(t in ['“', '”', "'", '"']))
        if k > 0:
            tPrev = tokens[k - 1][0]
            tokenFeatures.append("suf3Prev=" + tPrev[-3:])
            tokenFeatures.append("suf1Prev=" + tPrev[-1:])
            tokenFeatures.append("prevBigram=" + tPrev + "_" + t)
            tokenFeatures.append("posPrev=" + nltk.pos_tag([tPrev])[0][1])
            # tokenFeatures.append("formPrev=" + tPrev)
            # tokenFeatures.append("suf2Prev=" + tPrev[-2:])
            # tokenFeatures.append("pref3Prev=" + tPrev[:3])
            # tokenFeatures.append("pref2Prev=" + tPrev[:2])
            # tokenFeatures.append("pref2Prev=" + tPrev[:1])
            # tokenFeatures.append("lengthPrev=" + str(len(tPrev)))
        else:
            tokenFeatures.append("BoS")
        if k < len(tokens) - 1:
            tNext = tokens[k + 1][0]
            tokenFeatures.append("formNext=" + tNext)
            tokenFeatures.append("suf3Next=" + tNext[-3:])
            tokenFeatures.append("suf2Next=" + tNext[-2:])
            tokenFeatures.append("nextBigram=" + t + "_" + tNext)
            # tokenFeatures.append("suf1Next=" + tNext[-1:])
            # tokenFeatures.append("posNext=" + nltk.pos_tag([tNext])[0][1])
            # tokenFeatures.append("pref1Next=" + tNext[:1])
            # tokenFeatures.append("pref2Next=" + tNext[:2])
            # tokenFeatures.append("pref3Next=" + tNext[:3])
            # tokenFeatures.append("lengthNext=" + str(len(tNext)))
        result.append(tokenFeatures)
    return result


## --------- MAIN PROGRAM -----------
## --
## -- Usage:  baseline-NER.py target-dir
## --
## -- Extracts Drug NE from all XML files in target-dir, and writes
## -- them in the output format requested by the evaluation programs.
## --


# directory with files to process
datadir = sys.argv[1]

# process each file in directory
for f in listdir(datadir):

    # parse XML file, obtaining a DOM tree
    tree = parse(datadir + "/" + f)

    # process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences:
        sid = s.attributes["id"].value  # get sentence id
        spans = []
        stext = s.attributes["text"].value  # get sentence text
        entities = s.getElementsByTagName("entity")
        for e in entities:
            # for discontinuous entities, we only get the first span
            # (will not work, but there are few of them)
            (start, end) = e.attributes["charOffset"].value.split(";")[0].split("-")
            typ = e.attributes["type"].value
            spans.append((int(start), int(end), typ))

        # convert the sentence to a list of tokens
        tokens = tokenize(stext)
        # extract sentence features
        features = extract_features(tokens)

        # print features in format expected by crfsuite trainer
        for i in range(0, len(tokens)):
            # see if the token is part of an entity
            tag = get_tag(tokens[i], spans)
            print(sid, tokens[i][0], tokens[i][1], tokens[i][2], tag, "\t".join(features[i]), sep='\t')

        # blank line to separate sentences
        print()





