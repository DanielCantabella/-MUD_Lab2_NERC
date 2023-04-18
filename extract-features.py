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
    # example: tokens = [('La', 0, 1), ('meva', 3, 6), ('mare', 8, 11), ('es', 13, 14), ('la', 16, 17), ('millor', 19, 24), ('pero', 26, 29), ('vull', 31, 34), ('un', 36, 37), ('entrepa', 39, 45), ('de', 47, 48), ('Hydrogen', 50, 57)]
    # for each token, generate list of features and add it to the result
    drugSuffixes = ["-ine", "-ol", "-azole", "-vir", "-mab", "-tidine", "-pril", "-sartan", "-statin", "-barbital",
                    "-itide", "-afloxacin", "-zepam", "-oxetine", "-prazole", "-tidone", "-ridone", "-gliflozin",
                    "-tinib",
                    "-vastatin"]
    drugPrefixes = ["iso-", "cis-", "trans-", "para-", "meta-", "anti-", "pro-", "pre-", "post-", "co-",
                    "en-", "de-", "re-", "un-", "dis-", "over-", "under-", "sub-", "super-", "inter-"]
    brandSuffixes = ["-max", "-dex", "-pro", "-gen", "-con", "-x", "-in", "-vir", "-lia", "-zo", "-on", "-ron", "-an",
                     "-ix", "-ol", "-tan", "-vel", "-cal", "-viran", "-to"]
    brandPrefixes = ["neo-", "nova-", "pro-", "lux-", "med-", "ortho-", "omni-", "reli-", "sym-", "ultra-", "vit-",
                     "xeno-", "zeno-", "endo-", "novo-", "bio-", "glo-", "inno-", "sola-", "terra-"]
    groupSuffixes = ["-oids", "-amines", "-azepines", "-drals", "-oxanes", "-prils", "-olols", "-dipines", "-astatins",
                     "-statins", "-barbitals"]
    groupPrefixes = ["anti-", "pro-", "re-", "co-", "pre-", "post-", "in-", "im-", "sub-", "super-", "trans-", "ex-",
                     "multi-",
                     "pseudo-", "de-", "inter-", "intra-", "non-", "para-", "peri-"]
    drug_nSuffixes = ["-ides", "-anes", "-actams", "-ximabs", "-ruxicobs", "-ciclibs", "-lizumabs", "-biximabs",
                      "-zumabs", "-olimabs", "-umimabs", "-radinibs", "-asibs", "-axons", "-golixes", "-zixes",
                      "-ocogibs", "-adibats",
                      "-tabines", "-cirans"]
    drug_nPrefixes = ["neo-", "pre-", "post-", "in-", "im-", "sub-", "super-", "trans-", "ex-", "multi-", "pseudo-",
                      "de-", "inter-", "intra-", "non-", "para-", "peri-", "co-", "pro-", "re-"]

    drug_pattern = re.compile(
        r'\b(?:\w+[,-])*([A-Z][a-z]{2,}(?:\s[A-Z][a-z]+)*(?:\s\w+)?)(?:,[A-Z][a-z]{2,}(?:\s[A-Z][a-z]+)*(?:\s\w+)*)*\b')

    drug_names = set()
    brand_names = set()
    group_names = set()

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

    result = []
    for k in range(0, len(tokens)):  # len(tokens) is the number of words
        tokenFeatures = []
        t = tokens[k][0]

        tokenFeatures.append("form=" + t)  # Useful
        tokenFeatures.append("suf3=" + t[-3:])  # Useful
        tokenFeatures.append("suf2=" + t[-2:])  # Useful
        tokenFeatures.append("suf1=" + t[-1:])  # useful
        # tokenFeatures.append("prefix1=" + t[:1]) #unuseful, worsens
        tokenFeatures.append("prefix2=" + t[:2])  # Useful
        tokenFeatures.append("prefix3=" + t[:3])  # Useful
        # tokenFeatures.append("prefix4=" + t[:4]) #Unuseful, does not affect

        tokenFeatures.append("pos=" + nltk.pos_tag([t])[0][1])  # part of speech of the word USEFUL
        # tokenFeatures.append("capitalized=" + str(t.isupper())) #Unuseful, it worsens by 0.3%
        tokenFeatures.append("length=" + str(len(t)))  # Useful, it helps around 1%
        # tokenFeatures.append("numeric=" + str(t.isnumeric())) # Unuseful, any difference
        # tokenFeatures.append("lowercase=" + str(t.isalpha() and t.islower())) # Unuseful, any difference
        tokenFeatures.append("uppercase=" + str(t.isalpha() and t.isupper()))  # USEFUL
        tokenFeatures.append("startwithdash=" + str(t[0] == '-'))  # Useful, but just a bit from 76,3 to 76,1
        # tokenFeatures.append("startwithpunctuation=" + str(t[0] in string.punctuation)) # Unuseful, not many difference
        tokenFeatures.append("nucleotidesequence=" + str(t in ['a', 'c', 'g', 't']))  # USEFUL
        # tokenFeatures.append("quote=" + str(t in ['“', '”', "'", '"']))  Unuseful, not many difference

        tokenFeatures.append("isDrugName=" + str(int(t.lower() in drug_names)))  # Useful
        tokenFeatures.append("isBrandName=" + str(int(t.lower() in brand_names)))  # Useful
        tokenFeatures.append("isGroupName=" + str(int(t.lower() in group_names)))  # Useful

        drug_match = drug_pattern.match(t)
        tokenFeatures.append("drugPattern=" + str(int(bool(drug_match))))  # Useful

        # tokenFeatures.append("drug_suffix=" + str(any(t.endswith(suffix) for suffix in drugSuffixes)))
        # tokenFeatures.append("drug_prefix=" + str(any(t.startswith(prefix) for prefix in drugPrefixes)))
        # tokenFeatures.append("brand_suffix=" + str(any(t.endswith(suffix) for suffix in brandSuffixes)))
        # tokenFeatures.append("brand_prefix=" + str(any(t.startswith(prefix) for prefix in brandPrefixes)))
        # tokenFeatures.append("group_suffix=" + str(any(t.endswith(suffix) for suffix in groupSuffixes)))
        # tokenFeatures.append("group_prefix=" + str(any(t.startswith(prefix) for prefix in groupPrefixes)))
        # tokenFeatures.append("drug_n_suffix=" + str(any(t.endswith(suffix) for suffix in drug_nSuffixes)))
        # tokenFeatures.append("drug_n_prefix=" + str(any(t.startswith(prefix) for prefix in drug_nPrefixes)))

        if k > 0:
            tPrev = tokens[k - 1][0]
            # tokenFeatures.append("formPrev=" + tPrev) #Unuseful
            tokenFeatures.append("suf3Prev=" + tPrev[-3:])  # USEUFUL, VERY USEFUL
            # tokenFeatures.append("suf2Prev=" + tPrev[-2:]) #Unuseful
            tokenFeatures.append("suf1Prev=" + tPrev[-1:])  # Useful
            tokenFeatures.append("prevBigram=" + tPrev + "_" + t)  # Useful, very useful
            tokenFeatures.append("posPrev=" + nltk.pos_tag([tPrev])[0][1])  # Useful
            # tokenFeatures.append("pref3Prev=" + tPrev[:3]) #Unuseful
            # tokenFeatures.append("pref2Prev=" + tPrev[:2]) #Unuseful
            # tokenFeatures.append("pref2Prev=" + tPrev[:1]) #Unuseful
            # tokenFeatures.append("lengthPrev=" + str(len(tPrev)))

        else:
            tokenFeatures.append("BoS")

        if k < len(tokens) - 1:
            tNext = tokens[k + 1][0]
            tokenFeatures.append("formNext=" + tNext)  # Useful
            tokenFeatures.append("suf3Next=" + tNext[-3:])  # Useful
            tokenFeatures.append("suf2Next=" + tNext[-2:])  # Useful
            # tokenFeatures.append("suf1Next=" + tNext[-1:]) #Unuseful, dos enot affect
            tokenFeatures.append("nextBigram=" + t + "_" + tNext)  # next word bigram #USeful, very useful
            # tokenFeatures.append("posNext=" + nltk.pos_tag([tNext])[0][1]) #Unuseful, it imporves 0.1 if we dont use it
            # tokenFeatures.append("pref1Next=" + tNext[:1])  #Unuseful
            # tokenFeatures.append("pref2Next=" + tNext[:2]) #Unuseful, no aporta
            # tokenFeatures.append("pref3Next=" + tNext[:3]) #Unuseful

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
