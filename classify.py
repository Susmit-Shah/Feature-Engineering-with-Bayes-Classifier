#!/usr/bin/env python
from collections import defaultdict
from csv import DictReader, DictWriter

import nltk
import codecs
import sys
from nltk.corpus import wordnet as wn
from nltk.tokenize import TreebankWordTokenizer
import string
import re

kTOKENIZER = TreebankWordTokenizer()

def morphy_stem(word):
    """
    Simple stemmer
    """
    stem = wn.morphy(word)
    if stem:
        return stem.lower()
    else:
        return word.lower()

class FeatureExtractor:
    def __init__(self):
        """
        You may want to add code here
        """

        None

    def features(self, text):
        d = defaultdict(int)
        # d will contain term frequency for each word in document

        # for ii in kTOKENIZER.tokenize(text):
        #     d[morphy_stem(ii)] += 1

        #text = text.strip()

        # ################ Check if text ends with punctuation ##############################

        # if text.endswith(string.punctuation):
        #     d["end_punctuation"] = True
        # else:
        #     d["end_punctuation"] = False

        words = text.split(" ")
        # words = kTOKENIZER.tokenize(text)

        # Feature :: Last character of text
        d["ending_"+text[-1]+""] += 1

        # Feature :: Number of words in line
        d["line_length"] = len(words)

        # Feature :: Number of characters in text
        d["num_char"] = len("".join(words))

        # Feature :: Average length/size of words used in text
        d["average_word_len"] = d["num_char"]/len(words)

        # Feature :: Number of vowels present in text
        d["vowel_count"] = len(re.sub('[^aeiou]', "", "".join(words)))

        # Feature :: Number of digits used in text
        d["digit_count"] = len(re.findall(r'\d', text))

        # Feature :: Number of punctuation in text.
        # d["punc_count"] = len(re.sub('[^\'!/\"#$%&\\\()*+,\-./:;<=>?@\^_`{|}~]', "", "".join(words)))
        # d["punc_count"] = len(re.sub('[\w]', "", "".join(words)))


        # ########################### N- GRAMS #############################################

        ## 2 - grams ##
        # t = re.sub('[^\w ]', "", text)
        # words_1 = t.split(" ")
        # words_1 = [x for x in words_1 if x]
        # for i in range(0, len(words_1) - 1):
        #     d[words_1[i] + " " + words_1[i + 1]] += 1
        #     #l.append((words_1[i] + " " + words_1[i + 1] + " " + words_1[i + 2]))

        # ## 3 - grams ##
        t = text                     #re.sub('[^\w ]', "", text)
        words_1 = t.split(" ")
        words_1 = [x for x in words_1 if x]
        for i in range(0, len(words_1) - 2):
            d[words_1[i] + " " + words_1[i + 1] + " " + words_1[i+2]] += 1
            # l.append((words_1[i] + " " + words_1[i + 1] + " " + words_1[i+2]))
        # print words
        # print l

        # ## 4 - grams ##
        # for i in range(0, len(words_1) - 3):
        #     d[words_1[i] + " " + words_1[i + 1] + " " + words_1[i + 2] + " " + words_1[i + 3]] += 1



        p = list(string.punctuation)

        for ii in words:

            ii = ii.strip()

            #i = ii.strip(string.punctuation+" ")

            # Feature :: Check if entire word is in Upper case
            # if i.isupper(): # and len(i)>1:
            #     # print ii, " :: ", cat
            #    d["all_upper"] += 1

            ii = re.sub('[^\w]', '', ii)            # Strips all characters except for a-zA-Z from ii

            # Feature :: Term Frequency
            if ii != '':
                d[morphy_stem(ii.lower())] += 1

        text = text.strip(string.punctuation)

        # Feature :: Check for number of comma separated text in poem line
        # comma_split = text.split(',')
        # d['comma_split'] = len(comma_split)

        w = text.split(" ")
        w = [x for x in w if x]  # Removes " " in list if present
        # Remove everything except word character
        end = re.sub('[^\w]', '', w[-1])
        start = re.sub('[^\w]', '', w[0])

        # Feature :: Ending word of text
        d["ends_with_word+" + end.lower()] += 1

        # Feature :: Starting word of text
        d["starts_with_word+" + start.lower()] += 1

        return d


reader = codecs.getreader('utf8')
writer = codecs.getwriter('utf8')


def prepfile(fh, code):
  if type(fh) is str:
    fh = open(fh, code)
  ret = gzip.open(fh.name, code if code.endswith("t") else code+"t") if fh.name.endswith(".gz") else fh
  if sys.version_info[0] == 2:
    if code.startswith('r'):
      ret = reader(fh)
    elif code.startswith('w'):
      ret = writer(fh)
    else:
      sys.stderr.write("I didn't understand code "+code+"\n")
      sys.exit(1)
  return ret

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--trainfile", "-i", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="input train file")
    parser.add_argument("--testfile", "-t", nargs='?', type=argparse.FileType('r'), default=None, help="input test file")
    parser.add_argument("--outfile", "-o", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="output file")
    parser.add_argument('--subsample', type=float, default=1.0,
                        help='subsample this fraction of total')
    args = parser.parse_args()
    trainfile = prepfile(args.trainfile, 'r')
    if args.testfile is not None:
        testfile = prepfile(args.testfile, 'r')
    else:
        testfile = None
    outfile = prepfile(args.outfile, 'w')

    # Create feature extractor (you may want to modify this)
    fe = FeatureExtractor()

    # Read in training data
    train = DictReader(trainfile, delimiter='\t')

    # Split off dev section
    dev_train = []
    dev_test = []
    full_train = []

    #fh13 = open("lines.txt","w")

    for ii in train:
        if args.subsample < 1.0 and int(ii['id']) % 100 > 100 * args.subsample:
            continue

        #fh13.write(ii['text'] + '###' + ii['cat'] +'\n')
        feat = fe.features(ii['text'])
        if int(ii['id']) % 5 == 0:
            dev_test.append((feat, ii['cat']))
        else:
            dev_train.append((feat, ii['cat']))
        full_train.append((feat, ii['cat']))


    # Train a classifier
    sys.stderr.write("Training classifier ...\n")
    classifier = nltk.classify.NaiveBayesClassifier.train(dev_train)

    errors = []
    right = 0
    total = len(dev_test)
    for ii in dev_test:
        prediction = classifier.classify(ii[0])
        if prediction == ii[1]:
            right += 1
        # else:
        #     errors.append((ii[0], ii[1], prediction))
        # for i in errors:
        #     print i
        #     print("\n")
    sys.stderr.write("Accuracy on dev: %f\n" % (float(right) / float(total)))

    if testfile is None:
        sys.stderr.write("No test file passed; stopping.\n")
    else:
        # Retrain on all data
        classifier = nltk.classify.NaiveBayesClassifier.train(dev_train + dev_test)

        # Read in test section
        test = {}
        for ii in DictReader(testfile, delimiter='\t'):
            test[ii['id']] = classifier.classify(fe.features(ii['text']))

        # Write predictions
        o = DictWriter(outfile, ['id', 'pred'])
        o.writeheader()
        for ii in sorted(test):
            o.writerow({'id': ii, 'pred': test[ii]})

    print classifier.show_most_informative_features(75)
