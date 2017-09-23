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

    def features(self, text, cat=""):
        d = defaultdict(int)
        # d will contain term frequency for each word in document

        text = text.strip()

        # for ii in kTOKENIZER.tokenize(text):
        #     d[morphy_stem(ii)] += 1


        # if text.endswith(string.punctuation):
        #     d["end_punctuation"] = True
        # else:
        #     d["end_punctuation"] = False


        words = text.split(" ")

        d["line_length"] = len(words)
        d["num_char"] = len("".join(words))
        d["average_word_len"] = d["num_char"]/len(words)
        #d["starts_with+"+words[0].lower()] += 1
        #print "".join(words)
        d["vowel_count"] = len(re.sub('[^aeiou]', "", "".join(words)))

        l = []

        ## 2- grams ##
        # words = [x for x in words if x]
        # for i in range(0, len(words) - 1):
        #     d[words[i] + " " + words[i + 1]] += 1
        #     l.append((words[i] + " " + words[i + 1]))
        # print words
        # print l


        p = list(string.punctuation)

        for ii in words:

            ii = ii.strip()
            # for pp in p :
            #     if pp in ii:
            #         print ii

            # i = ii.strip(string.punctuation)
            # if i.isupper(): # and len(i)>1:
            #     # print ii, " :: ", cat
            #     d["all_upper"] += 1

            # if ii[0] in p:
            #     d[ii[0]] += 1
            #     #print ii
            # if ii[-1] in p:
            #     d[ii[-1]] += 1

            ii = ii.strip(string.punctuation)

            d[morphy_stem(ii.lower())] += 1

            #d[ii.lower()] += 1

        text = text.strip(string.punctuation)
        w = text.split(" ")
        w = [x for x in w if x]  # Removes " " in list if present
        d["ends_with+" + w[-1].lower()] += 1
        d["starts_with+" + w[0].lower()] += 1

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

    for ii in train:
        if args.subsample < 1.0 and int(ii['id']) % 100 > 100 * args.subsample:
            continue

        feat = fe.features(ii['text'], ii['cat'])
        if int(ii['id']) % 5 == 0:
            dev_test.append((feat, ii['cat']))
        else:
            dev_train.append((feat, ii['cat']))
        full_train.append((feat, ii['cat']))


    # Train a classifier
    sys.stderr.write("Training classifier ...\n")
    classifier = nltk.classify.NaiveBayesClassifier.train(dev_train)

    right = 0
    total = len(dev_test)
    for ii in dev_test:
        prediction = classifier.classify(ii[0])
        if prediction == ii[1]:
            right += 1
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
