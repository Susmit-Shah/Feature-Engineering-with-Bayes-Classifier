#!/usr/bin/env python
from collections import defaultdict
from csv import DictReader, DictWriter

import nltk
import codecs
import sys
from nltk.corpus import wordnet as wn
from nltk.tokenize import TreebankWordTokenizer

import math
import string

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
    def get_doc_freq(self, training_data):
        doc_freq_dict = defaultdict(int)
        for eachLine in training_data:
            eachLineString = eachLine['text']
            #print "Each Line :: ", eachLineString
            tokenized_line = kTOKENIZER.tokenize(eachLineString)
            #print tokenized_line
            unique_words_in_line = set(tokenized_line)
            for eachWord in unique_words_in_line:
                if eachWord not in string.punctuation:
                    doc_freq_dict[morphy_stem(eachWord)] += 1
        #print "Main Dictionary :: ", doc_freq_dict
        #print("\n\n\n")
        return doc_freq_dict

    # def features(self, text):
    #     d = defaultdict(int)
    #     for ii in kTOKENIZER.tokenize(text):
    #         d[morphy_stem(ii)] += 1
    #     return d

    def features(self, text, document_frequency):
        features = defaultdict(int)
        term_frequency_dict = defaultdict(int)
        token_line = kTOKENIZER.tokenize(text)
        #for ii in kTOKENIZER.tokenize(text.lower()):
        for ii in token_line:
            if ii not in string.punctuation:
                term_frequency_dict[morphy_stem(ii)] += 1
            #print term_frequency_dict

        for each_word in term_frequency_dict:
            #print "Each Word :: ", each_word
            if document_frequency[each_word] > 0:
                features[each_word] = term_frequency_dict[each_word] * (
                math.log10(len(document_frequency) / document_frequency[each_word]))
                # idf = math.log10(len(document_frequency)/document_frequency[each_word])
                # if (idf > 1.6) and (idf < 3):
                #     features[each_word] = term_frequency_dict[each_word] * (math.log10(len(document_frequency)/document_frequency[each_word]))
                # else:
                #     features[each_word] = 0
            else:
                features[each_word] = term_frequency_dict[each_word] * (
                math.log10(len(document_frequency) / 1))
                # idf = math.log10(len(document_frequency)/1)
                # if (idf > 1.6) and (idf < 3):
                #     features[each_word] = term_frequency_dict[each_word] * (
                #     math.log10(len(document_frequency) / 1))
                # else:
                #     features[each_word] = 0

        return features



reader = codecs.getreader('utf8')
writer = codecs.getwriter('utf8')


def prepfile(fh, code):
    if type(fh) is str:
        fh = open(fh, code)
    ret = gzip.open(fh.name, code if code.endswith("t") else code + "t") if fh.name.endswith(".gz") else fh
    if sys.version_info[0] == 2:
        if code.startswith('r'):
            ret = reader(fh)
        elif code.startswith('w'):
            ret = writer(fh)
        else:
            sys.stderr.write("I didn't understand code " + code + "\n")
            sys.exit(1)
    return ret


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--trainfile", "-i", nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                        help="input train file")
    parser.add_argument("--testfile", "-t", nargs='?', type=argparse.FileType('r'), default=None,
                        help="input test file")
    parser.add_argument("--outfile", "-o", nargs='?', type=argparse.FileType('w'), default=sys.stdout,
                        help="output file")
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
    d_train = []
    d_test = []
    f_train = []
    dev_train = []
    dev_test = []
    full_train = []

    for each_document in train:
        if args.subsample < 1.0 and int(each_document['id']) % 100 > 100 * args.subsample:
            continue
        if int(each_document['id']) % 5 == 0:
            d_test.append(each_document)
        else:
            d_train.append(each_document)
        f_train.append(each_document)

    document_frequency = fe.get_doc_freq(d_train)
    #print sorted(((v, k) for k,v in document_frequency.iteritems()), reverse=True)
    d_f = dict()
    for i in document_frequency:
        d_f[i] = float(len(d_train)/document_frequency[i])
    s = sorted(((v, k) for k, v in d_f.iteritems()), reverse=True)
    for l in s:
        print l
    #print "Document Frequency :: ", document_frequency
    print "\n\n"

    for ii in d_train:
        feat = fe.features(ii['text'], document_frequency)
        #print "Train Features :: ", feat
        #print "\n"
        dev_train.append((feat, ii['cat']))

    for ii in d_test:
        feat = fe.features(ii['text'], document_frequency)
        #print "Test Features :: ", feat
        #print "\n"
        dev_test.append((feat, ii['cat']))

    for ii in f_train:
        feat = fe.features(ii['text'], document_frequency)
        #print "Full Train Features :: ", feat
        #print "\n"
        full_train.append((feat, ii['cat']))


    #print dev_train
    #print d_test


    # for ii in train:
    #     if args.subsample < 1.0 and int(ii['id']) % 100 > 100 * args.subsample:
    #         continue
    #
    #     feat = fe.features(ii['text'])
    #     # print "Features :: ", feat
    #     if int(ii['id']) % 5 == 0:
    #         dev_test.append((feat, ii['cat']))
    #     else:
    #         dev_train.append((feat, ii['cat']))
    #     full_train.append((feat, ii['cat']))

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

    print classifier.show_most_informative_features(50)

