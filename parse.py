# SI 561 Natural Language Processing
# Assignment 2: Multilingual Dependency Parsing
# Author: Fengmin Hu

import sys
import nltk
from providedcode import dataset
from providedcode.transitionparser import TransitionParser
from providedcode.evaluate import DependencyEvaluator
from featureextractor import FeatureExtractor
from transition import Transition
from providedcode.dependencygraph import DependencyGraph

tp = TransitionParser.load(sys.argv[1])

for sent in sys.stdin:
    sentence = DependencyGraph.from_sentence(sent)
    for key, dct in sentence.nodes.items():
	dct['ctag'] = nltk.tag.mapping.map_tag("en-ptb", "universal", dct['ctag'])
    parsed = tp.parse([sentence])
    print parsed[0].to_conll(10).encode('utf-8')

    


