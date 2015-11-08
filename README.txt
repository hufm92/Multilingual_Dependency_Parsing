EECS 595: Natural Language Processing
Assignment 2: Multilingual Dependency Parsing
Author: Fengmin Hu (hufm)

Description:
In this assignment, we will be implementing a Dependency Parsing algorithm. 

---------------------------------------------------------------
Part 1: Dependency graphs
---------------------------------------------------------------
a. Generate a visualized dependency graph of a sentence from each of the English, Danish, Korean and Swedish training data sets.
The visualized dependency graphs are named figure_en/da/ko/sw and stored in the root of Homework2.

b. How to determine if a dependency graph is projective?
There is a function called _is_projective() in providecode/transitionparser.py to determine if a dependency graph is projective or not.
Given a dependency graph, we first create a arc_list to store the dependency relationships between nodes. So that we go through each node in the dependency graph first. For each node, if the 'head' attribute available, which means that it has a parent node, we add the tuple (parentIdx, childIdx) into the arc_list.
After that, we will go through the arc_list. For each arc, we first make sure that childIdx < parentIdx, then we want to make sure whether there is a arc, which connect a node between childIdx and parentIdx with a node outside the range of (childIdx, parentIdx). If this kind of arc exists, it will intersect with the current arc, which means that this dependency graph is not projective. 
If all arcs are not intersect with other arcs, we will return True to indicate that the dependency graph is projective. Otherwise, the function will return False.

c. Examples of projective/non-projective sentences.
Projective:	I will go to a bookstore with my parents.
Non-Projective:	I will go to a bookstore with my parents which opened last week.


---------------------------------------------------------------
Part 2: Manipulating configurations (Assignment Part 1)
---------------------------------------------------------------
a. Implement the transition.py to manipulate the parser configuration.
We could implement the transition-based dependency parsing by Nivre's paper. 

b. Examine the performance of our parser using the provided badfeatures.model
The performance of our parser using the provided badfeature.model is as follows,

    UAS: 0.229038040231
    LAS: 0.125473013344

According to the definition, LAS is the percentage of tokens that our parser has predicted the correct head and dependency relation label while UAS is the percentage of tokens that our parser has predicted the correct head without label. In this way, we found the performance is relatively lower than what we expected. 
After checking with current featureextractor.py, we found current features include: STK_0_FORM, STK_0_FEATS, STK_0_L/RDEP, BUF_0_FORM, BUF_0_FEATS, BUF_0_L/RDEP. Although those features are sufficient to build a simple parser, the number of features is still too small for the training task. A more complex parser including the LEMMA, TAG and CHILDREN information will definitely achieve a better performance than the simple badfeature.model.

---------------------------------------------------------------
Part 3: Dependency Parsing (Assignment Part 2)
---------------------------------------------------------------
a. Edit featureextractor.py and try to improve the performance of our feature extractor.
The basic idea of our feature engineering is from the Table 3.2 of the book Dependency Parsing by Kuebler, McDonald and Nivre. 
The original feature extractor already includes STK_0_FORM, STK_0_FEATS, STK_0_L/RDEP, BUF_0_FORM, BUF_0_FEATS, BUF_0_L/RDEP. According to the Table 3.2, we found following those features could improve our performance. The detailed description and complexity of new features are listed below:
    1) LEMMA(lemma or base form) of STK[0]/BUF[0]. Each token has an attribute of 'lemma', so that it will take O(1) time complexity to extract this feature for each token if available.
    2) POSTAG(ctag and tag) of STK[0/1]/BUF[0/1/2/3]. Each token has attributes of 'ctag' and 'tag', so that it will take us O(1) time complexity to extract this POSTAG feature for each token if available. Noticed that, we could use nltk.tag.mapping.map_tag function to convert some incorrect coarse-grained tags. (Really appreciate the students' answer @174 at Piazza)
    3) FORM of BUF[1]. If BUF[1] is available, we also want to add this token directly into features. This also take O(1) time complexity to extarct each BUF[1].
    4) Distance. Besides the features in the Table 3.2, we also investigated the distance feature. It is popular to use the distance between two words, typically  the word on top of the stack and the first word in the input buffer. Since the distance could be computed directly by the indexes of two words. We will take O(1) avarage time complexity to extract this feature.
    5) number of left/right children of STK[0]/BUF[0]. This is also a common type of feature. To implement this part of count, I refer to the left_children and right_children methods and wrote them in similar way. Since everytime we got a index of current token, we will go through whole children lists to compare and count. So that we will have O(len(children)) time complexity to add this feature. In genearl, the time complexity to count all numbers of left/right children is O(len(arc_list)). According to Nivre's ACL Slides, we know that the number of left/right-arc transitions is at most 2n, which is O(n). So the average time complexity for this feature extraction is still O(1).

The detailed implementation is in featureextractor.py and the performance of this feature extractor on english, danish and swedish is listed and discussedd as follows. In order to investigate the performances of features, we would like to compare the performance of the badfeature.model to the performance of badfeature.model plus each of feature (i.e, baseline + feature A, baseline +  feature B, baseline + feature C). In this step, we choose three features include distance of words, number of left/right children and POSTAGs to study. The detailed comparison is as follows,
    1) Baseline 
			  UAS		     LAS
	english	    0.276543209877	0.22962962963 
	danish	    0.701596806387	0.616766467066
	swedish	    0.229038040231	0.125473013344

    2) Baseline + Feature A (Distance)
                          UAS                LAS
        english	    0.279012345679	0.237037037037
        danish	    0.719760479042	0.636726546906
        swedish     0.255725951006      0.143995220076

    3) Baseline + Feature B (Number of Left/Right Children)
                          UAS                LAS
        english	    0.743209876543	0.565432098765
        danish	    0.825349301397	0.716367265469	
        swedish     0.667596096395      0.419438358893

    4) Baseline + Feature C (POSTAG of STK[0] and BUF[0])
                          UAS                LAS
        english	    0.550617283951	0.523456790123
        danish	    0.76626746507	0.694211576846
        swedish     0.722764389564	0.624178450508

According to the performance listed above, we found some interesting findings.
1) First of all, we found that although the simple feature model (badfeature.model) works relatively poor for english and swedish, it provides a satisfied baseline for danish. 
2) Secondly, we found that the feature of word distance between the top of stack and the first word in buffer is not a very powerful features. Since the performance is slightly better than the baseline, not so much.
3) Thirdly, the number of left/right children is an important feature which improve the performance significantly. 
4) The POSTAGs of STK[0] and BUF[0] also provide useful information for training task. In our final feature extractor, we also include more POSTAGs of other tokens, which will improve the overall performance too.


b. Generate the models for English, Danish and Swedish datasets, and save the trained model for later evaluation.
The trained models (english.model, danish.model and swedish.model) stored in the root of Homework2. 
The training datasets are randomly generated and the descriptions are as follows,
    For english.model: 
	Number of training examples : 200
	Number of valid (projective) examples : 200
    For danish.model:
	Number of training examples : 200
	Number of valid (projective) examples : 174
    For swedish.model:
	Number of training examples : 200
	Number of valid (projective) examples : 180

c. Score your models against the test datasets.
In this step, we used the danish_test, swedish_test and english_dev as our test datasets. The overall performance of our parser is as follows,
    For english.model:
	UAS: 0.864197530864
	LAS: 0.82962962963
    For danish.model:
	UAS: 0.87624750499
	LAS: 0.782634730539
    For swedish.model:
	UAS: 0.85082652858
	LAS: 0.727544313882
From the results, we found the performance is dramatically improved compared with the original badfeature.model and all models achieve 70% or higher for both UAS and LAS with 200 training sentences.

d. Discuss the complexity of the arc-eager shift-reduce parser and what tradeoffs it makes.
The time complexity of Nivre's parsing method is, by itself, O(n), and n is the size of inputs (length of sentence). (Conclusion from Nivre's ACL Slides) Since most features we extracted are attributes of each token, it take O(1) to get those features except for the number of left/right children. As for the feature of number of left/right children, we actually will go through every arc for constant times in total. So the average time complexity will be the O(len(arcs)/n). According to Nivre's conclusion, the number of arcs is less than the number of left/right-arc transitions, which is also O(n). So that the overall time complexity of the system with our final featureextractor will still be O(n).

Arc-eager makes tradeoffs between ambiguity and memory. 
Arc-eager try to use early composition to reduce memory for the push-down automata. However, this eager attachments are made with less bottom-up evidence. Compared with arc standard method, which attachments are made when constituents are complete, arc-eager is more efficient but less safe.


---------------------------------------------------------------
Part 4: Parser Executable
---------------------------------------------------------------
Create a file parse.py 
The parse.py is located in the root of Homework2. The standard output is a valid CoNLL-format file and can be viewed by the MaltEval evaluator.


