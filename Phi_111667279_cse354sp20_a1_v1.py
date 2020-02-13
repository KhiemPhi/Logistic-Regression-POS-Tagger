
# CSE354 Sp20; Assignment 1 Template v1
##################################################################

import sys

##################################################################
#1. Tokenizer

import re #python's regular expression package

def tokenize(sent):
    #input: a single sentence as a string.
    #output: a list of each "word" in the text
    # must use regular expressions  
    
    # Retain capitalization.
    # Separate punctuation from words, except for (a) abbreviations of capital letters (e.g. “U.S.A.”), (b) hyphenated words (e.g. “data-driven”) and contractions (e.g. “can’t”).  
    # Allow for hashtags and @mentions as single words (e.g. “#sarcastic”, “@sbunlp”) 
    
    #<FILL IN>

    pattern = r"[\#\@]?(?:[A-Z]\.)+|[\#\@]?\w+[\'\-]\w+|[#\@]?[A-z]+|[#\@\$]?\d+[\.\%]?\d*|[#\@]?[^\s]+"
    # [\#\@]?(?:[A-Z]\.?){2,} ---> Abbreviations
    # [\#\@]?\w+[\'\-]\w+ --->  Contractions and hyphens
    # [#\@]?[A-z]+  ---> Any word
    # [#\@]?\$?\d+[\.\%]?\d* ---> Percentages, Dollar Currency, Numbers, Decimal
    # [#\@]?[^\s]+ ---> Any combination of non-space characters 1 or more times
    
    
    tokens = re.findall(pattern, sent)


    return tokens


##################################################################
#2. Pig Latinizer
def first_vowel(token):
    #input : token, a singular token or string
    #output: the index of the first_vowel

    i = re.search("[aeiou]", token, re.IGNORECASE)
    return -1 if i == None else i.start()

def pigLatinizer(tokens):
    #input: tokens: a list of tokens,
    #output: plTokens: tokens after transforming to pig latin
   
    plTokens = []
     # if the word starts with a consonant, have all consonants up to the first vowel appended to the end of the word and add “ay”.  E.g. (“stony brook” => “onystay ookbray”).
    # if the word starts with a vowel, append it with “way”. (e.g. “our university” => “ourway universityway”.
    # if the word has characters that are not vowels and consonants, should remain the same (hashtags, punctuations   )
    vowels = ('a','e','i','o','u','A','E','I','O','U')
    for token in tokens:
        if token.isalpha():      
            if (token.startswith(vowels)):
                # start with vowel
                token_pig_latinize = token + "way"
                plTokens.append(token_pig_latinize)
            else:
                token_pig_latinize = ""
                first_vowel_index = first_vowel(token)
                if (first_vowel_index != -1):
                    characters_until_first_vowel = token[0:first_vowel_index]
                    characters_after_first_vowel = token[first_vowel_index:]
                    token_pig_latinize = characters_after_first_vowel + characters_until_first_vowel + "ay"
                else:
                    token_pig_latinize = token + "ay"
                plTokens.append(token_pig_latinize)
        else:
            plTokens.append(token)
                


    #<FILL IN>
        
    return plTokens
    

##################################################################
#3. Feature Extractor

import numpy as np

def getFeaturesForTokens(tokens, wordToIndex):
    #input: tokens: a list of tokens,
    #wordToIndex: dict mapping 'word' to an index in the feature list.
    #output: list of lists (or np.array) of k feature values for the given target

    # Number of vowels in the target word
    # Number of consonants in the target word
    # One-hot representations of (all case-insensitive): 
                # Previous word
                # Target word
                # Next word
    

    num_words = len(tokens)
    
    featuresPerTarget = list() #holds arrays of feature per word
    for targetI in range(num_words):
        #<FILL IN>     
        word = tokens[targetI]
        # Counting Consonants and Vowels
        c_count = 0
        v_count = 0
        vowels = ('a','e','i','o','u','A','E','I','O','U')
        for i in word:
           if i in vowels:
               v_count = v_count + 1
           elif ( (i>='a' and i <= 'z') or  (i>='A' and i <= 'Z') ):
               c_count = c_count + 1
        letters_count = [c_count, v_count]  

        #One hot representations
        word_before = tokens[targetI-1] 
        word_after = tokens[targetI+1]
        word_before_encode = [0] * len(wordToIndex)
        word_encode = [0] * len(wordToIndex)
        word_after_encode = [0] * len(wordToIndex)

        # Setting Words That Exist To 1        
        word_before_encode[wordToIndex[word_before]] = 1
        word_after_encode[wordToIndex[word_after]] = 1
        word_encode[wordToIndex[word]] = 1

        featuresPerTarget[targetI]  = word_encode + word_after_encode + word_before_encode + letters_count

    return featuresPerTarget #a (num_words x k) matrix


##################################################################
#4. Adjective Classifier

from sklearn.linear_model import LogisticRegression

def trainAdjectiveClassifier(features, adjs):
    #inputs: features: feature vectors (i.e. X)
    #        adjs: whether adjective or not: [0, 1] (i.e. y)
    #output: model -- a trained sklearn.linear_model.LogisticRegression object

    model = None
    #<FILL IN>

    return model


##################################################################
##################################################################
## Main and provided complete methods
## Do not edit.
## If necessary, write your own main, but then make sure to replace
## and test with this before you submit.
##
## Note: Tests below will be a subset of those used to test your
##       code for grading.

#!/usr/bin/python3

#!/usr/bin/python3
# CSE354 Sp20; Assignment 1 Template v02
##################################################################
_version_ = 0.2

import sys

##################################################################
#1. Tokenizer

import re #python's regular expression package

def tokenize(sent):
     # input: a single sentence as a string.
    # output: a list of each "word" in the text
    # must use regular expressions

    # Retain capitalization.
    # Separate punctuation from words, except for (a) abbreviations of capital letters (e.g. “U.S.A.”), (b) hyphenated words (e.g. “data-driven”) and contractions (e.g. “can’t”).
    # Allow for hashtags and @mentions as single words (e.g. “#sarcastic”, “@sbunlp”)

    # <FILL IN>

    pattern = r"[\#\@]?(?:[A-Z]\.)+|[\#\@]?\w+[\'\-]\w+|[#\@]?[A-z]+|[#\@\$]?\d+[\.\%]?\d*|[#\@]?[^\s]+"
    # [\#\@]?(?:[A-Z]\.?){2,} ---> Abbreviations
    # [\#\@]?\w+[\'\-]\w+ --->  Contractions and hyphens
    # [#\@]?[A-z]+  ---> Any word
    # [#\@]?\$?\d+[\.\%]?\d* ---> Percentages, Dollar Currency, Numbers, Decimal
    # [#\@]?[^\s]+ ---> Any combination of non-space characters 1 or more times

    tokens = re.findall(pattern, sent)

    return tokens

##################################################################
#2. Pig Latinizer

def first_vowel(token):
    # input : token, a singular token or string
    # output: the index of the first_vowel

    i = re.search("[aeiou]", token, re.IGNORECASE)
    return -1 if i == None else i.start()


def pigLatinizer(tokens):
    # input: tokens: a list of tokens,
    # output: plTokens: tokens after transforming to pig latin

    plTokens = []
    # if the word starts with a consonant, have all consonants up to the first vowel appended to the end of the word and add “ay”.  E.g. (“stony brook” => “onystay ookbray”).
    # if the word starts with a vowel, append it with “way”. (e.g. “our university” => “ourway universityway”.
    # if the word has characters that are not vowels and consonants, should remain the same (hashtags, punctuations   )
    vowels = ('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U')
    for token in tokens:
        if token.isalpha():
            if (token.startswith(vowels)):
                # start with vowel
                token_pig_latinize = token + "way"
                plTokens.append(token_pig_latinize)
            else:
                token_pig_latinize = ""
                first_vowel_index = first_vowel(token)
                if (first_vowel_index != -1):
                    characters_until_first_vowel = token[0:first_vowel_index]
                    characters_after_first_vowel = token[first_vowel_index:]
                    token_pig_latinize = characters_after_first_vowel + characters_until_first_vowel + "ay"
                else:
                    token_pig_latinize = token + "ay"
                plTokens.append(token_pig_latinize)
        else:
            plTokens.append(token)

    # <FILL IN>

    return plTokens
    

##################################################################
#3. Feature Extractor

import numpy as np

def zero_signals (length):
    return [0] * length

def signal_flipper (index, zero_signals):
    zero_signals[index] = 1
    return zero_signals

def getFeaturesForTokens(tokens, wordToIndex):
    #input: tokens: a list of tokens,
    #wordToIndex: dict mapping 'word' to an index in the feature list.
    #output: list of lists (or np.array) of k feature values for the given target

    num_words = len(tokens)
    featuresPerTarget = list() #holds arrays of feature per word
    
    for targetI in range(num_words):                     
        # Setting up 3 Encoders  
        length = len(wordToIndex) + 1 # Adding 1 to be OOD Detector 
        word_after_encoder = zero_signals(length)
        word_before_encoder = zero_signals(length)
        word_current_encoder = zero_signals(length)
        
        # Encoding The Current Word  
        word = tokens[targetI].lower()     
        index_current = wordToIndex[word]
        signal_flipper(index_current, word_current_encoder)
        
        # Encoding The Next Word, only if next word exists
        if (targetI + 1 < num_words):
            word_after = tokens[targetI + 1].lower()
            index_after = wordToIndex[word_after]
            signal_flipper(index_after, word_after_encoder)

        # Encoding the Word Before, only if before word exists, i.e token is at index > 0
        if (targetI > 0):
            word_before = tokens[targetI - 1].lower()
            index_before = wordToIndex[word_before]
            signal_flipper(index_before, word_before_encoder)       

        # Counting Consonants and Vowels
        c_count = 0
        v_count = 0 
        vowels = ('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U')
        for i in word:
            if i in vowels:
                v_count = v_count + 1
            elif ((i >= 'a' and i <= 'z') or (i >= 'A' and i <= 'Z')):
                c_count = c_count + 1
        letters_count = [c_count, v_count]
        
        featuresPerTarget.append(word_current_encoder + word_before_encoder + word_after_encoder + letters_count) 

    return featuresPerTarget #a (num_words x k) matrix


##################################################################
#4. Adjective Classifier

from sklearn.linear_model import LogisticRegression

def trainAdjectiveClassifier(features, adjs):
    #inputs: features: feature vectors (i.e. X)
    #        adjs: whether adjective or not: [0, 1] (i.e. y)
    #output: model -- a trained sklearn.linear_model.LogisticRegression object
    
    #<FILL IN>

    #Splitting our feature (X_train) and adjs (y_train) into development set 
    X_trainsub, X_dev, y_trainsub, y_dev = train_test_split(features, adjs, 
                                                  test_size=0.10, random_state=42)
    
    #Defining Penalties And Accuracy And Model:
    init_C = 10000
    final_acc = 0
    continue_test_C = True   
    model = LogisticRegression(C=init_C, penalty="l1", solver='liblinear')
    model.fit(X_trainsub, y_trainsub)
    
    while (continue_test_C):
        model_test = LogisticRegression(C=init_C, penalty="l1", solver='liblinear')
        model_test.fit(X_trainsub, y_trainsub)
        y_pred = model.predict(X_dev)
        acc = (1 - np.sum(np.abs(y_pred - y_dev))/len(y_dev) )
        if (acc > final_acc):
            final_acc = acc
            model = model_test
        else:
            continue_test_C = False        
        init_C = init_C * 10
 
    return model


##################################################################
##################################################################
## Main and provided complete methods
## Do not edit.
## If necessary, write your own main, but then make sure to replace
## and test with this before you submit.
##
## Note: Tests below will be a subset of those used to test your
##       code for grading.

def getConllTags(filename):
    #input: filename for a conll style parts of speech tagged file
    #output: a list of list of tuples
    #        representing [[[word1, tag1], [word2, tag2]]]
    wordTagsPerSent = [[]]
    sentNum = 0
    with open(filename, encoding='utf8') as f:
        for wordtag in f:
            wordtag=wordtag.strip()
            if wordtag:#still reading current sentence
                (word, tag) = wordtag.split("\t")
                wordTagsPerSent[sentNum].append((word,tag))
            else:#new sentence
                wordTagsPerSent.append([])
                sentNum+=1
    return wordTagsPerSent

# Main
if __name__== '__main__':
    print("Initiating test. Version " , _version_)
    #Data for 1 and 2
    testSents = ['I am attending NLP class 2 days a week at S.B.U. this Spring.',
                 "I don't think data-driven computational linguistics is very tough.",
                 '@mybuddy and the drill begins again. #SemStart']

    #1. Test Tokenizer:
    print("\n[ Tokenizer Test ]\n")
    tokenizedSents = []
    for s in testSents:
        tokenizedS = tokenize(s)
        print(s, tokenizedS, "\n")
        tokenizedSents.append(tokenizedS)

    #2. Test Pig Latinizer:
    print("\n[ Pig Latin Test ]\n")
    for ts in tokenizedSents:
        print(ts, pigLatinizer(ts), "\n")
        
    #load data for 3 and 4 the adjective classifier data:
    taggedSents = getConllTags('daily547.conll')

    #3. Test Feature Extraction:
    print("\n[ Feature Extraction Test ]\n")
    #first make word to index mapping: 
    wordToIndex = set() #maps words to an index
    for sent in taggedSents:
        if sent:
            words, tags = zip(*sent) #splits [(w, t), (w, t)] into [w, w], [t, t]
            wordToIndex |= set([w.lower() for w in words]) #union of the words into the set
    print("  [Read ", len(taggedSents), " Sentences]")
    #turn set into dictionary: word: index
    wordToIndex = {w: i for i, w in enumerate(wordToIndex)}

    #Next, call Feature extraction per sentence
    sentXs = []
    sentYs = []
    print("  [Extracting Features]")
    for sent in taggedSents:
        if sent:
            words, tags = zip(*sent)
            sentXs.append(getFeaturesForTokens(words, wordToIndex)) 
            sentYs.append([1 if t == 'A' else 0 for t in tags])
    #test sentences
    print("\n", taggedSents[5], "\n", sentXs[5], "\n")
    print(taggedSents[192], "\n", sentXs[192], "\n")


    #4. Test Classifier Model Building
    print("\n[ Classifier Test ]\n")
    #setup train/test:
    from sklearn.model_selection import train_test_split
    #flatten by word rather than sent: 
    X = [j for i in sentXs for j in i]
    y= [j for i in sentYs for j in i]
    try: 
        X_train, X_test, y_train, y_test = train_test_split(np.array(X),
                                                            np.array(y),
                                                            test_size=0.20,
                                                            random_state=42)
    except ValueError:
        print("\nLooks like you haven't implemented feature extraction yet.")
        print("[Ending test early]")
        sys.exit(1)
    print("  [Broke into training/test. X_train is ", X_train.shape, "]")
    #Train the model.
    print("  [Training the model]")
    tagger = trainAdjectiveClassifier(X_train, y_train)
    print("  [Done]")
    

    #Test the tagger.
    from sklearn.metrics import classification_report
    #get predictions:
    y_pred = tagger.predict(X_test)
    #compute accuracy:
    leny = len(y_test)
    print("test n: ", leny)
    acc = np.sum([1 if (y_pred[i] == y_test[i]) else 0 for i in range(leny)]) / leny
    print("Accuracy: %.4f" % acc)
    #print(classification_report(y_test, y_pred, ['not_adj', 'adjective']))
    

                                                                                                                                    
