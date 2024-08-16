import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
import numpy as np


class StructuralRelation:
    '''
    Class calculating the structural relationship (structural similarity) between tweets using jaccard similarity
    the similarity is compared between the named entities and the hashtags in the tweets
    '''
    def __init__(self):
        '''
        Initializes the StructuralRelation class by loading the spaCy model for named entity recognition.
        '''
        self.nlp = en_core_web_sm.load()
        
        
    def named_entities(self,text:str):
        '''
        Extracts named entities from the given text using spaCy.
        (text : str )-> list A list of named entities found in the text.
        '''
        text_nlp = self.nlp(text)
        entities_text = [ent.text for ent in text_nlp.ents]
        return entities_text
    
    def overlap_measure(self,tj:list,ti:list):
        '''
        Computes the Jaccard overlap between two lists of elements (named entities or hashtags).

        (ti:list,tj:list) -> float The Jaccard similarity coefficient, representing the overlap between the two lists.
        where tn is A list of elements (named entities or hashtags) from the n tweet.
        '''
        intersection = list(set((tj)).intersection(set((ti))))
        union = set(tj + ti)
        if len(union) == 0:
            return 0
        return len(intersection)/len(union)      

    def struct_relation_calculation(self,tj:str,ti:str,hashtag_ti:list,hashtag_tj:list,alpha:int,beta:int,hashtag_flag:bool):
        '''
        Computes the structural similarity between two tweets using the Jaccard difference.
        The similarity is based on the overlap of named entities and optionally hashtags.


        (tj:str,ti:str,hashtag_ti : list, hashtag_tj : list, alpha : int,beta : int,hashtag_flag : bool) 
        ->  float The structural similarity score between the two tweets.

        where
        tj : 
            Named entities from the first tweet.
        ti : 
            Named entities from the second tweet.
        hashtag_ti : 
            Hashtags from the first tweet.
        hashtag_tj : 
            Hashtags from the second tweet.
        alpha : int
            Weight for hashtag similarity in the final score.
        beta : int
            Weight for named entity similarity in the final score.
        hashtag_flag : bool
            Flag indicating whether to consider hashtag overlap in the calculation.


        '''
        hashtag_flag_int = int( hashtag_flag == True)
        hashtags_overlap = 0 
        if hashtag_flag :
            hashtags_overlap = self.overlap_measure(hashtag_tj,hashtag_ti)
        named_entities_overlap = self.overlap_measure(tj,ti)
        return (hashtag_flag_int*alpha*hashtags_overlap) + (beta*named_entities_overlap)

    def get_struct_relation_matrix(self,tweets:list):
        '''
        Calculates the structural relationship matrix for a list of tweets.

        tweets : list of dict -> np.ndarray A matrix where each element (i, j) represents the structural similarity
                                            between tweets i and j.
        '''
        tweet_texts = [tweet['normalized'] for tweet in tweets]
        n = len(tweet_texts)
        relation_matrix = [[0 for _ in range(n)] for _ in range(n)]
        
        ner = []
        for text in tweet_texts:
            ner.append(self.named_entities(text))
        
        for i in range(n):
            for j in range(i,n):
                if i == j :
                    relation_matrix[i][j] = 1
                else :
                    value = self.struct_relation_calculation(ner[i],ner[j],None,None,0.0,1.0,False)
                    relation_matrix[i][j] = value
                    relation_matrix[j][i] = value
                # print(j)
        return np.array(relation_matrix)
