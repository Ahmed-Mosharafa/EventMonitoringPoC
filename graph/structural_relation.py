import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
import numpy as np


class StructuralRelation:
    def __init__(self):
        self.nlp = en_core_web_sm.load()
        
        
    def named_entities(self,text:str):
        text_nlp = self.nlp(text)
        # print(text_nlp)
        # names_entities = text_nlp.ents
        # print([(X.text, X.label_) for X in doc.ents])
        # print(text_nlp.ents)
        entities_text = [ent.text for ent in text_nlp.ents]
        # NE_list = list(text_nlp.ents.text)
        return entities_text
    
    def overlap_measure(self,tj:list,ti:list):
        intersection = list(set((tj)).intersection(set((ti))))
        # intersection = list(set(tj) & set(ti))
        union = set(tj + ti)
        if len(union) == 0:
            return 0 
        # print(tj)
        # print((ti))
        # print(len(intersection))
        # print(len(union))
        # print(len(intersection)/len(union) )
        return len(intersection)/len(union)      

    def struct_relation_calculation(self,tj:str,ti:str,hashtag_ti:list,hashtag_tj:list,alpha:int,beta:int,hashtag_flag:bool):
    #hashtag flag should 0 if false 1 if true
        hashtag_flag_int = int( hashtag_flag == True)
        hashtags_overlap = 0 
        if hashtag_flag :
            hashtags_overlap = self.overlap_measure(hashtag_tj,hashtag_ti)
        named_entities_overlap = self.overlap_measure(self.named_entities(tj),self.named_entities(ti))
        return (hashtag_flag_int*alpha*hashtags_overlap) + (beta*named_entities_overlap)

    def get_struct_relation_matrix(self,tweets):
        # tweet_texts = [tweet['normalized'] for tweet in tweets]
        # embeddings = self.get_contextual_embeddings(tweet_texts)
        # Calculate the embedding similarities
        tweet_texts = [tweet['normalized'] for tweet in tweets]
        n = len(tweet_texts)
        relation_matrix = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(i,n):
                if i == j :
                    relation_matrix[i][j] = 1
                else :
                    value = self.struct_relation_calculation(tweet_texts[i],tweet_texts[j],None,None,0.0,1.0,False)
                    relation_matrix[i][j] = value
                    relation_matrix[j][i] = value
                # print(j)
        return np.array(relation_matrix)
    
# struc = StructuralRelation()
# tweets = ["Alice walked through Central Park and enjoyed the fresh air","lice met John in Central Park and felt the cool breeze"
#           ,"John and Alice strolled by the River Thames and admired the serene view"]
# x = struc.get_struct_relation_matrix(tweets)
# # x = struc.overlap_measure(["x","haha"],["y","x"])
# print(x)


