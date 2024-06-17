import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm


class StructuralRelation:
    
    nlp = en_core_web_sm.load()
    def named_entities(text:str):
        text_nlp = nlp(text)
        # names_entities = text_nlp.ents
        # print([(X.text, X.label_) for X in doc.ents])
        return text_nlp.ents
    
    def overlap_measure(tj:list,ti:list):
    intersection = list(set(tj) & set(ti))
    union = tj + ti
    return len(intersection)/len(union)

    def struct_relation_calculation(tj:text,ti:text,hashtag_ti:list,hashtag_tj:list,alpha:int,beta:int,hashtag_flag:bool):
    #hashtag flag should 0 if false 1 if true
    hashtag_flag_int = int(x == True)
    if hashtag_flag :
        hashtags_overlap = overlap_measure(hashtag_tj,hashtag_ti)
    named_entities_overlap = overlap_measure(named_entities(tj),named_entities(ti))
    return (hashtag_flag_int*alpha*hashtags_overlap) + (beta*named_entities_overlap)