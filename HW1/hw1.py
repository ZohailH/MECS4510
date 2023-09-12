import sys
from collections import defaultdict
import math
import random
import os
import os.path
from matplotlib import pyplot as plt
import numpy as np
from itertools import chain
 
"""
MECS 4510 - Evolutionary Programming - Fall 2023 
Programming Homework 1 - Traveling Salesperson Problem

HW # 1 - Wilson Luo
"""



def read_file(filename: str):
    """
    Read the contents of .tsp or .tour file into a list
    """
    with open(filename) as f:
        content = [line.strip().split() for line in f.read().splitlines()]
        return content

def get_nodes(content: list):
    """
    Run this after running read_file on a .tsp file. 
    This function converts just the data portion of the .tsp file into a dictionary
    with the following format:
    
    {key| arg}: {'1'| ('103', '104')}
    """
    node_list = content[6:-1]
    node_dict = {}
    for city in node_list: 
        node_dict[city[0]] = (city[1], city[2])
    return node_dict
    
def get_gold_tour(content: list):
    """
    Run this after running read_file on a .tour file.
    This function converts just the data portion of the .tour file into a list 
    of the optimal tour for the given dataset with the following format:
        
    [103, 3, 24, 35, 22...]
    """
    gold_list = content[4:-1]
    gold_tour_list = list(chain.from_iterable(gold_list))
    return gold_tour_list



class TSPModel(object):
    """
    Class that lets you optimize a given dataset 
    """
    
    def __init__(self, node_dict):

        self.nodes = node_dict
        
        def __init__(self, corpusfile):
        
            # Iterate through the corpus once to build a lexicon 
            generator = corpus_reader(corpusfile)
            self.lexicon = get_lexicon(generator)
            self.lexicon.add("UNK")
            self.lexicon.add("START")
            self.lexicon.add("STOP")
            
            # Count Total Word Tokens
            #generator = corpus_reader(corpusfile, self.lexicon)
            self.total_words = 0
    
            # Now iterate through the corpus again and count ngrams
            generator = corpus_reader(corpusfile, self.lexicon)
            self.count_ngrams(generator)
            
            # Find Perplexity of Base Model 
            generator = corpus_reader(corpusfile, self.lexicon)
            self.model_perplexity = self.perplexity(generator)
            
    
        def count_tokens_in_sentence(self, sentence):
            unigram_token = get_ngrams(sentence, 1)
            num_tokens = len(unigram_token) - 1
            return num_tokens
            
        
        def count_ngrams(self, corpus):
            """
            COMPLETE THIS METHOD (PART 2)
            Given a corpus iterator, populate dictionaries of unigram, bigram,
            and trigram counts. 
            """
       
            self.unigramcounts = {} # might want to use defaultdict or Counter instead
            self.bigramcounts = {} 
            self.trigramcounts = {} 
            
            # Iterate Through Each Sentence
            for sentence in corpus: 
                
                self.total_words += self.count_tokens_in_sentence(sentence)
                
                # Record Unigrams 
                unigram_token = get_ngrams(sentence, 1)
                for var in unigram_token:
                    if var in self.unigramcounts:
                        self.unigramcounts[var] += 1
                    else:
                        self.unigramcounts[var] = 1
                        
                
                # Record Bigrams
                bigram_token = get_ngrams(sentence, 2)
                for var in bigram_token:
                    if var in self.bigramcounts:
                        self.bigramcounts[var] += 1
                    else:
                        self.bigramcounts[var] = 1
                
                # Record Trigrams
                trigram_token = get_ngrams(sentence, 3)
                for var in trigram_token:
                    if var in self.trigramcounts:
                        self.trigramcounts[var] += 1
                    else:
                        self.trigramcounts[var] = 1
                
    
            return
        self.current_tour = []
        self.best_tour = []
        
        # List of Distances from Each Tour (len = no. of tours)
        self.tour_performance = []
        
        self.curr_tour_distance = 0
        self.prev_tour_distance = 1000000000000000000
        
        """ This variable controls how many runs to do"""
        self.tour_threshold = 1000000
    
    
    def get_distance_between_city(self, key1, key2):
        """"
        Given: 2 int representing 2 cities
        Return: distance between 2 cities with key1 and key2
        """
        coord1 = self.nodes[key1]
        coord2 = self.nodes[key2]
        dist = math.sqrt((int(coord1[0]) - int(coord2[0])) ** 2 + 
                         (int(coord1[1]) - int(coord2[1])) ** 2)
        return dist
    
    def get_tour_distance(self, tour : list):
        """
        Given: list of cities 
        Return: total distance between each city (including from last and first city)
        """
        total_distance = 0
        for index, city in enumerate(tour):
            current_dist = self.get_distance_between_city(city, tour[index-1])
            total_distance += current_dist
        return total_distance
        
    
    def generate_rand_tour(self):
        """
        Generate a list of cities with a random order
        """
        self.current_tour = list(self.nodes.keys()).copy()
        random.shuffle(self.current_tour)
        
    def swap_two_rand_city(self):
        """
        Given: the current list of cities
        Return: the same list of cities with 2 cities randomly swapped
        """
        cities = random.choices(self.current_tour, k=2)
        city1 = self.current_tour.index(cities[0])
        city2 = self.current_tour.index(cities[1])
        swap_tour = self.current_tour.copy()
        swap_tour[city1], swap_tour[city2] = self.current_tour[city2], self.current_tour[city1]
        return swap_tour  
        
    def random_search(self):
        """
        Generate an initial random city order. Randomly select 2 cities to swap and compare 
        if total distance is greater or less than the previous. If less, use the new swapped order.
        If more, ignore swap and try a different swap. Keep going until threshold is reached.
        """
        self.generate_rand_tour()
        
        for i in range(self.tour_threshold):
            self.curr_tour_distance = self.get_tour_distance(self.current_tour)
            
            swap_tour = self.swap_two_rand_city()
            swap_tour_dist = self.get_tour_distance(swap_tour)
            
            if swap_tour_dist < self.curr_tour_distance:
                self.current_tour = swap_tour
                self.tour_performance.append(swap_tour_dist)
            else:
                self.tour_performance.append(self.curr_tour_distance)
            
            
        
        self.best_tour = self.current_tour
            
            

        


if __name__ == "__main__":
    
    data_file = "a280.tsp"
    tour_file = "a280.opt.tour"
    
    data_content = read_file(data_file)
    node_dict = get_nodes(data_content)
    
    gold_content = read_file(tour_file)
    gold_list = get_gold_tour(gold_content)

    model = TSPModel(node_dict)
    model.random_search()
    print(model.tour_threshold, ' Total Runs')
    print('Best Tour: ', model.best_tour)
    print('Best Tour has a distance of ', model.tour_performance[-1])
    #print('Distance over each Run: ', model.tour_performance)
    print('Gold Tour', gold_list)
    print('Gold Tour has a distance of ', model.get_tour_distance(gold_list))
    
    plt.plot(np.arange(model.tour_threshold), model.tour_performance)
    
