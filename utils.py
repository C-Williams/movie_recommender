"""
UTILS 
- Helper functions to use for your recommender funcions, etc
- Data: import files/models here e.g.
    - movies: list of movie titles and assigned cluster
    - ratings
    - user_item_matrix
    - item-item matrix 
- Models:
    - nmf_model: trained sklearn NMF model
"""
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics.pairwise import cosine_similarity

rating_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

movies = pd.read_csv('./data/movies.csv', index_col=0) 
movies_list = movies.title.to_list()

with open('./data/nmf_model.pkl','rb') as file:
    nmf_model = pickle.load(file)

Ratings = pd.read_csv('./data/Ratings.csv', index_col=0)

cosine_sim = pd.read_csv('./data/cosine_sim.csv', index_col=0)

def movie_to_id(string_titles):
    '''
    converts movie title to id for use in algorithms'''
    
    movieID = movies.set_index('title').loc[string_titles]['movieid']
    movieID = movieID.tolist()
    
    return movieID


def id_to_movie(movieID):
    '''
    converts movie Id to title
    '''
    rec_title = movies.set_index('movieid').loc[movieID]['title']
    
    return rec_title


def fill_variables(query):
    """
    Establishes several variables for use in recommending with neighbors.
    """   
    initial = cosine_sim.assign(new_user=query)

    user_item = initial.fillna(0)

    user_user_matrix = cosine_similarity(user_item.T)
    user_user_matrix = pd.DataFrame(user_user_matrix, columns = user_item.columns, index = user_item.columns).round(2)

    unseen = initial[initial.new_user.isna()].index

    top_five_user = user_user_matrix.new_user.sort_values(ascending=False).index[1:6]

    new_user = 'new_user'

    return initial, unseen, top_five_user, user_item, user_user_matrix, new_user
