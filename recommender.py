"""
Contains various recommondation implementations
all algorithms return a list of movieids
"""

import pandas as pd
import numpy as np
import pickle

from utils import movies, movies_list, Ratings, nmf_model, fill_variables

def recommend_random(k=10):
    return movies['title'].sample(k).to_list()

def recommend_with_NMF(query, model=nmf_model, k=10):
    """
    NMF Recommender
    INPUT
    - user_vector with shape (1, #number of movies)
    - user_item_matrix
    - trained NMF model

    OUTPUT
    - a list of movieIds
    """
    recommendations = []

    new_user_dataframe = pd.DataFrame(query, columns=movies_list, index=["new_user"])
    new_user_dataframe_imputed = new_user_dataframe.fillna(Ratings.mean())

    P_new_user_matrix = nmf_model.transform(new_user_dataframe_imputed)

    R_hat_new_user_matrix = np.dot(P_new_user_matrix, nmf_model.components_)
    R_hat_new_user = pd.DataFrame(data=R_hat_new_user_matrix,
                                  columns=nmf_model.feature_names_in_,
                                  index = ['new_user'])
    
    sorted_list = R_hat_new_user.transpose().sort_values(by='new_user',ascending=False).index.to_list()

    rated_movies = list(query.keys())
    
    recommendations = [movie for movie in sorted_list if movie not in rated_movies]
    
    return recommendations[0:k]


def recommend_neighborhood(query, k=10):

    initial, unseen, top_five_user, user_item, user_user_matrix, new_user = fill_variables(query)

    recommended = list()
    ratio_list = list()

    for movie in unseen:

        other_users = initial.columns[~initial.loc[movie].isna()]
        other_users = set(other_users)

        num = 0
        den = 0
        ratio = 0

        for other_user in other_users.intersection(set(top_five_user)):

            rating = user_item[other_user][movie]
            sim = user_user_matrix[new_user][other_user]

            num += (rating*sim)
            den += sim + 0.0001

            ratio = num/den 

        recommended.append(movie)
        ratio_list.append(ratio)

    rec_df = pd.DataFrame(recommended)
    rat_df = pd.DataFrame(ratio_list)

    total_df = pd.concat([rec_df, rat_df], axis=1)
    total_df.columns = ['title','rating']

    recommended_df = total_df.sort_values('rating',ascending=False).head(k)

    recommendations = recommended_df.title.to_list()

    return recommendations
    