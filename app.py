import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
from streamlit import session_state as session

from recommender import recommend_random, recommend_with_NMF, recommend_neighborhood
from utils import movies, Ratings, rating_list

nav = st.sidebar.radio(
    "Please chose one of the following options:",
    ["See Movies","Random", "NMF", "Cosine Similarity"]
    ) 

if nav == "See Movies":
    st.markdown(
    """ # Here's a list of all the movies!
    """
    )

    st.write(movies)

    st.markdown(
    """ # Here's a sample of the ratings table used in the creation of these models.
    """
    )

    st.markdown(
    """ ##### If you inspect closely, you will see several users with the same rating for \
    some movies. This is because the NMF requires all values to be filled. We chose to fill \
    These values with the average rating per movie.
    """
    )

    st.write(Ratings.head(30))


if nav == "Random":
    st.markdown(
    """ ## Welcome to the Random Movie Recommender.
    """
    )
    st.image("https://media.giphy.com/media/Qm5fbbxOEY9E5bULOK/giphy.gif", width=400)
    
    st.markdown(
    """ ### This model randomly selects movies from our movie list for you to view.
    """
    )

    st.markdown(
    """ ##### Tonight you should watch...
    """
    )
        
    movie_list = recommend_random()

    for i in movie_list:
        st.markdown(i)


if nav == "NMF":
    st.markdown(
    """ # **Welcome to Non-negative Matrix Factorization Recommender.**
    """
    )
   
    st.markdown(
    """ ### *How does this work?*
    """
    )

    st.image("https://media.giphy.com/media/W0QduXZQEcNEa8r0oY/giphy.gif", width=400)

    st.markdown(
    """This model works by taking in data which has a list of movies, users, and \
    the ratings that each user gave each movie. Using these ratings and *MATH*, a number \
    of genres are created (in this case we used 500). With *MATH* each movie is graded \
    as to how well it fits into each genre. Then each user is graded as to how much they \
    like each genre. Doing this gives us certain values for each movie and each user. \
    """
    )

    st.markdown(
    """After we find these values, we can add a new user with some information about which \
    movie they like/dislike and use our new *MATH* to assign to them some movies that the \
    general population also likes.
    """
    )

    st.markdown(
    """ ### With images it works like this:
    """
    )

    st.image("https://media.geeksforgeeks.org/wp-content/uploads/20210429230138/Example1-660x195.png")

    st.markdown(
    """ ### *Pros and Cons*
    """
    )

    st.markdown(
    """ 
    * \+ This is a great method for determining what the most popular movies are.
    * \+ Works great when the dimensions are super large.
    * \- In our case, the model tends to recommend only those movies which many people \
    have reviewed.
    * \- This leads to everyone being recommended a handful of movies regardless of their \
    ratings.
    """
    )

    dataframe = None

    st.text("")
    st.text("")
    st.text("")
    st.text("")

    session.options1 = st.multiselect(label="Select Movies", options=movies)

    st.text("")
    st.text("")

    session.options2 = st.multiselect(label="Rating", options=rating_list)

    st.text("")
    st.text("")

    if len(session.options1) != len(session.options2):
        st.write("Double check your entries. They should be the same length!")

    new_query = dict(zip(session.options1, session.options2))

    buffer1, col1, buffer2 = st.columns([1.45, 1, 1])

    is_clicked = col1.button(label="Recommend")

    if is_clicked:
        dataframe = recommend_with_NMF(new_query)

    st.text("")
    st.text("")
    st.text("")
    st.text("")

    if dataframe is not None:
        st.table(dataframe)


if nav == "Cosine Similarity":

    st.markdown(
    """ # **Welcome to Cosine Similarity Recommender.**
    """
    )
   
    st.markdown(
    """ ### *How does this work?*
    """
    )

    st.image("https://media.giphy.com/media/4JVTF9zR9BicshFAb7/giphy.gif", width=400)

    st.markdown(
    """Neighborhood Based Collaborative Filtering leverages the behavior of other users to \
    know what our user might enjoy. It may find people similar to our user and recommend movies \
    they liked or recommend movies that other people saw after rating what our user has seen. \
    In effect, it looks at what you rated, and compares you to other users who rated those \
    same movies in a similar manner.
    """
    )

    st.markdown(
    """Even if two users may look different at first, the algorithm proves that they are in\
    fact related!
    """
    )

    st.image("https://media.giphy.com/media/xYTExvnaF4KW1eaYZY/giphy.gif")

    st.markdown(
    """ ### *Pros and Cons*
    """
    )

    st.markdown(
    """ 
    * \+ We don't need in depth knowledge of the culture because the embeddings are \
        automatically learned.
    * \+ The model can help users discover new interests.
    * \- If an item is not seen during training, the system can't create an embedding for \
        it and can't query the model with this item.
    * \- Side features are difficult to include. For movie recommendations, the side features \
        might include country or age.
    """
    )

    dataframe = None

    st.text("")
    st.text("")
    st.text("")
    st.text("")

    session.options1 = st.multiselect(label="Select Movies", options=movies)

    st.text("")
    st.text("")

    session.options2 = st.multiselect(label="Rating", options=rating_list)

    st.text("")
    st.text("")

    if len(session.options1) != len(session.options2):
        st.write("Double check your entries. They should be the same length!")

    new_query = dict(zip(session.options1, session.options2))

    buffer1, col1, buffer2 = st.columns([1.45, 1, 1])

    is_clicked = col1.button(label="Recommend")

    if is_clicked:
        dataframe = recommend_neighborhood(new_query)

    st.text("")
    st.text("")
    st.text("")
    st.text("")

    if dataframe is not None:
        st.table(dataframe)