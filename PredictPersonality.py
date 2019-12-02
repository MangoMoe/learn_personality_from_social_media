from gen_data import *
import numpy as np
from itertools import permutations 
from sklearn import tree    
from sklearn.neighbors import KNeighborsClassifier


def discritize_data(train_people_likes, test_people_likes, permut_n, M):
    #format person data 
    people_train = []
    for likes in train_people_likes:
        person = np.zeros(M)
        person[likes] = 1
        people_train.append(person)
    
    people_test = []
    for likes in test_people_likes:
        person = np.zeros(M)
        person[likes] = 1
        people_test.append(person)

    #discritize range of personalities for classification 
    classes = np.array(
                list(permutations(np.linspace(-1, 1, permut_n), 5))
                )
    return people_train, people_test, classes


def method_1(train_people_scores, train_people_likes, test_people_likes, M, n_train, n_test):
    """Assume every page has a personality score and guess it based off data. Guess a persons personality
    score to be the average of all the pges they liked. 
    """
    #train model
    ############
    person_scores_of_liked_pages = {}

    #Predict page scores by taking the mean personality score of everyone who liked each page
    for i in range(n_train):
        person_score = train_people_scores[i].copy()
        person_likes, = train_people_likes[i]
        for page_idx in person_likes:
            if page_idx in person_scores_of_liked_pages:
                person_scores_of_liked_pages[page_idx] = np.vstack([
                    person_scores_of_liked_pages[page_idx],
                    person_score
                ])
            else:
                person_scores_of_liked_pages[page_idx] = person_score

    pred_page_scores = [np.mean(person_scores_of_liked_pages[i], axis=0) for i in range(M)]

    #Predict person scores by taking the average of all the *predicted* page scores they liked
    pred_test_people_scores = np.zeros((n_test, 5))
    for i in range(n_test):
        page_likes, = test_people_likes[i]
        page_likes_matrix = np.zeros((len(page_likes), 5))
        for j, page_idx in enumerate(page_likes):
            page_likes_matrix[j, :] = pred_page_scores[page_idx]
        pred_test_people_scores[i, :] = np.mean(page_likes_matrix, axis=0)
    
    return pred_test_people_scores

def method_2(train_people_likes, train_people_scores, test_people_likes, M):
    """make a person a vector of all the pages they liked and didnt like and run 
    decision tree"""    
    people_train, people_test, classes = discritize_data(train_people_likes, test_people_likes, 8, M)

    #find closest class for each person in training data
    train_people_scores_disc = []
    for score in train_people_scores:
        train_people_scores_disc.append(np.argmin(
            np.linalg.norm(classes - score, axis=1)
        ))

    #build model
    clf = tree.DecisionTreeClassifier()

    #fit
    clf.fit(people_train, train_people_scores_disc)

    #predict classes
    class_prediction = clf.predict(people_test)

    #get personality predictions
    test_people_scores_disc = classes[class_prediction]

    return test_people_scores_disc


def method_3(train_people_likes, train_people_scores, test_people_likes, M):
    """make a person a vector of all the pages they liked and didnt like and run 
    KNN"""
    people_train, people_test, classes = discritize_data(train_people_likes, test_people_likes, 8, M)
    
    #find closest class for each person in training data
    train_people_scores_disc = []
    for score in train_people_scores:
        train_people_scores_disc.append(np.argmin(
            np.linalg.norm(classes - score, axis=1)
        ))

    #create model
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

    #train model
    knn.fit(people_train, train_people_scores_disc)

    #predict
    test_people_scores_disc = classes[knn.predict(people_test)]
    return test_people_scores_disc

# def method_4():


def test_methods():
    #CONSTS
    #######

    #amount of pages
    M = 40
    PAGE_SCORES = gen_ocean_score(M)
    #numher of pages each person will like
    PAGE_LIKE_COUNT = 5

    #training Data
    ##############
    n_train = 10000
    train_people_scores = gen_ocean_score(n_train)
    likes_train = np.zeros((n_train, 5))
    train_people_likes = [gen_page_likes(train_people_scores[i], PAGE_SCORES, PAGE_LIKE_COUNT) for i in range(n_train)]

    #testing data
    #############
    n_test = 200
    test_people_scores = gen_ocean_score(n_test)
    test_people_likes = [gen_page_likes(test_people_scores[i], PAGE_SCORES, PAGE_LIKE_COUNT) for i in range(n_test)]

    predictions = [
        [method_1(train_people_scores, train_people_likes, test_people_likes, M, n_train, n_test), "Mean to Mean"],
        [method_2(train_people_likes, train_people_scores, test_people_likes, M), "Decision Tree"],
        [method_3(train_people_likes, train_people_scores, test_people_likes, M), "KNN"]
        ]

    for p, name in predictions:
        #test models
        ###########
        print(f"{name}: MSE = ",
                np.mean(
                    np.linalg.norm(p - test_people_scores, axis=1)
                )
            )

test_methods()