from gen_data import *
import numpy as np
from scipy.optimize import leastsq, least_squares
from tqdm import tqdm
def test_leastsq():
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

    #train model
    ############
    person_scores_of_liked_pages = {}

    #Predict page scores by taking the mean personality score of everyone who liked each page
    print("Predicting page scores bassed on training data")
    for i in tqdm(range(n_train)):
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

    # TODO okay so we are going to try the information geometry methods (look up InformationGeometry-Exp.ipynb)
    # r stands for residual or error, but its not right
    #   Variance for a uniform random distribution on (a,b) is (b-a)^2 / 12, but do we have to factor in the gaussian variance as well? Well since its the identity, and I think you just invert the covariance, it wouldn't change the answer
    #   lstsq takes in a vector to fit
    def r(x, unkown_likes):
        # TODO penalize really bad x values somehow
        # Attempting to penalize crazy x values
        if np.max(np.abs(x)) > 1.0:
            return np.ones(x.shape) * np.exp(np.max(np.abs(x)))
        unkown_like_ones = np.zeros(PAGE_SCORES.shape[0])
        unkown_like_ones[unkown_likes] = 1
        # Try some monte carlo
        input_mc = []
        for _ in range(50):
            input_like_ones = np.zeros(PAGE_SCORES.shape[0])
            input_like_ones[gen_page_likes(x, PAGE_SCORES, PAGE_LIKE_COUNT, include_noise=False)] = 1
            input_mc.append(input_like_ones)
        input_mc = np.array(input_mc)
        conglomerant = np.mean(input_mc, axis=0)
        conglomerant[conglomerant > 0.5] = 1.0
        conglomerant[conglomerant < 0.5] = 0.0
        # TODO TODO figure out how to find the difference correctly for this method, nothing seems to be working
        #   Um I think the way I'm using this its supposed to be the differences of each one
        # result = np.power(np.abs(np.power(unkown_like_ones, 2) - np.power(conglomerant, 2)), 0.5)
        # return result / 3.0
        # return np.linalg.norm(unkown_like_ones - conglomerant, ord=2) / 3.0
        # return np.linalg.norm(unkown_like_ones - conglomerant, ord=2)
        return unkown_like_ones - conglomerant # this seems to work the best
        # return (unkown_like_ones - input_like_ones) / 3.0
        # return ((unkown_likes - gen_page_likes(x, PAGE_SCORES, PAGE_LIKE_COUNT, include_noise=False))/3.0).reshape(5)

    #Predict person scores by taking the average of all the *predicted* page scores they liked
    print("Predicting test data scores")
    pred_test_people_scores = np.zeros((n_test, 5))
    for i in tqdm(range(n_test)):
        page_likes, = test_people_likes[i]
        page_likes_matrix = np.zeros((len(page_likes), 5))
        for j, page_idx in enumerate(page_likes):
            page_likes_matrix[j, :] = pred_page_scores[page_idx]

        # pred_test_people_scores[i, :] = np.mean(page_likes_matrix, axis=0)

        pred_score = np.mean(page_likes_matrix, axis=0)
        # TODO scipy also has a least_squares function, maybe try using that?
        final_pred, msg = leastsq(r, pred_score, args=(page_likes))
        # final_pred, msg = least_squares(r, pred_score, args=(page_likes))
        # final_pred, msg = leastsq(r, gen_ocean_score(), args=(page_likes))
        pred_test_people_scores[i, :] = final_pred
        
    # print("Using just mean of predicted page scores")
    print("Using leastsq to try to fit better")

    #test model
    ###########
    print("MSE: ",
            np.mean(
                np.linalg.norm(pred_test_people_scores - test_people_scores, axis=1)
            )
        )
    return np.mean( np.linalg.norm(pred_test_people_scores - test_people_scores, axis=1))

leastsq_mean = test_leastsq()
for i in range(9):
    leastsq_mean = np.vstack([leastsq_mean, test_leastsq()])
print(np.mean(leastsq_mean, axis=0))
# from itertools import permutations 
# from sklearn import tree    
# from sklearn.neighbors import KNeighborsClassifier


# def discritize_data(train_people_likes, test_people_likes, permut_n, M):
#     #format person data 
#     people_train = []
#     for likes in train_people_likes:
#         person = np.zeros(M)
#         person[likes] = 1
#         people_train.append(person)
    
#     people_test = []
#     for likes in test_people_likes:
#         person = np.zeros(M)
#         person[likes] = 1
#         people_test.append(person)

#     #discritize range of personalities for classification 
#     classes = np.array(
#                 list(permutations(np.linspace(-1, 1, permut_n), 5))
#                 )
#     return people_train, people_test, classes


# def method_1(train_people_scores, train_people_likes, test_people_likes, M, n_train, n_test):
#     """Assume every page has a personality score and guess it based off data. Guess a persons personality
#     score to be the average of all the pges they liked. 
#     """
#     #train model
#     ############
#     person_scores_of_liked_pages = {}

#     #Predict page scores by taking the mean personality score of everyone who liked each page
#     for i in range(n_train):
#         person_score = train_people_scores[i].copy()
#         person_likes, = train_people_likes[i]
#         for page_idx in person_likes:
#             if page_idx in person_scores_of_liked_pages:
#                 person_scores_of_liked_pages[page_idx] = np.vstack([
#                     person_scores_of_liked_pages[page_idx],
#                     person_score
#                 ])
#             else:
#                 person_scores_of_liked_pages[page_idx] = person_score

#     pred_page_scores = [np.mean(person_scores_of_liked_pages[i], axis=0) for i in range(M)]

#     #Predict person scores by taking the average of all the *predicted* page scores they liked
#     pred_test_people_scores = np.zeros((n_test, 5))
#     for i in range(n_test):
#         page_likes, = test_people_likes[i]
#         page_likes_matrix = np.zeros((len(page_likes), 5))
#         for j, page_idx in enumerate(page_likes):
#             page_likes_matrix[j, :] = pred_page_scores[page_idx]
#         pred_test_people_scores[i, :] = np.mean(page_likes_matrix, axis=0)
    
#     return pred_test_people_scores

# def method_2(train_people_likes, train_people_scores, test_people_likes, M):
#     """make a person a vector of all the pages they liked and didnt like and run 
#     decision tree"""    
#     people_train, people_test, classes = discritize_data(train_people_likes, test_people_likes, 8, M)

#     #find closest class for each person in training data
#     train_people_scores_disc = []
#     for score in train_people_scores:
#         train_people_scores_disc.append(np.argmin(
#             np.linalg.norm(classes - score, axis=1)
#         ))

#     #build model
#     clf = tree.DecisionTreeClassifier()

#     #fit
#     clf.fit(people_train, train_people_scores_disc)

#     #predict classes
#     class_prediction = clf.predict(people_test)

#     #get personality predictions
#     test_people_scores_disc = classes[class_prediction]

#     return test_people_scores_disc


# def method_3(train_people_likes, train_people_scores, test_people_likes, M):
#     """make a person a vector of all the pages they liked and didnt like and run 
#     KNN"""
#     people_train, people_test, classes = discritize_data(train_people_likes, test_people_likes, 8, M)
    
#     #find closest class for each person in training data
#     train_people_scores_disc = []
#     for score in train_people_scores:
#         train_people_scores_disc.append(np.argmin(
#             np.linalg.norm(classes - score, axis=1)
#         ))

#     #create model
#     knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

#     #train model
#     knn.fit(people_train, train_people_scores_disc)

#     #predict
#     test_people_scores_disc = classes[knn.predict(people_test)]
#     return test_people_scores_disc

# def baseline(train_people_likes, train_people_scores, test_people_likes, M):
#     estimates = []
#     for _ in test_people_likes:
#         estimates.append(gen_ocean_score()[0])
#     return np.array(estimates)


# def test_methods():
#     #CONSTS
#     #######

#     #amount of pages
#     M = 40
#     PAGE_SCORES = gen_ocean_score(M)
#     #numher of pages each person will like
#     PAGE_LIKE_COUNT = 5

#     #training Data
#     ##############
#     n_train = 10000
#     # n_train = 3000
#     # n_train = 100
#     train_people_scores = gen_ocean_score(n_train)
#     likes_train = np.zeros((n_train, 5))
#     train_people_likes = [gen_page_likes(train_people_scores[i], PAGE_SCORES, PAGE_LIKE_COUNT) for i in range(n_train)]

#     #testing data
#     #############
#     n_test = 200
#     test_people_scores = gen_ocean_score(n_test)
#     test_people_likes = [gen_page_likes(test_people_scores[i], PAGE_SCORES, PAGE_LIKE_COUNT) for i in range(n_test)]

#     predictions = [
#         [baseline(train_people_likes, train_people_scores, test_people_likes, M), "Baseline (Random Guessing)"],
#         [method_1(train_people_scores, train_people_likes, test_people_likes, M, n_train, n_test), "Mean to Mean"],
#         [method_2(train_people_likes, train_people_scores, test_people_likes, M), "Decision Tree"],
#         [method_3(train_people_likes, train_people_scores, test_people_likes, M), "KNN"],
#         ]

#     for p, name in predictions:
#         #test models
#         ###########
#         print(f"{name}: MSE = ",
#                 np.mean(
#                     np.linalg.norm(p - test_people_scores, axis=1)
#                 )
#             )

# test_methods()
