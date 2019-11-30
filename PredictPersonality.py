from gen_data import *
import numpy as np
from scipy.optimize import leastsq, least_squares
from tqdm import tqdm

#CONSTS
#######

#amount of pages
M = 40
PAGE_SCORES = gen_ocean_score(M)
#numher of pages each person will like
PAGE_LIKE_COUNT = 5

#training Data
##############
n_train = 300
train_people_scores = gen_ocean_score(n_train)
likes_train = np.zeros((n_train, 5))
train_people_likes = [gen_page_likes(train_people_scores[i], PAGE_SCORES, PAGE_LIKE_COUNT) for i in range(n_train)]

#testing data
#############
n_test = 500
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