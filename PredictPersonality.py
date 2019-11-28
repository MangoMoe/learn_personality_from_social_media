from gen_data import *
import numpy as np
from scipy.optimize import leastsq

#CONSTS
#######

#amount of pages
M = 40
PAGE_SCORES = gen_ocean_score(M)
#numher of pages each person will like
PAGE_LIKE_COUNT = 5

#training Data
##############
n_train = 3000
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

# TODO okay so we are going to try the information geometry methods

# Not sure what r stands for (maybe error), but I think I know what it does
#   ys can be the likes vector for an unknown person, f can be the generated page likes for the given parameters
#   TODO I think 0.01 is the variance but like I'm not exactly sure how to do that for a freaking matrix (especially since the "noise" is uniform random...)
#       Variance for a uniform random distribution on (a,b) is (b-a)^2 / 12, but do we have to factor in the gaussian variance as well? Well since its the identity, and I think you just invert the covariance, it wouldn't change the answer
#   lstsq takes in a vector to fit
def r(x, unkown_likes):
    # print(x.shape)
    # print(((unkown_likes - gen_page_likes(x, PAGE_SCORES, PAGE_LIKE_COUNT))/3.0).reshape(5).shape)
    return ((unkown_likes - gen_page_likes(x, PAGE_SCORES, PAGE_LIKE_COUNT))/3.0).reshape(5)

#Predict person scores by taking the average of all the *predicted* page scores they liked
pred_test_people_scores = np.zeros((n_test, 5))
for i in range(n_test):
    page_likes, = test_people_likes[i]
    # page_likes_matrix = np.zeros((len(page_likes), 5))
    # for j, page_idx in enumerate(page_likes):
    #     page_likes_matrix[j, :] = pred_page_scores[page_idx]

    # pred_test_people_scores[i, :] = np.mean(page_likes_matrix, axis=0)

    # pred_score = np.mean(page_likes_matrix, axis=0)
    # final_pred, msg = leastsq(r, pred_score(), args=(page_likes))
    final_pred, msg = leastsq(r, gen_ocean_score(), args=(page_likes))
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