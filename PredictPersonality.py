from gen_data import *
import numpy as np

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

#Predict person scores by taking the average of all the *predicted* page scores they liked
pred_test_people_scores = np.zeros((n_test, 5))
for i in range(n_test):
    page_likes, = test_people_likes[i]
    page_likes_matrix = np.zeros((len(page_likes), 5))
    for j, page_idx in enumerate(page_likes):
        page_likes_matrix[j, :] = pred_page_scores[page_idx]
    pred_test_people_scores[i, :] = np.mean(page_likes_matrix, axis=0)
    
#test model
###########
print("MSE: ",
        np.mean(
            np.linalg.norm(pred_test_people_scores - test_people_scores, axis=1)
        )
    )