# %%
import numpy as np
import gen_data as gd

# New plan, 
#   stream of data is stream of likes from 5 people with known scores,
#   b vector is stream of data with person with unknown scores. 
#   Parameters are each person's "similarity" to the unknown person.
#   just stack the ocean scores of the output pages (so basically the page ocean scores are known) vertically because the similarity is the same for the different ocean parameters of each person
# TODO document what you reused

NUM_KNOWN_PEOPLE = 3
NUM_PAGES = 40
pages = gd.gen_ocean_score(NUM_PAGES)
# TODO merge other code into this so we have the updated page likes generation function probably...
known_people = gd.gen_ocean_score(NUM_KNOWN_PEOPLE)
unknown_person = gd.gen_ocean_score()
params = np.zeros((1, NUM_KNOWN_PEOPLE)).reshape((1,NUM_KNOWN_PEOPLE)).T

# TODO this could end up being too hard, you might have to use the same draw to produce all the vectors
def gen_likes_matricies(known_people, pages, unknown_person):
    # Return A matrix (well the additional part anyway) and the b vector (additional part)
    
    # x person is cool
    A = np.array([])
    A = np.concatenate([pages[gd.gen_page_likes(person, pages, 1)].reshape(1,5) for person in known_people])
    # for person in known_people:
    #     person_like = gd.gen_page_likes(person, pages, 1)
    #     liked_page_score = pages[person_like].reshape(1,5)
    #     A = np.concatenate([A, person_like])
    
    b = pages[gd.gen_page_likes(unknown_person, pages, 1)].reshape(1,5)
    return A.T, b.T

A, b = gen_likes_matricies(known_people, pages, unknown_person)
# new_A, new_b = gen_likes_matricies(known_people, pages, unknown_person)
# A = np.vstack([ A, new_A ])
# b = np.vstack([ b, new_b ])

WINDOW = 2

# P = R^-1, R = A^H A
# f is input signal, which is probably A or in other words, our stream of data from the other people
# d is data or the signal we are trying to follow, so b
# TODO don't calculate the inverse directly?
# initialize
t = 0
delta = 0.001
# d = [b]
fs = [A]
k = 0 # initial value for K is empty
# TODO do we need to store the previous P matricies? probably not, also probably not k
P = (1/delta) * np.identity()
h = params
q = np.zeros(5 * WINDOW, NUM_KNOWN_PEOPLE)
NUM_ITER = 5
for t in range(NUM_ITER)
    A, d = gen_likes_matricies(known_people, pages, unknown_person)
    fs.append(A)
    if t < WINDOW + 1:
        q = np.zeros(q.shape)
        # TODO test these slices to make sure they work right...
        q[:t*5, :] = np.vstack(fs[t::-1])
    else:
        q = np.vstack(fs[t:t - WINDOW + 1:-1])

    # TODO 1 or identity?
    k = P@q@np.inv(1 + q.T@P@q)
    P = P - k@q.T@P
    h = h + k(d - q.T@params)

# at the end, predict the score of the person by combining the other people's scores
# 
estimated_score = known_people.T@h
print(estimated_score)
print(unknown_person)
print(estimated_score - unknown_person)



