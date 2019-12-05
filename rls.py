# %%
import numpy as np
import gen_data as gd
from tqdm import tqdm

# New plan, 
#   stream of data is stream of likes from 5 people with known scores,
#   b vector is stream of data with person with unknown scores. 
#   Parameters are each person's "similarity" to the unknown person.
#   just stack the ocean scores of the output pages (so basically the page ocean scores are known) vertically because the similarity is the same for the different ocean parameters of each person
# TODO document what you reused

NUM_KNOWN_PEOPLE = 30
NUM_PAGES = 400
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

WINDOW = 100
NUM_ITER = 1000

# P = R^-1, R = A^H A
# f is input signal, which is probably A or in other words, our stream of data from the other people
# d is data or the signal we are trying to follow, so b
# TODO don't calculate the inverse directly?
# initialize
t = 0
delta = 0.001
# d = [b]
fs = [A]
data = [b]
k = 0 # initial value for K is empty
# TODO do we need to store the previous P matricies? probably not, also probably not k
# P = (1/delta) * np.identity(5 * WINDOW)
P = (1/delta) * np.identity(NUM_KNOWN_PEOPLE)
h = params
q_init = np.zeros((5 * WINDOW, NUM_KNOWN_PEOPLE))
d_init = np.zeros((5 * WINDOW, 1))
for t in tqdm(range(1,NUM_ITER+1)):
    A, b = gen_likes_matricies(known_people, pages, unknown_person)
    fs.append(A)
    data.append(b)
    if t < WINDOW + 1:
        q = np.zeros(q_init.shape)
        for i in range(t):
            q[i *5: i*5+5,:] = fs[t - i]
        d = np.zeros(d_init.shape)
        for i in range(t):
            d[i *5: i*5+5,:] = data[t - i]
    else:
        q = np.vstack(fs[t:t - WINDOW + 1 - 1:-1])
        d = np.vstack(data[t:t - WINDOW + 1 - 1:-1])
    q = q.T # apparently this worked... TODO make sure this is the case

    k = P@q@np.linalg.inv(np.identity(5 * WINDOW) + q.T@P@q) # TODO I used identity instead of one...
    P = P - k@q.T@P
    h = h + k@(d - q.T@h)

# at the end, predict the score of the person by combining the other people's scores
# 
estimated_score = (known_people.T@h).T
print("Estimated Person score: {}".format(estimated_score))
print("Actual Person score: {}".format(unknown_person))
print("Error: {}".format(estimated_score - unknown_person))
print("Norm of error: {}".format(np.linalg.norm(estimated_score - unknown_person, ord=2)))



