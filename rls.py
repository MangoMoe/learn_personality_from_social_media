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
print(params)

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
print(A)
print(b)
new_A, new_b = gen_likes_matricies(known_people, pages, unknown_person)
A = np.vstack([ A, new_A ])
b = np.vstack([ b, new_b ])
print(A)
print(b)


