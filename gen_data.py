import numpy as np
from numpy import linalg as LA

mean = np.array([1,1,1,1,1])

traits_to_index = {"openness": 0, "conscientiousness": 1, "extraversion": 2, "agreeableness": 3, "neuroticism": 4}
index_to_traits = {0: "openness", 1: "conscientiousness", 2: "extraversion", 3: "agreeableness", 4: "neuroticism"}

# generates 5 ocean scores between -1 and 1
# the order of the scores are [openness, Conscientiousness, extraversion, agreeableness, neuroticism]
# TODO make this take in as a parameter how many to generate
def gen_ocean_score():
    # TODO should this be a uniform distribution or like a normal distribution with a mean of 0?
    scores = np.random.uniform(-1.0, 1.0, 5)
    return scores

def gen_page_likes(ocean_score, page_scores, n):
    """Each facebook page should have a personality score.
    A page like is generated from taking a draw
    from a normal distribution centered around a persons
    personality and then finding the closest page score
    to that draw.
    
    ocean_score (nd-array) - used as mean of normal distribution
    page_score (nd-array) - ocean scores of each page
    n (int) - number of page likes to generate
    """
    #make sure there are more uniqe pages than desired likes
    m = min(n, len(page_scores))
    cov = np.identity(5)
    page_likes = []
    for _ in range(m):
        draw  = np.random.multivariate_normal(ocean_score, cov, 1)
        #take the norm between the draw and every page and then select
        #the index of the smallest distance
        closest_page_idx = np.argmin(
                LA.norm(draw - page_scores, ord=2, axis = 1)
        )
        page_likes.append(closest_page_idx)
        #remove page to avoid duplicate likes
        # TODO TODO this is a problem because removing the page changes the indexes of the other pages after it, so you can get the same index multiple times
        page_scores = np.delete(page_scores, closest_page_idx, 0)
        
    return page_likes

# generates posts corresponding to a person with the given ocean score.
# draws from a multivariate with means at the ocean scores, then finds the maximum score (pos or neg) in the drawn score and sets that to +- 1.0 and the rest to 0
def gen_posts(ocean_score, n):
    """ lorem ipsum... """
    # TODO what the heck should the covariance matrix be?
    cov = np.identity(5)
    posts = []
    for i in range(n):
        draw = np.random.multivariate_normal(ocean_score, cov).T
        # print(draw)
        max_score = max(draw.min(), draw.max(), key=abs)
        if max_score not in draw:
            # re-negative it
            max_score = -max_score
        for j in range(len(draw)):
            if draw[j] != max_score:
                draw[j] = 0
            else:
                draw[j] = 1.0 * np.sign(max_score)
        # print(draw)
        posts.append(draw)

# gen_posts(gen_ocean_score(), 5)

page_scores = []
for i in range(5):
    page_scores.append(gen_ocean_score())
page_scores = np.array(page_scores)

# print(page_scores)
person = gen_ocean_score()
# print(person)
result = gen_page_likes(person, page_scores, 3)
print(result)
