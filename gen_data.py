import numpy as np
from numpy import linalg as LA

mean = np.array([1,1,1,1,1])

traits_to_index = {"openness": 0, "conscientiousness": 1, "extraversion": 2, "agreeableness": 3, "neuroticism": 4}
index_to_traits = {0: "openness", 1: "conscientiousness", 2: "extraversion", 3: "agreeableness", 4: "neuroticism"}

# generates 5 ocean scores between -1 and 1
# the order of the scores are [openness, Conscientiousness, extraversion, agreeableness, neuroticism]
# TODO find probabilities of different ocean scores even being generated (i.e. are open people more common than non open people, etc.)
def gen_ocean_score(n=None):
    # TODO should this be a uniform distribution or like a normal distribution with a mean of 0?
    if n is None:
        # default to 1
        scores = np.random.uniform(-1.0, 1.0, (1,5))
    else:
        scores = np.random.uniform(-1.0, 1.0, (n,5))
    return scores

def gen_page_likes(ocean_score, page_scores, n, include_noise=True):
    """Each facebook page should have a personality score.
    A page like is generated from taking a draw
    from a normal distribution centered around a persons
    personality and then finding the closest page score
    to that draw.
    
    ocean_score (nd-array) - used as mean of normal distribution
    page_score (nd-array) - ocean scores of each page
    n (int) - number of page likes to generate
    """

    if len(ocean_score.shape) == 1:
        ocean_score = ocean_score.reshape(1, ocean_score.shape[0])
    #make sure there are more uniqe pages than desired likes
    m = min(n, len(page_scores))
    cov = np.identity(5)
    page_likes = []
    # TODO this might be a problem we need to solve
    # To avoid repeated page likes, I implemented it like this. 
    #   Deleting things from the page_scores list is problematic because it changes the indicies
    #   This might also be problematic because it might loop a lot trying to get pages that are far from the person's score (for example if n is the same length as page_scores)
    # print(ocean_score.shape[0])
    for i in range(ocean_score.shape[0]):
        page_likes.append([])
        while len(page_likes[i]) < m:
            # 30% chance that the person picks a totally random page
            if np.random.uniform() < 0.3 and include_noise:
                draw = np.random.uniform(-1.0, 1.0, (1,5))
            else:
                draw  = np.random.multivariate_normal(ocean_score[i], cov, 1)[0]
            #take the norm between the draw and every page and then select
            #the index of the smallest distance
            closest_page_idx = np.argmin(
                    LA.norm(draw - page_scores, ord=2, axis = 1)
            )
            # avoid duplicate likes
            if closest_page_idx not in page_likes[i]:
                page_likes[i].append(closest_page_idx)
            
    return np.array(page_likes)

# generates posts corresponding to a person with the given ocean score.
# draws from a multivariate with means at the ocean scores, then finds the maximum score (pos or neg) in the drawn score and sets that to +- 1.0 and the rest to 0
def gen_posts(ocean_score, n):
    """ lorem ipsum... """
    # TODO what the heck should the covariance matrix be?
    cov = np.identity(5)
    posts = []
    for _ in range(n):
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
    return np.array(posts)

# ------ TESTING IT
# posts = gen_posts(gen_ocean_score(), 5)
# print(posts)
# # TODO perhaps this format will be better for posts to return? 
# #   Although maybe it loses some information (ex 1 + -1 = 0 but contains different info from 0)
# print(np.sum(posts, axis=0))

# # page_scores = []
# # for i in range(5):
# #     page_scores.append(gen_ocean_score())
# # page_scores = np.array(page_scores)
# page_scores = gen_ocean_score(5)

# person = gen_ocean_score()
# result = gen_page_likes(person, page_scores, 3)
# print(len(result.shape))

# # TODO to check that this really works, we need to print/save the 
# #   draws from the multivariate and verify that these page scores were closest...
# for index in result:
#     print(page_scores[index])