import gen_data as gd
import numpy as np

# generate p pages

p = 10
print("Number of pages: {}".format(p))
pages = gd.gen_ocean_score(p)

# generate x people who's scores are known
x = 1
print("Number of known scores: {}".format(x))
known_people = gd.gen_ocean_score(x)
# TODO should we do the same thing with other functions allowing it to return a multidimensional array?
num_likes = 5
print("Number of likes per person: {}".format(num_likes))
known_likes = gd.gen_page_likes(known_people, pages, num_likes)
known_likes_ones = []
# convert likes into a ones vector
for like in known_likes:
    ones = np.zeros(pages.shape[0])
    ones[like] = 1
    known_likes_ones.append(ones)
known_likes_ones = np.array(known_likes_ones)

# generate y unknown people
y = 100
print("Number of unknown scores: {}".format(y))
unknown_people = gd.gen_ocean_score(y)
unknown_likes = gd.gen_page_likes(unknown_people, pages, num_likes)

# print("pages")
# print(pages)
# print("known_people")
# print(known_people)
# print("known_likes")
# print(known_likes)

# now try to cluster a person
#   first idea: just average of known people's scores weighted by their distance from those people's page likes
#   Second idea: just cluster them, and then take some sort of distance metric from clusters to figure out score probably...
#   third idea: try to generate our own page likes for a person and use some sort of gradient descent or nonlinear regression
#   fourth idea: maximum likelihood of gaussian that produced that page like combo??? is that legal? model can't know page ocean scores
guessed_score = []
two_norm_total = 0
one_norm_total = 0
for i, like in enumerate(unknown_likes):
    # turn likes into a vector of zeros and ones
    like_ones = np.zeros(pages.shape[0])
    like_ones[like] = 1
    # find the distance from each known person's likes
    dists = []
    for known_like in known_likes_ones:
        dist = np.linalg.norm(known_like - like_ones, ord=1)
        dists.append(dist)
    dists = np.array(dists)
    # do a weighted average of each of their ocean scores
    estimate = np.zeros(5)
    # TODO figure out why you are getting a 0 in the denominator (it has something to do with the unknown person having the same likes as the known person)
    for j, dist in enumerate(dists):
        print("{} / {} = {}".format(dist, np.sum(dists), dist / np.sum(dists)))
        print(known_likes[j])
        print(unknown_likes[i])
        estimate += known_people[j] * (dist / np.sum(dists))
    # see how close it is to the actual one
    # print("Guess vs actual value")
    # print(estimate)
    # print(unknown_people[i])
    # print("one and two norms of error")
    # print(np.linalg.norm(estimate - unknown_people, ord=1))
    # print(np.linalg.norm(estimate - unknown_people, ord=2))
    one_norm_total += np.linalg.norm(estimate - unknown_people[i], ord=1)
    two_norm_total += np.linalg.norm(estimate - unknown_people[i], ord=2)

print("Average one norm: {}".format(one_norm_total / y))
print("Average two norm: {}".format(two_norm_total / y))