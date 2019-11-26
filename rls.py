# %%
import numpy as np
import gen_data as gd

# TODO okay so here's the plan
#   firstly we make the person, then we generate pages to like (try a fixed amount to start)
#   Then generate a stream of likes (output of data vector is just which pages they liked)
#   Have your f be a gaussian generating scores with same variance (for now) as target distribution
#   Coefficients are the means?

# %%
# Create actual plant
plant = gd.gen_ocean_score()
print(plant)
# create pages
num_pages = 1000
print("Number of pages: {}".format(num_pages))
pages = gd.gen_ocean_score(num_pages)
# TODO so maybe set all values for f to be 1 and then multiply those by the coefficients to get what to add to a regular gaussian

# %%
# set up variables
#   Since most of these have a time subscript but we only need the latest one so we don't need to store all the others
m = 10 # window
d = []
# f = np.ones((5,m))

# non transpose version
# q = np.zeros((m,5)) # q[t]
# transpose version
q = np.zeros((5,m)) # q[t]

delta = 0.00001
# P = (1/delta) * np.identity(m) # P[t - 1]
P = (1/delta) * np.identity(5) # P[t - 1]
h = np.zeros((1,5)) # initialize means to zero
# t = 0 # just for funzies so we can track

# %%
for t in range(1,15): # TODO I'm not sure this is right

    # generate q
    #   q[i] = [f[i],f[i - 1]...f[i - m + 1]].T

    # non transpose version
    # if t < m:
    #     for i in range(t,-1,-1):
    #         q[i,:] = np.ones(5)
    #     q[t:, :] = np.zeros((m - t, 5))
    # else:
    #     # TODO will this slow things down? Heck does generating a new ones matrix slow things down?
    #     q = np.ones((m,5))
    # transpose version
    if t < m:
        for i in range(t,-1,-1):
            q[:,i] = np.ones(5)
        q[:, t:] = np.zeros((5, m-t))
    else:
        # TODO will this slow things down? Heck does generating a new ones matrix slow things down?
        q = np.ones((5,m))


    # print("t: {}".format(t))
    # print(q)
    # print(P.shape)
    # print(q.shape)
    # print((P@q).shape)
    # print((q.T@P).shape)
    # print((q.T@P@q).shape)
    # print(np.identity(q.shape[1]) + q.T@P@q)
    # TODO find a more efficient way to do this
    # TODO should this be 1 or identity? well 1 gives a singular matrix
    # TODO I'm not sure which transposition of the initial q is correct
    # k = (P@q)@np.linalg.inv(np.identity(q.shape[1]) + q.T@P@q)
    k = (P@q)@np.linalg.inv(np.identity(q.shape[1]) + q.T@P@q) # TODO understand this better
    # print(q.T.shape)
    # print(h.shape)
    # TODO I'm not entirely sure this is right... shouldn't the output be
    #   TODO TODO see below silly
    # print((q@h).shape)
    # print((q.T@h.T).shape)
    # TODO this will have to be generated carefully, its not an entirely matrix operation
    # generate a value for d, then use a multivariate gaussian and the other
    # pick page
    like = gd.gen_page_likes(plant, pages, 1)[0]
    # then get page's score
    d = pages[like]
    # then do gaussian with 0 mean
    cov = np.identity(5)
    # TODO maybe just use h as the means
    # print(h[0])
    estimate = np.random.multivariate_normal(h[0], cov, 1)[0]
    # print(estimate)
    # then add q.T@h?
    e = d - estimate
    print(e)

# %%
