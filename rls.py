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

# non transpose version
# q = np.zeros((m,5)) # q[t]
# transpose version
q = np.zeros((5,m)) # q[t]

delta = 0.00001
# P = (1/delta) * np.identity(m) # P[t - 1]
P = (1/delta) * np.identity(5) # P[t - 1]
h = np.zeros((1,5)) # initialize means to zero

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

    # TODO find a more efficient way to do this
    # TODO should this be 1 or identity? well 1 gives a singular matrix
    # TODO I'm not sure which transposition of the initial q is correct
    k = (P@q)@np.linalg.inv(np.identity(q.shape[1]) + q.T@P@q) # TODO understand this better
    # TODO this will have to be generated carefully, its not an entirely matrix operation
    # generate a value for d, then use a multivariate gaussian and the other
    # pick page
    like = gd.gen_page_likes(plant, pages, 1)[0]
    # then get page's score
    d = pages[like]
    # then do gaussian with 0 mean
    cov = np.identity(5)
    # TODO maybe just use h as the means
    estimate = np.random.multivariate_normal(h[0], cov, 1)[0]
    # TODO so like... this error is different.... than the algorithm... hopefully it works
    e = d - estimate
    # TODO TODO if all else fails, you could just use 5 different RLS algorithms to find each mean
    print(e)

# %%
