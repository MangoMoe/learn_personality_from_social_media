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

def rls_fit(num_known_people, num_pages, likes_per_person, window, num_iter, hide_page_scores=True):

    NUM_KNOWN_PEOPLE = num_known_people
    NUM_PAGES = num_pages
    LIKES_PER_PERSON = likes_per_person if hide_page_scores else 1

    WINDOW = window
    NUM_ITER = num_iter
    if hide_page_scores:
        DATA_SIZE = NUM_PAGES # use those one vectors composed of which pages they liked
    else:
        DATA_SIZE = 5 # length of an ocean score
    # print("\n\nRunning RLS Algorithm.\n Number of known people is {},\n Number of pages is {},\n Likes per person is {},\n window size is {} sets of data,\n we are iterating {} times,\n and are we hiding the page scores? {}".format(
    #     NUM_KNOWN_PEOPLE, NUM_PAGES, LIKES_PER_PERSON, WINDOW, NUM_ITER, "Yes" if hide_page_scores else "No"
    # ))
    pages = gd.gen_ocean_score(NUM_PAGES)
    known_people = gd.gen_ocean_score(NUM_KNOWN_PEOPLE)
    unknown_person = gd.gen_ocean_score()
    params = np.zeros((1, NUM_KNOWN_PEOPLE)).reshape((1,NUM_KNOWN_PEOPLE)).T

    def gen_likes_matricies(known_people, pages, unknown_person, hide_page_scores=True):
        # Return A matrix (well the additional part anyway) and the b vector (additional part)
        if hide_page_scores:
            stuff = []
            for person in known_people:
                one_vec = np.zeros(pages.shape[0])
                one_vec[gd.gen_page_likes(person, pages, LIKES_PER_PERSON)] = 1
                stuff.append(one_vec.reshape(1, DATA_SIZE))

            A = np.concatenate(stuff)

            b = np.zeros(pages.shape[0])
            b[gd.gen_page_likes(unknown_person, pages, LIKES_PER_PERSON)] = 1
            b = b.reshape(1,DATA_SIZE)
        else:
            A = np.concatenate([pages[gd.gen_page_likes(person, pages, LIKES_PER_PERSON)].reshape(LIKES_PER_PERSON,DATA_SIZE) for person in known_people])
            b = pages[gd.gen_page_likes(unknown_person, pages, LIKES_PER_PERSON)].reshape(1,DATA_SIZE)
            
        return A.T, b.T

    A, b = gen_likes_matricies(known_people, pages, unknown_person, hide_page_scores)

    # P = R^-1, R = A^H A
    # f is input signal, which is probably A or in other words, our stream of data from the other people
    # d is data or the signal we are trying to follow, so b
    # TODO don't calculate the inverse directly?
    # initialize
    t = 0
    delta = 0.001
    fs = [A]
    data = [b]
    k = 0 # initial value for K is empty
    # P = (1/delta) * np.identity(5 * WINDOW)
    P = (1/delta) * np.identity(NUM_KNOWN_PEOPLE)
    h = params
    q_init = np.zeros((DATA_SIZE * WINDOW, NUM_KNOWN_PEOPLE))
    d_init = np.zeros((DATA_SIZE * WINDOW, 1))
    # for t in tqdm(range(1,NUM_ITER+1)):
    for t in range(1,NUM_ITER+1):
        A, b = gen_likes_matricies(known_people, pages, unknown_person, hide_page_scores)
        fs.append(A)
        data.append(b)
        if t < WINDOW + 1:
            q = np.zeros(q_init.shape)
            for i in range(t):
                q[i *DATA_SIZE: i*DATA_SIZE+DATA_SIZE,:] = fs[t - i]
            d = np.zeros(d_init.shape)
            for i in range(t):
                d[i *DATA_SIZE: i*DATA_SIZE+DATA_SIZE,:] = data[t - i]
        else:
            q = np.vstack(fs[t:t - WINDOW + 1 - 1:-1])
            d = np.vstack(data[t:t - WINDOW + 1 - 1:-1])
        q = q.T # apparently this worked... TODO make sure this is the case

        k = P@q@np.linalg.inv(np.identity(DATA_SIZE * WINDOW) + q.T@P@q) # I used identity instead of one...
        P = P - k@q.T@P
        h = h + k@(d - q.T@h)

    # at the end, predict the score of the person by combining the other people's scores
    if hide_page_scores:
        # TODO trying to figure out correct scaling for h when transitioning from descretized page likes to ocean scores
        h = h / np.linalg.norm(h, ord=1)
        # h = h *  DATA_SIZE / 5
        # h = h * 2
    estimated_score = (known_people.T@h).T
    baseline = gd.gen_ocean_score()

    # print("End parameters: {}".format(h))
    # print("Estimated person score: {}".format(estimated_score))
    # print("Actual Person score: {}".format(unknown_person))
    # # TODO TODO TODO compare to just mean of page scores for person likes
    # #   even if it does work, its useful in other contexts where you are comparing people to see if they are similar
    # # print("Error: {}".format(estimated_score - unknown_person))
    # print("Norm of error: {}".format(np.linalg.norm(estimated_score - unknown_person, ord=2)))
    # # print("Baseline person score: {}".format(baseline))
    # print("Norm of baseline error: {}".format(np.linalg.norm(baseline - unknown_person, ord=2)))
    return np.linalg.norm(estimated_score - unknown_person, ord=2)

AVG_AMOUNT = 10

# Testing various amounts of known scores
# with open("page_scores_num_people2.txt", "w") as f:
#     print("Beginning to test various values for known user scores")
#     num_known_peoples = [1, 10, 30, 100, 300]
#     # num_known_peoples = [1, 10, 30, 100]
#     norms = []
#     for amount in num_known_peoples:
#         runs = []
#         print("testing {} people for known values".format(amount))
#         for i in tqdm(range(AVG_AMOUNT)):
#             runs.append(rls_fit(num_known_people = amount, num_pages = 200, likes_per_person = 10, window = 2, num_iter = 500, hide_page_scores = False))
#         f.write(str(amount) + ", ")
#         runs = np.array(runs)
#         norms.append(np.mean(runs))

#     f.write("\n")
#     for norm in norms:
#         f.write(str(norm) + ", ")
#     print("finished testing that")


# # Testing various amounts of known pages
# with open("page_scores_num_pages2.txt", "w") as f:
#     print("Beginning to test various values for known pages")
#     num_pages = [1, 10, 30, 100, 200, 1000]
#     norms = []
#     for amount in num_pages:
#         runs = []
#         print("testing {} people for known values".format(amount))
#         for i in tqdm(range(AVG_AMOUNT)):
#             runs.append(rls_fit(num_known_people = 300, num_pages = amount, likes_per_person = 10, window = 2, num_iter = 500, hide_page_scores = False))
#         f.write(str(amount) + ", ")
#         runs = np.array(runs)
#         norms.append(np.mean(runs))

#     f.write("\n")
#     for norm in norms:
#         f.write(str(norm) + ", ")
#     print("finished testing that")

# # Testing various amounts of window sizes
# with open("page_scores_window2.txt", "w") as f:
#     print("Beginning to test various values window size")
#     window_size = [1, 2, 5, 10, 30, 100]
#     norms = []
#     for amount in window_size:
#         runs = []
#         print("testing {} people for known values".format(amount))
#         for i in tqdm(range(AVG_AMOUNT)):
#             runs.append(rls_fit(num_known_people = 300, num_pages = 200, likes_per_person = 10, window = amount, num_iter = 500, hide_page_scores = False))
#         f.write(str(amount) + ", ")
#         runs = np.array(runs)
#         norms.append(np.mean(runs))

#     f.write("\n")
#     for norm in norms:
#         f.write(str(norm) + ", ")
#     print("finished testing that")

# # Testing various amounts of number of iterations
# with open("page_scores_num_iter2.txt", "w") as f:
#     print("Beginning to test various values for number of iterations")
#     num_iter = [1, 10, 30, 100, 200, 500, 1000, 2000]
#     norms = []
#     for amount in num_iter:
#         runs = []
#         print("testing {} people for known values".format(amount))
#         for i in tqdm(range(AVG_AMOUNT)):
#             runs.append(rls_fit(num_known_people = 300, num_pages = 200, likes_per_person = 10, window = 2, num_iter = amount, hide_page_scores = False))
#         f.write(str(amount) + ", ")
#         runs = np.array(runs)
#         norms.append(np.mean(runs))

#     f.write("\n")
#     for norm in norms:
#         f.write(str(norm) + ", ")
#     print("finished testing that")

# Testing various amounts of number of known people for unknown pages
# with open("no_scores_num_known_people2.txt", "w") as f:
#     print("Beginning to test various values for number of known people with hidden page scores")
#     num_people = [1, 10, 30, 100, 300]
#     norms = []
#     for amount in num_people:
#         runs = []
#         print("testing {} people for known values".format(amount))
#         for i in tqdm(range(AVG_AMOUNT)):
#             runs.append(rls_fit(num_known_people = amount, num_pages = 200, likes_per_person = 10, window = 2, num_iter = 500, hide_page_scores = True))
#         f.write(str(amount) + ", ")
#         runs = np.array(runs)
#         norms.append(np.mean(runs))

#     f.write("\n")
#     for norm in norms:
#         f.write(str(norm) + ", ")
#     print("finished testing that")

# Testing various amounts of number of pages for unknown pages
with open("no_scores_num_pages2.txt", "w") as f:
    print("Beginning to test various values for number of number of pages with hidden page scores")
    num_pages = [1, 10, 30, 100, 200]
    norms = []
    for amount in num_pages:
        runs = []
        print("testing {} people for known values".format(amount))
        for i in tqdm(range(AVG_AMOUNT)):
            runs.append(rls_fit(num_known_people = 100, num_pages = amount, likes_per_person = 5, window = 2, num_iter = 500, hide_page_scores = True))
        f.write(str(amount) + ", ")
        runs = np.array(runs)
        norms.append(np.mean(runs))

    f.write("\n")
    for norm in norms:
        f.write(str(norm) + ", ")
    print("finished testing that")

# Testing various amounts of number of window size for unknown pages
with open("no_scores_window_size2.txt", "w") as f:
    print("Beginning to test various values for window size with hidden page scores")
    window_size = [1, 2, 5, 10]
    norms = []
    for amount in window_size:
        runs = []
        print("testing {} people for known values".format(amount))
        for i in tqdm(range(AVG_AMOUNT)):
            runs.append(rls_fit(num_known_people = 300, num_pages = 200, likes_per_person = 10, window = amount, num_iter = 500, hide_page_scores = True))
        f.write(str(amount) + ", ")
        runs = np.array(runs)
        norms.append(np.mean(runs))

    f.write("\n")
    for norm in norms:
        f.write(str(norm) + ", ")
    print("finished testing that")

# Testing various amounts of number of likes_per_person for unknown pages
with open("no_scores_likes_per_person2.txt", "w") as f:
    print("Beginning to test various values for number of likes per person with hidden page scores")
    num_likes = [1, 2, 5, 10, 20, 30 ]
    norms = []
    for amount in num_likes:
        runs = []
        print("testing {} people for known values".format(amount))
        for i in tqdm(range(AVG_AMOUNT)):
            runs.append(rls_fit(num_known_people = 300, num_pages = 200, likes_per_person = amount, window = 2, num_iter = 500, hide_page_scores = True))
        f.write(str(amount) + ", ")
        runs = np.array(runs)
        norms.append(np.mean(runs))

    f.write("\n")
    for norm in norms:
        f.write(str(norm) + ", ")
    print("finished testing that")
# Testing various amounts of number iterations for unknown pages
with open("no_scores_num_iterations2.txt", "w") as f:
    print("Beginning to test various values for number of iterations with hidden page scores")
    num_iter = [1, 10, 30, 100, 200, 500, 1000, 2000]
    norms = []
    for amount in num_iter:
        runs = []
        print("testing {} people for known values".format(amount))
        for i in tqdm(range(AVG_AMOUNT)):
            runs.append(rls_fit(num_known_people = 300, num_pages = 200, likes_per_person = 10, window = 2, num_iter = amount, hide_page_scores = True))
        f.write(str(amount) + ", ")
        runs = np.array(runs)
        norms.append(np.mean(runs))

    f.write("\n")
    for norm in norms:
        f.write(str(norm) + ", ")
    print("finished testing that")