import gen_data as gd

# generate p pages

p = 10
pages = gd.gen_ocean_score(p)

# generate x people who's scores are known
x = 10
known_people = gd.gen_ocean_score(x)
# TODO should we do the same thing with other functions allowing it to return a multidimensional array?
known_likes = gd.gen_page_likes(known_people, pages, 5)
# known_likes = []
# for person in known_people:
#     known_likes.append(gd.gen_page_likes(person, pages, 5))

# generate y unknown people
y = 100
unknown_people = gd.gen_ocean_score(y)
unknown_likes = gd.gen_page_likes(unknown_people, pages, 5)
# unknown_likes = []
# for person in unknown_people:
#     unknown_likes.append(gd.gen_page_likes(person, pages, 5))

print("pages")
print(pages)
print("known_people")
print(known_people)
print("known_likes")
print(known_likes)