from itertools import product

# list1 = ['a', 'b', 'c' , 'r']
# list2 = ['d', 'e', 'f' ,'9']

# pairs_iterator = product(list1, list2)
# pairs_list = list(pairs_iterator)
# print(pairs_list)


def get_pairwise_list(list1 , list2):
    pairs_iterator = product(list1 , list2)
    pairs_list = list(pairs_iterator)
    return pairs_list