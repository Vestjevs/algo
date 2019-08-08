import random

import numpy as np


def __left(i):
    return 2 * i + 1


def __right(i):
    return 2 * i + 2


def __parent(i):
    return i // 2


def __max_heapify(array, i, heap_size):
    l = __left(i)
    r = __right(i)
    if l < heap_size and array[l] > array[i]:
        largest = l
    else:
        largest = i
    if r < heap_size and array[r] > array[largest]:
        largest = r
    if largest != i:
        array[i], array[largest] = array[largest], array[i]
        __max_heapify(array, largest, heap_size)


def __min_heapify(array, i, heap_size):
    l = __left(i)
    r = __right(i)
    if l < heap_size and array[l] < array[i]:
        largest = l
    else:
        largest = i
    if r < heap_size and array[r] < array[largest]:
        largest = r
    if largest != i:
        array[i], array[largest] = array[largest], array[i]
        __min_heapify(array, largest, heap_size)


def insertion_sort(array):
    for i in range(1, len(array)):
        key = array[i]
        j = i - 1
        while j >= 0 and array[j] > key:
            array[j + 1] = array[j]
            j -= 1
        array[j + 1] = key


# heap sorting
def build_max_heap_1(array):
    for i in range((len(array) - 1) // 2, -1, -1):
        __max_heapify(array, i, len(array))


def __increase_key(array, index, value):
    array[index] = value
    while index > 0 and array[index] > array[__parent(index)]:
        array[index], array[__parent(index)] = array[__parent(index)], array[index]
        index = __parent(index)


def __max_heap_insert(array, value):
    array.append(-np.inf)
    __increase_key(array, len(array) - 1, value)


# not working
def build_max_heap_2(array):
    for i in range(0, len(array)):
        inserted = array[i]
        array.remove(inserted)
        __max_heap_insert(array, inserted)


def build_min_heap(array):
    for i in range((len(array) - 1) // 2, -1, -1):
        __min_heapify(array, i, len(array))


def heap_sort(array):
    build_max_heap_1(array)  # change to build_min_heap for decreasing sort
    heap_size = len(array)
    for i in range(len(array) - 1, 0, -1):
        array[0], array[i] = array[i], array[0]
        heap_size -= 1
        __max_heapify(array, 0, heap_size)  # change to __max_heapify for decreasing sort


class DaryHeap:
    def __init__(self, d):
        self.__d = d
        self.__heap = []
        self.__size = 0

    def __child(self, index, k):
        return index * self.__d + k

    def __parent(self, index):
        return index // self.__d


# quicksort


def quick_sort(array, p, r):
    """
    :average rating: O(nLgn)
    :param array: considered array
    :param p: left border
    :param r: right border
    """
    if p < r:
        q = __partition(array, p, r)
        quick_sort(array, p, q - 1)
        quick_sort(array, q + 1, r)


def __partition(array, p, r):
    """

    :param array: considered array, where occur changes the order of elements
    :param p: left border
    :param r: right border
    :return: middle index
    """
    pivot = array[r]
    i = p - 1
    for j in range(p, r):
        if array[j] <= pivot:
            i += 1
            array[i], array[j] = array[j], array[i]
    array[i + 1], array[r] = array[r], array[i + 1]
    return i + 1


# quick sort randomized version
def __randomized_partition(array, p, r):
    i = random.randint(p, r)
    array[i], array[r] = array[r], array[i]
    return __partition(array, p, r)


def randomized_quick_sort(array, p, r):
    if p < r:
        q = __randomized_partition(array, p, r)
        randomized_quick_sort(array, p, q - 1)
        randomized_quick_sort(array, q + 1, r)


# quick sort hoare partition, exist bug (unknown)
def __hoare_partition(array, p, r):
    pivot = array[p]
    i = p
    j = r
    while i < r and j > p:
        while array[j] > pivot:
            j -= 1
        while array[i] < pivot:
            i += 1
        if i < j and i < r:
            array[i], array[j] = array[j], array[i]
            print('+')
        else:
            return j


def quick_sort_hoare(array, p, r):
    if p < r:
        q = __hoare_partition(array, p, r)
        quick_sort_hoare(array, p, q - 1)
        quick_sort_hoare(array, q + 1, r)


# tail recursion
def quick_sort_tail_recursion(array, p, r):
    while p < r:
        q = __partition(array, p, r)
        quick_sort_tail_recursion(array, p, q - 1)
        p = q + 1


# stooge sort O(n^2,7)
def stooge_sort(array, i, j):
    if array[i] > array[j]:
        array[i], array[j] = array[j], array[i]
    if i + 1 >= j:
        return
    k = (j - i + 1) // 3
    stooge_sort(array, i, j - k)
    stooge_sort(array, i + k, j)
    stooge_sort(array, i, j - k)


# counting sort O(n + k)
def counting_sort(array, k):
    c = [0 for _ in range(k)]

    for j in range(len(array)):
        c[array[j]] += 1

    b = [-1 for _ in range(len(array))]

    for i in range(1, k):
        c[i] += c[i - 1]

    for j in range(len(array)):
        b[c[array[j]] - 1] = array[j]
        c[array[j]] -= 1

    return b


# radix sort O(d(n + k))
def __counting_sort_rd(array, k, index):
    c = [0 for _ in range(k)]

    for j in range(len(array)):
        c[get_digit(array[j], index)] += 1

    b = [-1 for _ in range(len(array))]

    for i in range(1, k):
        c[i] += c[i - 1]

    for j in range(len(array) - 1, -1, -1):
        b[c[get_digit(array[j], index)] - 1] = array[j]
        c[get_digit(array[j], index)] -= 1

    return b


def radix_sort(array, d):
    for i in range(d):
        array = __counting_sort_rd(array, 10, i)
    return array


def get_digit(number, k):
    return number // 10 ** k % 10


# bucket sort O(n)
def bucket_sort(array):
    aux = [[] for _ in range(len(array))]
    for i in range(len(array)):
        aux[int(np.floor(len(array) * array[i]))].append(array[i])

    result = []

    for j in range(len(array)):
        insertion_sort(aux[j])
        result += aux[j]

    array = result
    return array


# find minimum and maximum O(n) or 3 floor(n / 2) comparisons
def find_min_and_max(array):
    if len(array) % 2 == 0:
        start = 2
        if array[0] < array[1]:
            minimum = array[0]
            maximum = array[1]
        else:
            minimum = array[1]
            maximum = array[0]
    else:
        minimum = maximum = array[0]
        start = 1

    for i in range(start, len(array), 2):
        if array[i] < array[i + 1]:
            if array[i] < minimum:
                minimum = array[i]
            if array[i + 1] > maximum:
                maximum = array[i + 1]
        else:
            if array[i + 1] < minimum:
                minimum = array[i + 1]
            if array[i] > maximum:
                maximum = array[i]

    return minimum, maximum


def randomized_select(array, p, r, index):
    if index <= 0 or index >= len(array) + 1:
        raise Exception('Index out of bounds')
    if p == r or r < 0:
        return array[p]
    if p == r or p > len(array) - 1:
        return array[r]
    q = __randomized_partition(array, p, r)
    k = q - p + 1
    if index == k:
        return array[q]
    elif index < k:
        return randomized_select(array, p, q - 1, index)
    else:
        return randomized_select(array, q + 1, r, index - k)


def randomized_select_iter_vers(array, index):
    if index <= 0 or index >= len(array) + 1:
        raise Exception('Index out of bounds')
    p = 0
    r = len(array) - 1
    q = __randomized_partition(array, p, r)
    k = q - p + 1
    while k != index and r >= 0 and p < len(array):
        if index < k:
            r = q - 1
            q = __randomized_partition(array, p, r)
            k = q - p + 1
        else:
            p = q + 1
            q = __randomized_partition(array, p, r)
            k = q - p + 1
            index = index - k
    if index == k:
        return array[q]
    elif p == r or r < 0:
        return array[p]
    elif p == r or p > len(array) - 1:
        return array[r]


# for _ in range(10):
#     arr = [random.randint(1, 100) for i in range(15)]
#     insertion_sort(arr)
#     minimum1, maximum1 = find_min_and_max(arr)
#     if minimum1 != arr[0] and maximum1 != arr[-1]:
#         raise Exception('Awfully')
#     else:
#         print("found : {} ; selected : {}".format(maximum1, randomized_select(arr, 0, len(arr) - 1, 0)))
# for j in range(1, 101):
#     arr = [np.random.randint(1, 10 ** 3) for _ in range(100)]
#     choice = random.randint(1, 100)
#     if randomized_select(arr, 0, len(arr) - 1, choice) == sorted(arr)[choice - 1]:
#         print(j)
arr = [np.random.randint(1, 100) for _ in range(10)]
print(arr)
print(randomized_select_iter_vers(arr, 10))
print(sorted(arr))
