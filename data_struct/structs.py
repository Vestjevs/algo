# data structure
import numpy as np


class YoungTableauMax:
    def __init__(self, m, n):
        self.__tableau = [[-np.inf for _ in range(n)] for _ in range(m)]
        self.__size = 0
        self.__row = 0
        self.__col = 0
        self.__m = m
        self.__n = n

    def show(self):
        for row in self.__tableau:
            print(row)

    def size(self):
        return self.__size

    def insert(self, value):
        if self.__size == self.__m * self.__n:
            raise Exception(f'Tableau is full')
        self.__tableau[self.__row][self.__col] = value

        self.__lift_elem(self.__row, self.__col)
        if self.__col == self.__n - 1:
            self.__row += 1
            self.__col = 0
        else:
            self.__col += 1
        self.__size += 1

    def extract_maximum(self):
        extracted = self.__tableau[0][0]
        if self.__col == 0 and self.__row != 0:
            self.__tableau[0][0], self.__tableau[self.__row - 1][-1] = \
                self.__tableau[self.__row - 1][-1], self.__tableau[0][0]
            self.__tableau[self.__row - 1][-1] = -np.inf
        else:
            self.__tableau[0][0], self.__tableau[self.__row][self.__col - 1] = \
                self.__tableau[self.__row][self.__col - 1], self.__tableau[0][0]
            self.__tableau[self.__row][self.__col - 1] = -np.inf

        self.__size -= 1
        self.__descent_elem(0, 0)
        if self.__col == 0:
            self.__col = self.__n - 1
            self.__row -= 1
        else:
            self.__col -= 1
        return extracted

    def __lift_elem(self, i, j):
        self.__check(i, j)
        while (i > 0 and j > 0) and (self.__tableau[i][j] > self.__tableau[i - 1][j] or
                                     self.__tableau[i][j] > self.__tableau[i][j - 1]):
            if self.__tableau[i][j] > self.__tableau[i - 1][j] == np.minimum(self.__tableau[i - 1][j],
                                                                             self.__tableau[i][j - 1]):
                self.__tableau[i][j], self.__tableau[i - 1][j] = self.__tableau[i - 1][j], self.__tableau[i][j]
                i -= 1
            elif self.__tableau[i][j] > self.__tableau[i][j - 1] == np.minimum(self.__tableau[i - 1][j],
                                                                               self.__tableau[i][j - 1]):
                self.__tableau[i][j], self.__tableau[i][j - 1] = self.__tableau[i][j - 1], self.__tableau[i][j]
                j -= 1
        while i > 0 and self.__tableau[i][j] > self.__tableau[i - 1][j]:
            self.__tableau[i][j], self.__tableau[i - 1][j] = self.__tableau[i - 1][j], self.__tableau[i][j]
            i -= 1
        while j > 0 and self.__tableau[i][j] > self.__tableau[i][j - 1]:
            self.__tableau[i][j], self.__tableau[i][j - 1] = self.__tableau[i][j - 1], self.__tableau[i][j]
            j -= 1

    def __descent_elem(self, i, j):
        self.__check(i, j)
        while (i != self.__m - 1 and j != self.__n - 1) and (self.__tableau[i][j] < self.__tableau[i + 1][j] or
                                                             self.__tableau[i][j] < self.__tableau[i][
                                                                 j + 1]):
            if self.__tableau[i][j] < self.__tableau[i + 1][j] == np.maximum(
                    self.__tableau[i][j + 1], self.__tableau[i + 1][j]):
                self.__tableau[i][j], self.__tableau[i + 1][j] = self.__tableau[i + 1][j], self.__tableau[i][j]
                i += 1
            elif self.__tableau[i][j] < self.__tableau[i][j + 1] == np.maximum(self.__tableau[i][j + 1],
                                                                               self.__tableau[i + 1][j]):
                self.__tableau[i][j], self.__tableau[i][j + 1] = self.__tableau[i][j + 1], self.__tableau[i][j]
                j += 1
        while i != self.__m - 1 and self.__tableau[i][j] < self.__tableau[i + 1][j]:
            self.__tableau[i][j], self.__tableau[i + 1][j] = self.__tableau[i + 1][j], self.__tableau[i][j]
            i += 1
        while j != self.__n - 1 and self.__tableau[i][j] < self.__tableau[i][j + 1]:
            self.__tableau[i][j], self.__tableau[i][j + 1] = self.__tableau[i][j + 1], self.__tableau[i][j]
            j += 1

    def __check(self, i, j):
        if i >= self.__m or i < 0 or j >= self.__n or j < 0:
            raise IndexError


# data structure
class YoungTableauMin:
    def __init__(self, m, n):
        self.__tableau = [[np.inf for _ in range(n)] for _ in range(m)]
        self.__row = 0
        self.__col = 0
        self.__m = m
        self.__n = n
        self.__size = 0

    def show(self):
        for row in self.__tableau:
            print(row)

    def size(self):
        return self.__size

    def insert(self, value):
        if self.__size == self.__m * self.__n:
            raise Exception(f'Tableau is full')
        self.__tableau[self.__row][self.__col] = value

        self.__lift_elem(self.__row, self.__col)
        if self.__col == self.__n - 1:
            self.__row += 1
            self.__col = 0
        else:
            self.__col += 1
        self.__size += 1

    def extract_minimum(self):
        if self.__size == 0:
            raise Exception('Tableau is empty')
        extracted = self.__tableau[0][0]
        if self.__col == 0 and self.__row != 0:
            self.__tableau[0][0], self.__tableau[self.__row - 1][-1] = \
                self.__tableau[self.__row - 1][-1], self.__tableau[0][0]
            self.__tableau[self.__row - 1][-1] = np.inf
        else:
            self.__tableau[0][0], self.__tableau[self.__row][self.__col - 1] = \
                self.__tableau[self.__row][self.__col - 1], self.__tableau[0][0]
            self.__tableau[self.__row][self.__col - 1] = np.inf

        self.__size -= 1
        self.__descent_elem(0, 0)
        if self.__col == 0:
            self.__col = self.__n - 1
            self.__row -= 1
        else:
            self.__col -= 1
        return extracted

    def __lift_elem(self, i, j):
        self.__check(i, j)
        while (i > 0 and j > 0) and (self.__tableau[i][j] < self.__tableau[i - 1][j] or
                                     self.__tableau[i][j] < self.__tableau[i][j - 1]):
            if self.__tableau[i][j] < self.__tableau[i - 1][j] == np.maximum(self.__tableau[i][j - 1],
                                                                             self.__tableau[i - 1][j]):
                self.__tableau[i][j], self.__tableau[i - 1][j] = self.__tableau[i - 1][j], self.__tableau[i][j]
                i -= 1
            elif self.__tableau[i][j] < self.__tableau[i][j - 1] == np.maximum(self.__tableau[i][j - 1],
                                                                               self.__tableau[i - 1][j]):
                self.__tableau[i][j], self.__tableau[i][j - 1] = self.__tableau[i][j - 1], self.__tableau[i][j]
                j -= 1
        while i > 0 and self.__tableau[i][j] < self.__tableau[i - 1][j]:
            self.__tableau[i][j], self.__tableau[i - 1][j] = self.__tableau[i - 1][j], self.__tableau[i][j]
            i -= 1
        while j > 0 and self.__tableau[i][j] < self.__tableau[i][j - 1]:
            self.__tableau[i][j], self.__tableau[i][j - 1] = self.__tableau[i][j - 1], self.__tableau[i][j]
            j -= 1

    def __descent_elem(self, i, j):
        self.__check(i, j)
        while (i != self.__m - 1 and j != self.__n - 1) and \
                (self.__tableau[i][j] > self.__tableau[i + 1][j] or self.__tableau[i][j] > self.__tableau[i][
                    j + 1]):
            if np.minimum(self.__tableau[i][j + 1], self.__tableau[i + 1][j]) == self.__tableau[i + 1][j] and \
                    self.__tableau[i][j] > self.__tableau[i + 1][j]:
                self.__tableau[i][j], self.__tableau[i + 1][j] = self.__tableau[i + 1][j], self.__tableau[i][j]
                i += 1
            elif self.__tableau[i][j] > self.__tableau[i][j + 1]:
                self.__tableau[i][j], self.__tableau[i][j + 1] = self.__tableau[i][j + 1], self.__tableau[i][j]
                j += 1
            elif self.__tableau[i][j] > self.__tableau[i - 1][j]:
                self.__tableau[i][j], self.__tableau[i + 1][j] = self.__tableau[i + 1][j], self.__tableau[i][j]
                i += 1
            elif self.__tableau[i][j] > self.__tableau[i][j + 1]:
                self.__tableau[i][j], self.__tableau[i][j + 1] = self.__tableau[i][j + 1], self.__tableau[i][j]
                j += 1

    def __check(self, i, j):
        if i >= self.__m or i < 0 or j >= self.__n or j < 0:
            raise IndexError


# data structure
class PriorityQueueMax:
    def __init__(self):
        self.__queue = []
        self.__heap_size = 0

    def __len__(self):
        return self.__heap_size

    def heap_maximum(self):
        return self.__queue[0]

    def extract_max(self):
        if self.__heap_size == 0:
            raise Exception(f'priority queue is empty')
        maximum = self.__queue[0]
        self.__queue[0], self.__queue[self.__heap_size - 1] = self.__queue[self.__heap_size - 1], self.__queue[0]
        self.__queue.remove(maximum)
        self.__heap_size -= 1
        self.__max_heapify(0, self.__heap_size)
        return maximum

    def __max_heapify(self, i, heap_size):
        left = self.__left(i)
        right = self.__right(i)
        if left < heap_size and self.__queue[left] > self.__queue[i]:
            largest = left
        else:
            largest = i
        if right < heap_size and self.__queue[right] > self.__queue[largest]:
            largest = right
        if largest != i:
            self.__queue[i], self.__queue[largest] = self.__queue[largest], self.__queue[i]
            self.__max_heapify(largest, heap_size)

    def increase_key(self, index, key):
        if key < self.__queue[index]:
            raise Exception(f'the new key is less than current')
        self.__queue[index] = key
        while index > 0 and self.__queue[self.__parent(index)] < self.__queue[index]:
            self.__queue[index], self.__queue[self.__parent(index)] = self.__queue[self.__parent(index)], self.__queue[
                index]
            index = self.__parent(index)

    def insert(self, key):
        self.__heap_size += 1
        self.__queue.append(-np.inf)
        self.increase_key(self.__heap_size - 1, key)

    def delete(self, i):
        if i < 0 or i >= self.__heap_size:
            raise IndexError
        deleted = self.__queue[i]
        self.__queue[i], self.__queue[self.__heap_size - 1] = self.__queue[self.__heap_size - 1], self.__queue[i]
        self.__queue.remove(deleted)
        self.__heap_size -= 1
        self.__max_heapify(i, self.__heap_size)

    def merge(self, pq):
        while len(pq) > 0:
            extracted = pq.extract_max()
            self.insert(extracted)
            print(extracted)

    @staticmethod
    def __left(index):
        return 2 * index + 1

    @staticmethod
    def __right(index):
        return 2 * index + 2

    @staticmethod
    def __parent(index):
        return index // 2

    def show(self):
        print(self.__queue)


# data structure
class PriorityQueueMin:
    def __init__(self):
        self.__queue = []
        self.__heap_size = 0

    def heap_minimum(self):
        return self.__queue[0]

    def extract_minimum(self):
        if self.__queue == 0:
            raise Exception(f'priority queue is empty')
        minimum = self.__queue[0]
        self.__queue[0], self.__queue[self.__heap_size - 1] = self.__queue[self.__heap_size - 1], self.__queue[0]
        self.__queue.remove(minimum)
        self.__heap_size -= 1
        self.__min_heapify(0, self.__heap_size)
        return minimum

    def __min_heapify(self, i, heap_size):
        l = self.__left(i)
        r = self.__right(i)
        if l < heap_size and self.__queue[l] < self.__queue[i]:
            largest = l
        else:
            largest = i
        if r < heap_size and self.__queue[r] < self.__queue[largest]:
            largest = r
        if largest != i:
            self.__queue[i], self.__queue[largest] = self.__queue[largest], self.__queue[i]
            self.__min_heapify(largest, heap_size)

    def decrease_key(self, index, key):
        if key > self.__queue[index]:
            raise Exception(f'the new key is greater than current')
        self.__queue[index] = key
        while index > 0 and self.__queue[self.__parent(index)] > self.__queue[index]:
            self.__queue[self.__parent(index)], self.__queue[index] = self.__queue[index], self.__queue[
                self.__parent(index)]
            index = self.__parent(index)

    def insert(self, key):
        self.__heap_size += 1
        self.__queue.append(np.inf)
        self.decrease_key(self.__heap_size - 1, key)

    def delete(self, i):
        if i < 0 or i >= self.__heap_size:
            raise IndexError
        deleted = self.__queue[i]
        self.__queue[i], self.__queue[self.__heap_size - 1] = self.__queue[self.__heap_size - 1], self.__queue[i]
        self.__queue.remove(deleted)
        self.__heap_size -= 1
        self.__min_heapify(i, self.__heap_size)

    def merge(self, pq):
        while len(pq) > 0:
            self.insert(pq.extract_min())

    @staticmethod
    def __left(index):
        return 2 * index + 1

    @staticmethod
    def __right(index):
        return 2 * index + 2

    @staticmethod
    def __parent(index):
        return index // 2

    def show(self):
        print(self.__queue)


class Stack:
    def __init__(self, capacity):
        self.__capacity = capacity
        self.__container = []
        self.__top = 0

    def is_empty(self):
        return self.__top == 0

    def push(self, element):
        if self.__top == self.__capacity:
            raise OverflowError
        else:
            self.__container.append(element)
            self.__top += 1

    def pop(self):
        if self.__top == 0:
            raise Exception('EmptyflowError')
        else:
            self.__container.pop(self.__top - 1)
            self.__top -= 1

    def pick(self):
        if self.__top == 0:
            raise Exception('EmptyflowError')
        else:
            return self.__container[self.__top]

    def __len__(self):
        return len(self.__container)

    def __str__(self):
        return '\n'.join(str(self.__container[j]) for j in range(len(self.__container) - 1, -1, -1))


class Queue:
    def __init__(self, capacity):
        self.__capacity = capacity
        self.__tail = 0
        self.__head = 0
        self.__container = []

    def is_empty(self):
        return self.__tail == 0

    def enqueue(self, elem):
        if self.__tail == self.__capacity:
            raise OverflowError
        else:
            self.__container.append(elem)
            self.__tail += 1

    def dequeue(self):
        if self.__tail == 0:
            raise Exception('EmptyflowError')
        else:
            self.__container.pop(0)

    def __len__(self):
        return len(self.__container)

    def __str__(self):
        return '\n'.join(str(elem) for elem in self.__container)


queue = Queue(5)
queue.enqueue(1)
queue.enqueue(10)
queue.enqueue(2)
queue.dequeue()
print(queue)
