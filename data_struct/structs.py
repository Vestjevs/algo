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
            removed = self.__container[self.__top - 1]
            self.__container.pop(self.__top - 1)
            self.__top -= 1
        return removed

    def pick(self):
        if self.__top == 0:
            raise Exception('EmptyflowError')
        else:
            return self.__container[self.__top - 1]

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

    def enqueue(self, element):
        if self.__tail == self.__capacity:
            raise OverflowError
        else:
            self.__container.append(element)
            self.__tail += 1

    def dequeue(self):
        if self.__tail == 0:
            raise Exception('EmptyflowError')
        else:
            removed = self.__container[0]
            self.__container.pop(0)
        return removed

    def __len__(self):
        return len(self.__container)

    def __str__(self):
        return '\n'.join(str(elem) for elem in self.__container)


class QueueOnTwoStacks:
    def __init__(self, capacity):
        self.__capacity = capacity
        self.__stack = Stack(capacity)
        self.__aux_stack = Stack(capacity)

    def is_empty(self):
        return self.__stack.is_empty()

    def enqueue(self, element):
        self.__stack.push(element)

    def dequeue(self):
        size = len(self.__stack)
        for _ in range(size):
            self.__aux_stack.push(self.__stack.pop())
        removed = self.__aux_stack.pop()

        for _ in range(size - 1):
            self.__stack.push(self.__aux_stack.pop())
        return removed


class StackOnTwoQueues:
    def __init__(self, capacity):
        self.__capacity = capacity
        self.__queue = Queue(capacity)
        self.__aux_queue = Queue(capacity)

    def is_empty(self):
        return self.__queue.is_empty()

    def push(self, element):
        self.__queue.enqueue(element)

    def pop(self):
        size = len(self.__queue)
        for _ in range(size):
            self.__aux_queue.enqueue(self.__queue.dequeue())
        removed = self.__aux_queue.dequeue()

        for _ in range(size - 1):
            self.__queue.enqueue(self.__aux_queue.dequeue())
        return removed


class Node:
    def __init__(self, prev, key, next):
        self.__key = key
        self.__prev = prev
        self.__next = next

    def get_key(self):
        return self.__key

    def get_prev(self):
        return self.__prev

    def set_prev(self, prev):
        self.__prev = prev

    def get_next(self):
        return self.__next

    def set_next(self, next):
        self.__next = next


class LinkedList:
    def __init__(self, capacity):
        self.__capacity = capacity
        self.__head = None
        self.__tail = None
        self.__size = 0

    def is_empty(self):
        return self.__size == 0

    # add element to the end
    def insert(self, key):
        if self.__size == self.__capacity:
            raise OverflowError
        else:
            if self.is_empty():
                node = Node(None, key, None)
                self.__head = node
                self.__tail = node
                self.__size += 1
            else:
                node = Node(self.__tail, key, None)
                self.__tail.set_next(node)
                self.__tail = node
                self.__size += 1

    # add element to head
    def put(self, key):
        if self.__size == self.__capacity:
            raise OverflowError
        else:
            if self.is_empty():
                node = Node(None, key, None)
                self.__head = node
                self.__tail = node
                self.__size += 1
            else:
                node = Node(None, key, self.__head)
                self.__head.set_prev(node)
                self.__head = node
                self.__size += 1

    def __search(self, key):
        if self.is_empty():
            raise Exception('I\'m empty')
        else:
            current = self.__head
            while current is not None:
                if current.get_key() == key:
                    return current
                current = current.get_next()
        return None

    def is_contain(self, key):
        return self.__search(key) is not None

    def delete(self, key):
        if self.is_empty():
            raise Exception('Im empty')
        else:
            removed = self.__search(key)
            if removed is not None and removed.get_next() is not None and removed.get_prev() is not None:
                removed.get_prev().set_next(removed.get_next())
                removed.get_next().set_prev(removed.get_prev())
                self.__size -= 1
            elif removed is not None and removed.get_next() is None:
                removed.get_prev().set_next(None)
                self.__tail = removed.get_prev()
                self.__size -= 1
            elif removed is not None and removed.get_prev() is None:
                self.__head = removed.get_next()
                self.__size -= 1
            elif removed is not None and removed.get_prev() is None and removed.get_next() is None:
                self.__head = None
                self.__size -= 1

    def show(self):
        current = self.__head
        sample = ''
        while current is not None:
            sample += str(current.get_key()) + '\n'
            current = current.get_next()
        return sample


class TreeNode:
    def __init__(self, key=None, left=None, right=None, ancestor=None):
        self.__key = key
        self.__left = left
        self.__right = right
        self.__ancestor = ancestor

    @staticmethod
    def to_sstring(node, prefix):
        if node is None:
            return ""
        else:
            indent = prefix + '\t'
            return '{}Node -> {} \n{}{}'.format(indent, node.get_key(), TreeNode.to_sstring(node.get_left(), indent),
                                                TreeNode.to_sstring(node.get_right(), indent))

    def __str__(self):
        return 'Node key -> {}'.format(self.__key)

    def get_key(self):
        return self.__key

    def set_key(self, key):
        self.__key = key

    def get_right(self):
        return self.__right

    def get_left(self):
        return self.__left

    def set_left(self, node):
        self.__left = node

    def set_right(self, node):
        self.__right = node

    def set_ancestor(self, node):
        self.__ancestor = node

    def get_ancestor(self):
        return self.__ancestor

    def set_key(self, key):
        self.__key = key


class BST:
    def __init__(self):
        self.__root = None
        self.__size = 0

    def is_empty(self):
        return self.__size == 0

    def __len__(self):
        return self.__size

    def minimum(self):
        return self.__minimum(self.__root)

    def __minimum(self, node):
        while node.get_left() is not None:
            node = node.get_left()
        return node

    def maximum(self):
        node = self.__root
        while node.get_right() is not None:
            node = node.get_right()
        return node

    def __maximum(self, node):
        if node.get_right() is None:
            return node
        else:
            return self.__maximum(node.get_right())

    def rec_minimum(self):
        return self.__recursive_minimum(self.__root)

    def __recursive_minimum(self, node):
        if node.get_left() is None:
            return node
        else:
            return self.__recursive_minimum(node)

    def __search(self, key):
        node = self.__root
        while node is not None and key != node.get_key():
            if key < node.get_key():
                node = node.get_left()
            else:
                node = node.get_right()
        return node

    def is_contain(self, key):
        return self.__search(key) is not None

    def successor(self, key):
        node = self.__search(key)
        if node.get_right() is not None:
            return self.__minimum(node)

        y = node.get_ancestor()
        while y is not None and node == y.get_right():
            node = y
            y = y.get_ancestor()

        return y

    def insert(self, key):
        y = None
        x = self.__root
        node = TreeNode(key)
        while x is not None:
            y = x
            if node.get_key() < x.get_key():
                x = x.get_left()
            else:
                x = x.get_right()
        node.set_ancestor(y)
        if y is None:
            self.__root = node
            self.__size += 1
        else:
            if node.get_key() < y.get_key():
                y.set_left(node)
                self.__size += 1
            else:
                y.set_right(node)
                self.__size += 1

    def rec_insert(self, key):
        self.__root = self.__recursive_insert(self.__root, key)

    def delete(self, key):
        node = self.__search(key)
        if node is None:
            raise Exception('key does not contained here')
        else:
            if node.get_right() is None and node.set_left() is None:
                node.set_ancestor(None)
                self.__size -= 1
            else:
                # minimum from node.right

                aux1 = self.__minimum(node.get_right())
                # maximum from node.left

                aux2 = self.__maximum(node.get_left())
                if aux1 is not None:
                    node.set_key(aux1.get_key())
                    aux1.get_ancestor().set_left(aux1.get_right())
                    self.__size -= 1
                elif aux2 is not None:
                    node.set_key(aux2.get_key())
                    aux2.get_ancestor().set_right(aux2.get_left())
                    self.__size -= 1

    def to_sstring(self):
        return TreeNode.to_sstring(self.__root, " ")

    def __recursive_insert(self, node, key):
        if node is None:
            node = TreeNode(key)
            self.__size += 1
        else:
            if key < node.get_key():
                node.set_left(self.__recursive_insert(node.get_left(), key))
            else:
                node.set_right(self.__recursive_insert(node.get_right(), key))
        return node

    def inorder_walk(self):
        self.__inorder_walk(self.__root)

    def __inorder_walk(self, node):
        if node is not None:
            self.__inorder_walk(node.get_left())
            print(node)
            self.__inorder_walk(node.get_right())


bst = BST()
sample = ' '
#
# while len(bst) != 6:
#     value = input('Insert value: ')
#     sample += value + ' '
#     bst.rec_insert(int(value))
#     print(bst.to_sstring())
#
# print(len(bst))
#
# i = 0
# while i != 6:
#     i += 1
#     choice = int(input('Insert your value: '))
#     print(bst.is_contain(choice))
#     print(sample)
#
#     arr = sample.split(' ')
#     arr.remove(str(choice))
#     sample = ' '.join(arr)
#     bst.delete(choice)
#     print(bst.to_sstring())
bst.insert(50)
print(bst.to_sstring())
bst.insert(30)
bst.insert(46)
bst.insert(78)
bst.insert(18)
print(bst.to_sstring())
bst.insert(105)
bst.insert(115)
bst.insert(155)
bst.insert(55)
bst.insert(65)
print(bst.to_sstring())
bst.delete(50)
print(bst.to_sstring())
bst.inorder_walk()
