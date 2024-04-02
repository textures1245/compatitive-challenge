from abc import ABCMeta, abstractmethod
import sys
import queue
import sys
from random import randint
import re


#? Conditional Statements
def conditional_statements():
    N = int(input().strip())
    if N % 2 != 0:
        print("Weird")
    elif 2 <= N and N <= 5:
        print("Not Weird")
    elif 6 <= N and N <= 20:
        print("Weird")
    else:
        print("Not Weird")

#? Operators
def operators():
    meal_cost = float(input().strip())
    tip_percent = int(input().strip())
    tax_percent = int(input().strip())
    
    def solve(meal_cost, tip_percent, tax_percent):
        tip = float(meal_cost/100) * tip_percent
        tax = float(meal_cost/100) * tax_percent
        total_cost = meal_cost + tip + tax
        print(round(total_cost))
    
    solve(meal_cost, tip_percent, tax_percent)

#? Class vs. Instance
def class_vs_instance():
    class Person:
        def __init__(self, initialAge):
            self.age = initialAge
            if self.age < 0:
                self.age = 0
                print("Age is not valid, setting age to 0")
        
        def amIOld(self):
            if self.age < 13:
                print("You are young..")
            elif self.age >= 13 and self.age < 18:
                print("You are a teenager")
            else:
                print("You are old")
        
        def yearPasses(self):
            self.age += 1
        
    t = int(input())
    for i in range(0, t):
        age = int(input())         
        p = Person(age)  
        p.amIOld()
        for j in range(0, 3):
            p.yearPasses()       
        p.amIOld()
        print("")

#? Loops
def loops():
    n = int(input().strip())
    for i in range(1, 11):
        print("{0} x {1} = {2}".format(n, i, n*i))

#? Let's Review!
def lets_review():
    n = int(input())
    for i in range(n):
        s = input()
        first = s[::2]
        second = s[1::2]
        result = "{0} {1}".format(first, second)
        print(result)

#? Arrays
def arrays():
    n = int(input().strip())
    arr = list(map(int, input().rstrip().split()))
    print(" ".join(map(str, arr[::-1])))

#? Dictionaries and Maps
def dictionaries_and_maps():
    N = int(input())
    name_phone = [input().split() for _ in range(N)]
    phone = {n: p for n, p in name_phone}
    while True:
        try:
            name = input()
            if name in phone:
                print("{0}={1}".format(name, phone[name]))
            else:
                print("Not found")
        except:
            break

#? Recursion 3
def recursion_3():
    x = int(input())
    
    def factorial(x):
        if x == 1:
            return 1
        else:
            return (x * factorial(x-1))
    
    print(factorial(x))

#? Binary Numbers
def binary_numbers():
    n = int(input())
    print(len(max(bin(n)[2:].split('0'))))

#? 2D Arrays
def two_d_arrays():
    arr = []
    for _ in range(6):
        arr.append(list(map(int, input().rstrip().split())))
    res = []
    for x in range(0, 4):
        for y in range(0, 4):
            s = sum(arr[x][y:y+3]) + arr[x+1][y+1] + sum(arr[x+2][y:y+3])
            res.append(s)
    print(max(res))

#? Inheritance
def inheritance():
    class Person:
        def __init__(self, firstName, lastName, idNumber):
            self.firstName = firstName
            self.lastName = lastName
            self.idNumber = idNumber
        
        def printPerson(self):
            print("Name:", self.lastName + ",", self.firstName)
            print("ID:", self.idNumber)
    
    class Student(Person):
        def __init__(self, firstName, lastName, idNumber, scores):
            super().__init__(firstName, lastName, idNumber)
            self.scores = scores
        
        def calculate(self):
            average = sum(self.scores) / len(self.scores)
            if 90 <= average <= 100:
                return "O"
            elif 80 <= average < 90:
                return "E"
            elif 70 <= average < 80:
                return "A"
            elif 55 <= average < 70:
                return "P"
            elif 40 <= average < 55:
                return "D"
            else:
                return "T"
    
    line = input().split()
    firstName = line[0]
    lastName = line[1]
    idNum = line[2]
    numScores = int(input())
    scores = list(map(int, input().split()))
    s = Student(firstName, lastName, idNum, scores)
    s.printPerson()
    print("Grade:", s.calculate())

#? Abstract Classes
def abstract_classes():
    
    class Book(object, metaclass=ABCMeta):
        def __init__(self, title, author):
            self.title = title
            self.author = author
        
        @abstractmethod
        def display(self):
            pass
    
    class MyBook(Book):
        def __init__(self, title, author, price):
            super().__init__(title, author)
            self.price = price
        
        def display(self):
            print("Title:", self.title)
            print("Author:", self.author)
            print("Price:", self.price)
    
    title = input()
    author = input()
    price = int(input())
    new_novel = MyBook(title, author, price)
    new_novel.display()

#? Scope
def scope():
    class Difference:
        def __init__(self, a):
            self.__elements = a
            self.maximumDifference = 0
        
        def computeDifference(self):
            self.maximumDifference = abs(max(self.__elements) - min(self.__elements))
    
    _ = input()
    a = [int(e) for e in input().split()]
    d = Difference(a)
    d.computeDifference()
    print(d.maximumDifference)

#? Linked List
def linked_list():
    class Node:
        def __init__(self, data):
            self.data = data
            self.next = None
    
    class Solution:
        def display(self, head):
            current = head
            while current:
                print(current.data, end=' ')
                current = current.next
        
        def insert(self, head, data):
            if head is None:
                head = Node(data)
                self.tail = head
            else:
                node = Node(data)
                self.tail.next = node
                self.tail = node
            return head
    
    mylist = Solution()
    T = int(input())
    head = None
    for i in range(T):
        data = int(input())
        head = mylist.insert(head, data)
    mylist.display(head)

#? Exceptions - String to Integer
def exceptions_string_to_integer():
    def str2int(s):
        try:
            print(int(s))
        except ValueError:
            print("Bad String")
    
    S = input()
    str2int(S)

#? More Exceptions
def more_exceptions():
    class Calculator:
        def power(self, n, p):
            if n < 0 or p < 0:
                raise ValueError("n and p should be non-negative")
            else:
                return n ** p
    
    myCalculator = Calculator()
    T = int(input())
    for i in range(T):
        n, p = map(int, input().split())
        try:
            ans = myCalculator.power(n, p)
            print(ans)
        except Exception as e:
            print(e)

# Queues and Stacks

class Solution:
    def __init__(self):
        self._s = []

    def pushCharacter(self, s):
        return self._s.append(s)

    def enqueueCharacter(self, s):
        return self._s.append(s)

    def popCharacter(self):
        return self._s.pop()

    def dequeueCharacter(self):
        return self._s.pop(0)

def check_palindrome(s):
    obj = Solution()
    l = len(s)
    for i in range(l):
        obj.pushCharacter(s[i])
        obj.enqueueCharacter(s[i])

    isPalindrome = True
    for i in range(l // 2):
        if obj.popCharacter() != obj.dequeueCharacter():
            isPalindrome = False
            break

    if isPalindrome:
        print("The word, " + s + ", is a palindrome.")
    else:
        print("The word, " + s + ", is not a palindrome.")

# Interface
class AdvancedArithmetic(object):
    def divisorSum(n):
        raise NotImplementedError

class Calculator(AdvancedArithmetic):
    def divisorSum(self, n):
        self.n = n
        if n == 1:
            return 1
        else:
            factor_sum = 1 + n
            for i in range(2, n//2 + 1):
                if n % i == 0:
                    factor_sum += i
            return factor_sum

def calculate_divisor_sum():
    n = int(input())
    my_calculator = Calculator()
    s = my_calculator.divisorSum(n)
    print("I implemented: " + type(my_calculator).__bases__[0].__name__)
    print(s)

# Bubble Sorting
def bubbleSort(listed):
    has_swapped = True
    count = 0
    while(has_swapped):
        has_swapped = False
        for i in range(len(listed)):
            for j in range(len(listed) - 1):
                if listed[j] > listed[j + 1]:
                    count += 1
                    listed[i], listed[j + 1] = listed[j + 1], listed[i]
                    has_swapped = True
    return print("""Array is sorted in {0} swaps.\nFirst Element: {1}\nLast Element: {2}""".format(count,listed[0],listed[-1]))

def perform_bubble_sort():
    n = int(input().strip())
    a = list(map(int, input().rstrip().split()))
    bubbleSort(a)

# Binary Search Tree
class Node:
    def __init__(self, data):
        self.right = self.left = None
        self.data = data

class Solution:
    def insert(self, root, data):
        if root == None:
            return Node(data)
        else:
            if data <= root.data:
                cur = self.insert(root.left, data)
                root.left = cur
            else:
                cur = self.insert(root.right, data)
                root.right = cur
        return root

    def getHeight(self, root):
        if not root:
            return -1
        if not root.left and not root.right:
            return 0
        left_height = self.getHeight(root.left)
        right_height = self.getHeight(root.right)
        return max(left_height, right_height) + 1

def calculate_tree_height():
    T = int(input())
    myTree = Solution()
    root = None
    for i in range(T):
        data = int(input())
        root = myTree.insert(root, data)
    height = myTree.getHeight(root)
    print(height)

# BST Level-Order Traversal

class Node:
    def __init__(self, data):
        self.right = self.left = None
        self.data = data

class Solution:
    def insert(self, root, data):
        if root == None:
            return Node(data)
        else:
            if data <= root.data:
                cur = self.insert(root.left, data)
                root.left = cur
            else:
                cur = self.insert(root.right, data)
                root.right = cur
        return root

    def levelOrder(self, root):
        nodesToSearch = list()
        nodesTraversed = ""
        nodesToSearch.append(root)
        while len(nodesToSearch) > 0:
            node = nodesToSearch.pop(0)
            if node.left:
                nodesToSearch.append(node.left)
            if node.right:
                nodesToSearch.append(node.right)
            nodesTraversed += str(node.data) + ' '
        print(nodesTraversed)

def perform_level_order_traversal():
    T = int(input())
    myTree = Solution()
    root = None
    for i in range(T):
        data = int(input())
        root = myTree.insert(root, data)
    myTree.levelOrder(root)

# More Linked Lists
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class Solution:
    def insert(self, head, data):
        p = Node(data)
        if head == None:
            head = p
        elif head.next == None:
            head.next = p
        else:
            start = head
            while(start.next != None):
                start = start.next
            start.next = p
        return head

    def display(self, head):
        current = head
        while current:
            print(current.data, end=' ')
            current = current.next

    def removeDuplicates(self, head):
        if not head:
            return None
        current = head
        while current.next:
            if current.data == current.next.data:
                current.next = current.next.next
            else:
                current = current.next
        return head

def remove_duplicates_from_linked_list():
    mylist = Solution()
    T = int(input())
    head = None
    for i in range(T):
        data = int(input())
        head = mylist.insert(head, data)
    head = mylist.removeDuplicates(head)
    mylist.display(head)

# Running Time and Complexity
def is_prime(num):
    if num == 1:
        return False
    elif num == 2:
        return True
    elif num % 2 == 0:
        return False
    else:
        for i in range(3, int(num**(1/2)) + 1, 2):
            if num % i == 0:
                return False
        return True

def check_prime():
    T = int(input())
    for i in range(T):
        n = int(input())
        if is_prime(n):
            print("Prime")
        else:
            print("Not prime")

# Nested Logic
def calculate_fine():
    RD, RM, RY = [int(x) for x in input().split()]
    ED, EM, EY = [int(x) for x in input().split()]

    if (RY, RM, RD) <= (EY, EM, ED):
        fine = 0
        print(fine)
    elif (RY, RM) == (EY, EM):
        fine = 15 * (RD - ED)
        print(fine)
    elif RY == EY:
        fine = 500 * (RM - EM)
        print(fine)
    else:
        fine = 10000
        print(fine)

# Testing

def minimum_index(seq):
    if len(seq) == 0:
        raise ValueError("Cannot get the minimum value index from an empty sequence")
    min_idx = 0
    for i in range(1, len(seq)):
        if seq[i] < seq[min_idx]:
            min_idx = i
    return min_idx

class TestDataEmptyArray(object):
    @staticmethod
    def get_array():
        return []

class TestDataUniqueValues(object):
    data = set()
    while len(data) < 10:
        data.add(randint(0, 50))

    @staticmethod
    def get_array():
        data = TestDataUniqueValues.data
        return list(data)

    @staticmethod
    def get_expected_result():
        data = TestDataUniqueValues.get_array()
        return data.index(min(data))

class TestDataExactlyTwoDifferentMinimums(object):
    data = set()
    while len(data) < 9:
        data.add(randint(0, 50))
    newData = list(data)
    newData.append(min(newData))

    @staticmethod
    def get_array():
        data = TestDataExactlyTwoDifferentMinimums.newData
        return data

    @staticmethod
    def get_expected_result():
        data = TestDataExactlyTwoDifferentMinimums.get_array()
        return data.index(min(data))

def test_with_empty_array():
    try:
        seq = TestDataEmptyArray.get_array()
        result = minimum_index(seq)
    except ValueError as e:
        pass
    else:
        assert False

def test_with_unique_values():
    seq = TestDataUniqueValues.get_array()
    assert len(seq) >= 2
    assert len(list(set(seq))) == len(seq)
    expected_result = TestDataUniqueValues.get_expected_result()
    result = minimum_index(seq)
    assert result == expected_result

def test_with_exactly_two_different_minimums():
    seq = TestDataExactlyTwoDifferentMinimums.get_array()
    assert len(seq) >= 2
    tmp = sorted(seq)
    assert tmp[0] == tmp[1] and (len(tmp) == 2 or tmp[1] < tmp[2])
    expected_result = TestDataExactlyTwoDifferentMinimums.get_expected_result()
    result = minimum_index(seq)
    assert result == expected_result

def run_tests():
    test_with_empty_array()
    test_with_unique_values()
    test_with_exactly_two_different_minimums
    print("OK")

# RegEx, Patterns, and Intro to Databases

def filter_emails():
    N = int(input().strip())
    regexPattern = '@gmail\.com$'
    nameList = []
    for N_itr in range(N):
        first_multiple_input = input().rstrip().split()
        firstName = first_multiple_input[0]
        emailID = first_multiple_input[1]
        if re.search(regexPattern, emailID):
            nameList.append(firstName)
    print(*sorted(nameList), sep="\n")

# Bitwise AND
def bitwiseAnd(N, K):
    max_bitwise = 0
    for i in range(1, N + 1):
        for j in range(1, i):
            bitwise = i & j
            if max_bitwise < bitwise < K:
                max_bitwise = bitwise
                if max_bitwise == K - 1:
                    return max_bitwise
    return max_bitwise

def perform_bitwise_and():
    t = int(input().strip())
    listed = []
    for t_itr in range(t):
        first_multiple_input = input().rstrip().split()
        count = int(first_multiple_input[0])
        lim = int(first_multiple_input[1])
        res = bitwiseAnd(count, lim)
        listed.append(res)
    print(*listed, sep="\n")


def main(run_number):

    def run_task(run_number):
        switcher = {
            1: conditional_statements,
            2: operators,
            3: class_vs_instance,
            4: loops,
            5: lets_review,
            6: arrays,
            7: dictionaries_and_maps,
            8: recursion_3,
            9: binary_numbers,
            10: two_d_arrays,
            11: inheritance,
            12: abstract_classes,
            13: scope,
            14: linked_list,
            15: exceptions_string_to_integer,
            16: more_exceptions,
            17: perform_bubble_sort,
            18: calculate_tree_height,
            19: perform_level_order_traversal,
            20: remove_duplicates_from_linked_list,
            21: check_prime,
            22: calculate_fine,
            23: run_tests,
            24: filter_emails,
            25: perform_bitwise_and
        }
        func = switcher.get(run_number, lambda: print("Invalid run number."))
        func()

    run_task(run_number)

if __name__ == "__main__":
    run_number = int(input("Enter the task number: "))
    main(run_number)

# ----------------------------End Project-----------------------------------#