from math import prod
import re
#? fucntion write
# def is_leap(year):
#     leap = False 
#     leap = year % 4 == 0 or (year % 400 == 0 and year % 100 != 0)
#     return leap
# year = int(input())
# print(is_leap(year))

#? Print Function
# n = int(input())
# print(*range(1,n+1), sep='')

#? List comprehensions
# x = int(input())
# y = int(input())
# z = int(input())
# n = int(input())
# result = []
# for i in range(0,x+1):
#     for e in range(0,y+1):
#         for a in range(0,z+1):
#             if (i + e + a) != n:
#               result.append([i, e, a]) 
# print(result)

#? Find the Runner-Up Score
# n = int(input())
# arr = map(int, input().split())
# print (sorted(set(arr))[-2])

#? Nested Lists
# n = int(input())
# student_marks = {}
# for _ in range(n):
#     name, *line = input().split()
#     scores = list(map(float, line))
#     student_marks[name] = scores
# query_name = input()
# if query_name in student_marks:
#   x = ((float(student_marks[query_name][0]) + float(student_marks[query_name][1]) + float(student_marks[query_name][2])) / 3)
# print('%.02f' %(x))

#? Regular expression
# testString = "This is a string with an email@address.com embedded in it."
# matchObj = re.search(r"(?:\b)([^\s]*\@[^\s]*\.[a-zA-Z]{2,4})(?:\b)",testString)
# print (matchObj.group(1))

#? List
# N = int(input())
# result = []
# for i in range(0,N):
#     ip = input().split()
#     if ip[0] == "print":
#         print(result)
#     elif ip[0] == "insert":
#         result.insert(int(ip[1]),int(ip[2]))
#     elif ip[0] == "remove":
#         result.remove(int(ip[1]))
#     elif ip[0] == "pop":
#         result.pop()
#     elif ip[0] == "append":
#         result.append(int(ip[1]))
#     elif ip[0] == "sort":
#         result.sort()
#     else:
#         result.reverse()

#? Tuples
# n = int(input())
# integer_list = map(int, input().split())
# t = tuple(integer_list)
# print(hash(t))

#? Nested list
# # declare input variable to list and set
# l = []
# second_lowest_names = []
# scores = set()
# # input integer in list and set
# for _ in range(int(input())):
#     name = input()
#     score = float(input())
#     l.append([name, score])
#     scores.add(score)
# # Defined second lowest output 
# second_lowest = sorted(scores)[1]
# # Defind name's second lowest
# for name, score in l:
#     if score == second_lowest:
#         second_lowest_names.append(name)
# # return output with sorted values(name)
# for name in sorted(second_lowest_names):
#     print(name, end='\n')

#? String Split and Join
# def split_and_join(line):
#     x = line.split(" ")
#     x = "-".join(x)
#     return x
# line = input()
# result = split_and_join(line)
# print(result)

#? name
# def print_full_name(first, last):
#     print('Hello', (first,last), '! You just delved into python')

# first_name = input()
# last_name = input()
# print_full_name(first_name, last_name)

#? Mutations
# def mutate_string(string, position, character):
#     s_new = string[:position] + character + string[position+1:]
#     return s_new

# s = input()
# i, c = input().split()
# s_new = mutate_string(s, int(i), c)
# print(s_new)

#? Mutation
# def count_substring(string, sub_string):
#     count = 0
#     i = string.find(sub_string) 
#     while i != -1: # use != -1 to reverse from not found(-1) to found
#         count += 1
#         i = string.find(sub_string, i+1)
#     return count

# string = input().strip()
# sub_string = input().strip()    
# count = count_substring(string, sub_string)
# print(count)

#? String Validators
# s = input()
# print(any([char.isalnum() for char in s]))
# print(any([char.isalpha() for char in s]))
# print(any([char.isdigit() for char in s]))
# print(any([char.islower() for char in s]))
# print(any([char.isupper() for char in s]))

#? Text alignment
#* Heart icon
# import math
# c='♥'
# width = 40

# print ((c*2).center(width//2)*2)

# for i in range(1,width//10+1):
#     print (((c*int(math.sin(math.radians(i*width//2))*width//4)).rjust(width//4)+
#            (c*int(math.sin(math.radians(i*width//2))*width//4)).ljust(width//4))*2)

# for i in range(width//4,0,-1):
#     print ((c*i*4).center(width))
# print ((c*2).center(width))
#* Arrow icon
# thickness = int(input()) #This must be an odd number
# c = 'H'
# # Top Cone
# for i in range(thickness):
#     print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))
# #Top Pillars
# for i in range(thickness+1):
#     print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))
# #Middle Belt
# for i in range((thickness+1)//2):
#     print((c*thickness*5).center(thickness*6))    
# #Bottom Pillars
# for i in range(thickness+1):
#     print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    
# #Bottom Cone
# for i in range(thickness):
#     print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

#? Text Wrap
# import textwrap

# def wrap(string, max_width):
#     warp = textwrap.fill(string,max_width)
#     return warp

# string = input()
# max_width = int(input())
# result = wrap(string, max_width)
# print(result)

#? Pyramid defind
# n, m = map(int,input().split())
# c = '.|.'
# for i in range(n):
#     print((c*i).rjust(n+m)+c+(c*i).ljust(n+m))

#? Designer Door Mat
# n, m = map(int,input().split())
# pattern = [('.|.'*(2*i + 1)).center(m, '-') for i in range(n//2)]
# print('\n'.join(pattern + ['WELCOME'.center(m, '-')] + pattern[::-1]))\

#? String Formatting
#* Example
# data = ("John", "Doe", 53.44)
# format_string = "Hello %s %s. Your current balance is $%s."
# print(format_string % data)
#* Task
# def print_formatted(n):
#     for i in range(1,n + 1):
#       pad = n.bit_length() # bit_length() return int length by not include 0
#       print(f'{i:{pad}d} {i:{pad}o} {i:{pad}X} {i:{pad}b}')

# n = int(input())
# print_formatted(n)

#? Alphabet rangoli
# def print_rangoli(size):
#     n = size
#     #* Define character by referring mapping to Ascii code
#     li = list(map(chr,range(97,123)))
#     #* Create logic for processing by dividing each set followed input
#     x = li[n-1: :-1] + li[1:n] 
#     #* lenght input from user
#     m = len('-'.join(x))
#     #* print character on upper part
#     for i in range(1, n):
#         print('-'.join(li[n-1:n-i:-1]+li[n-i:n]).center(m,'-'))
#     #* print character on lower part
#     for i in range(n, 0, -1):
#         print('-'.join(li[n-1:n-i:-1]+li[n-i:n]).center(m,'-'))
#     #! li[n-1:n-i:-1]+li[n-i:n] is sort char by divide each part

# n = int(input())
# print_rangoli(n)

#! Reverse Word and Swap Cases
# def reverse_words_order_and_swap_cases(sentence):
#     re = sentence
#     reverse = ' '.join(reversed(re))
#     swap = reverse.swapcase()
#     return swap

# sentence = input().split()
# result = reverse_words_order_and_swap_cases(sentence)
# print(result)

#! Dominant Cells (NOT DONE YET!)
# def numCells(grid):
#     n = grid_rows
#     m = grid_columns
# grid_rows = int(input().strip())
# grid_columns = int(input().strip())
# grid = []
# for _ in range(grid_rows):
#         grid.append(list(map(int, input().rstrip().split())))
# result = numCells(grid)
# print(result)

#? Capitalize
# #* My idea is work but it's not capitalize on each first word that contained with numeric. 
# def solve(s):
#     joined = ''.join(s)
#     cap = joined.title()
#     return cap
# s = input()
# result = solve(s)
# print(result)
# #* Actcually work
# def solve(s):
#     m = s.split(' ')
#     output = ' '.join((word.capitalize() for word in m))
#     return output
# s = input()
# result = solve(s)

#? Call items list from index
# n = ["","January","February","March","Apri","May","June","July","August","September","October","November","December"]
# m = int(input("Enter number for checking month: "))
# if m not in range(1,13):
#   print("The input isn't matched. Please check if your input is between 1-12.")
#   exit()
# call = n[m]
# print(call)

#? SwapCase
# def swap_case(s):
#     swap = s.swapcase()
#     return swap

# s = input()
# result = swap_case(s)
# print(result)

#? Merge the Tools
#* My Idea ;))
# def merge_the_tools(string, k):
#     s = string
#     lentgh = len(s)
#     n = 0
#     d = k 
#     for i in range(lentgh+k):
#       if i == k:
#         slic = slice(n,k)
#         m = (s[slic])
#         result = print("".join(dict.fromkeys(m)))
#         k += d
#         n += d
#     return result
# string, k = input(), int(input())
# merge_the_tools(string, k)
#* Short ver.
# def merge_the_tools(string, k):
#     for i in range(0, len(string), k):
#         uniq = ''
#         for c in string[i : i+k]:
#             if (c not in uniq):
#                 uniq+=c
#         print(uniq)
# string, k = input(), int(input())
# merge_the_tools(string, k)

#? Set
# def average(array):
#     output = sum(set(array))/len(set(array))
#     return output
# n = int(input())
# arr = list(map(int, input().split()))
# result = average(arr)
# print(result)

#? No Idea!
#! This solution is work with some input
# #* Create variable input by followed the sample input
# n, m = map(int,input().split())
# arr = list(map(int,input().split()))
# A = map(int,input().split())
# B = map(int,input().split())
# print(sum((i in A) - (i in B) for i in arr))
# #* Explained: sum = for i in arr (if A(i) belong arr(i)) - (if B(i) belong arr(i))
# #* By any value in set is count by index (values set = index start with 1)
# #* Example arr = [5,6,8,9] A = {5,9} B = {6,2,8}
# #* A = 2 units - B = 1 unit cause A(i) is all belong to arr (5,9), but B(i) have only one that belong to arr (6)
#! This solution is work with all input
# n,m = map(int,input().split())
# arr = list(map(int,input().split()))
# A = set(map(int,input().split()))
# B = set(map(int,input().split()))
# #Union set A & B
# U = A | B
# #Exclude all numbers which doesn't exit in both A & B
# arr = (i for i in arr if i in U)
# #Add 1 if number is in set A else subtract 1
# print(sum(1 if i in A else -1 for i in arr ))

#? Python If-Else
# N = int(input().strip())
# if N % 2 != 0:
#     print("Weird")
# elif 2 <= N and N <= 5:
#     print("Not Weird")
# elif 6 <= N and N <= 20:
#     print("Weird")
# else:
#     print("Not Weird")

#? Symmetric Difference
# M = int(input())
# A = list(map(int,input().split()))
# setA = set(A)
# N = int(input())
# B = list(map(int,input().split()))
# setB = set(B)
# differA = setA.difference(setB)
# differB = setB.difference(setA)
# U = differA.union(differB)
# listed = list(U)
# sort = sorted(listed)
# for i in range(len(sort)):
#     print(sort[i])

#? Set .discard(), .remove() & .pop()
#* Normal version
# n = int(input())
# s = set(map(int, input().split()))
# N = int(input())
# for i in range(N):
#     cmd = input().split()
#     if cmd[0] == "pop":
#         s.pop()
#     elif cmd[0] == "remove":
#         s.remove(int(cmd[1]))
#     elif cmd[0] == "discard":
#         s.discard(int(cmd[1]))
#     else:
#         print("Error")
# print(sum(s))
#* use eval() methold for excuteted code in string
# n = int(input())
# s = set(map(int, input().split())) 
# for i in range(int(input())):
#     #* {0} is cmd input and {1} is set(int(s)).
#     #* empty list in the end is for pop method cause formatting need 2 parameters
#     #* but pop method doesn't
#     eval('s.{0}({1})'.format(*input().split()+['']))
# print(sum(s))

#? Set.add()
# s = set()
# n = int(input())
# for i in range(n):
#     name = str(input())
#     s.add(name)
# print(len(s))

#? Set .intersection() Operation
# a = int(input())
# a_roll = set(map(int,input().split()))
# b = int(input())
# b_roll = set(map(int,input().split()))
# intersec = a_roll.intersection(b_roll)
# print(len(intersec))

#? Set .difference() Operation
# a = int(input())
# a_roll = set(map(int,input().split()))
# b = int(input())
# b_roll = set(map(int,input().split()))
# diff = a_roll.difference(b_roll)
# print(len(diff))

#? Set .symmetric_difference() Operation
#* Like a difference but return both while not in intersection
# a = int(input())
# a_roll = set(map(int,input().split()))
# b = int(input())
# b_roll = set(map(int,input().split()))
# symmetric_diff = a_roll.symmetric_difference(b_roll)
# print(len(symmetric_diff))

#? Set Mutations
# n = int(input())
# setA = set(map(int,input().split()))
# line = int(input())
# for i in range(line):
#     cmd, args = input().split(" ")
#     setB = set(map(int, input().split(" ")))
#     eval('setA.'+cmd+'(setB)')
# print(sum(setA))

#? Union
# a = int(input())
# a_roll = set(map(int,input().split()))
# b = int(input())
# b_roll = set(map(int,input().split()))
# U = a_roll.union(b_roll)
# print(len(U))

#? The Captain's Room
#* How it work 
#! this methold is looking for every elements in arr and check if was a non-repeated number.
#! So this will slow down processing, USE! When if you don't know pattern for setting group elements.
# def captianRoom(arr, n):
#     #* loop elements in arr one by one
#     for i in range(n):
#         j = 0
#         #* check elements if was more than one
#         while(j < n):
#             if (i != j and arr[i] == arr[j]):
#                 break
#             j += 1
#         if (j == n):
#             return arr[i]
#     return -1
# s = int(input())
# arr = list(map(int,input().split()))
# n = len(arr)
# print(captianRoom(arr, n))
#* ShortVer
#! We simply calculate the difference in what the sum would be 
#! if there were K elements of all groups. We will have (k-1 * captains room number left),
#! we simply didve by k-1 to get the answer.
# k,arr = int(input()),list(map(int, input().split()))
# myset = set(arr)
# print( ( (sum(myset) * k) - (sum(arr)) )//(k-1) )

#? Check Subset
# T = int(input())
# for i in (range(T)):
#     a = int(input())
#     setA = set(map(int,input().split()))
#     b = int(input())
#     setB = set(map(int,input().split()))
#     print(setA.issubset(setB))

#? Check Strict Superset
# #* The issuperset() method returns True if a set has every 
# #* elements of another set (passed as an argument). If not, it returns False.
# setA = set(map(int,input().split()))
# n = int(input())
# setB = set(map(int,input().split()))
# setC = set(map(int,input().split()))
# if (setA.issuperset(setB)) and (setA.issuperset(setC)):
#     print(True)
# else:
#     print(False)

#? Collections.Counter()
# from collections import Counter
# N_list = int(input())
# listed = Counter(map(int,input().split()))
# n = int(input())
# income = 0
# for i in range(n):
#     size, price = map(int,input().split())
#     if listed[size]:
#         income += price
#         listed[size] -= 1
# print(income)

#? itertools.product()
# # * product() it's equivalent to nested for-loops 
# # * (Relationship on Cartesian Products {SetX} x {SetY})
# from itertools import product
# listA = list(map(int,input().split()))
# listB = list(map(int,input().split()))
# print(*product(listA,listB)) # * operator is for unpack type values

#? itertools.permutations()
# #* This tool returns successive r length permutations of elements in an iterable.
# #* If r is not specified or is None, then  r defaults to the length of the iterable, 
# #* and all possible full length permutations are generated.
# from itertools import permutations
# string, integer = input().split(" ")
# tolal = sorted(list(permutations(string,int(integer))))
# for i in tolal:
#     print(''.join(i))

#? DefaultDict Tutorial
#* myVersion.
# from collections import defaultdict
# d = defaultdict(list)
# n, m = map(int,input().split())
# listA = []
# listB = []
# for i in range(n):
#     listA.append(input())
# for j in range(m):
#     listB.append(input())

# for i in range(n):
#     d[listA[i]].append(i+1)
# for i in listB:
#     if i in d:
#         print(*d[i])
#     else:
#         print(-1)
#* shortVersion.
# from collections import defaultdict
# n ,m = map(int,input().split())
# d = defaultdict(list)
# for i in range(1,n+1):
#     d[input()].append(str(i))
# for i in range(m):
#     print(' '.join(d[input()]) or -1)

#? Maximize It!
#* myVerison.
# from itertools import product
# def maxNum(arr):
#     max_number = max(arr)
#     return max_number

# arr = []
# k, m = map(int,input().split())
# for i in range(k):
#     listed = list(map(int,input().split()))
#     arr.append(maxNum(listed))
# result = map(lambda a: sum(i**2 for i in a) % m, product(arr))
# print(sum(result))
#* shortVer.
# from itertools import product
# K,M = map(int,input().split())
# N = (list(map(int, input().split()))[1:] for _ in range(K))
# results = map(lambda x: sum(i**2 for i in x)%M, product(*N))
# print(max(results))

#? The Minion Game
# def minion_game(s):
#     s = s.upper()
#     print(s)
#     stuart,kevin = 0, 0
#     for i in range(len(s)):
#         if s[i] in ["A","E","I","O","U"]:
#             kevin += len(s) - i
#         else: 
#             stuart += len(s) - i

#     if kevin > stuart:
#         print("Kevin", kevin)
#     elif kevin < stuart:
#         print("Stuart", stuart)
#     else:
#         print("Draw")
# s = input()
# minion_game(s)

#* Polar Coordinates
# import cmath
# c = complex(input().strip())
# unpack = cmath.polar(c)
# for i in unpack:
#     print(i)