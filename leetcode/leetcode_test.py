
# ? 1. Two sum
# - My idea (really not work actaully lol)
import math
import time


def twoSum1():
    class Solution(object):
        def twoSum(nums, target):
            """
            :type nums: List[int]
            :type target: int
            :rtype: List[int]
            """
            listed = []
            for i in range(len(nums)):
                for j in range(len(nums) - 1):
                    if nums[i] + nums[j+1] == target:
                        listed.append(nums[i])
                        listed.append(nums[j + 1])
                        return listed
    nums = list(map(int,input().split()))
    print(nums)
    target = int(input())
    result = Solution.twoSum(nums,target)
    print(result)
    listed2 = []
    for i in nums:
        for j in result:
            if i == j:
                index = nums.index(i)
                listed2.append(index)
    print(listed2)
# - ShotVer.
def twoSum2():
    class Solution(object):
        def twoSum(self, nums, target):
            prevMap = {} #* hashMap: val: index
            for i, n in enumerate(nums):
                diff = target - n
                if diff in prevMap:
                    return [prevMap[diff], i]
                prevMap[n] = i
            return
    nums = list(map(int,input().split()))
    target = int(input())
    result = Solution.twoSum(nums,target)
    print(result)

# ? 7. Reverse Integer
# - My IDEA (The most performace!)!!
def reverse_int1():
    class Solution(object):
        def reverse(x):
            strip = [str(a) for a in str(x)]
            while True:
                if strip[0] == "0":
                    break
                elif strip[len(strip) - 1] == "0":
                    strip.pop()
                else:
                    break
            string = ""
            if strip[0] == "-":
                reverse = list(reversed(strip[1:]))
                reverse.insert(0,'-')
            else:
                reverse = strip[::-1] #* reversed
            for i in reverse:
                string += i

            if int(string) >= -2 ** 31 and int(string) <= (2**31 - 1):
                return int(string)
            else:
                return 0
    result = Solution.reverse(-2147483412)
    print(result)
#- ShortVer.
def reverse_int2():
    class Solution:
        def reverse(x):
            x = str(x)
            if x[0].isdigit(): #* a positive number
                x = x[::-1]
                x = int(x)
            else: #* a negative number
                x = x[1:]
                x = x[::-1]
                x = -1 * int(x)

            if x >= -2 ** 31 and x <= (2**31 - 1):
                return x
            else:
                return 0
    result = Solution.reverse(-2147483412)
    print(result)
# - mathematics Version.
def reverse_int3():
    class Solution:
        def reverse(x):
            MIN = -2147483648 # -2**31
            MAX = 2147483647  #- 2**31 - 1
            res = 0
            while x:
                digit= int(math.fmod(x, 10))
                x = int(x/10)

                if (res > MAX // 10 or
                    (res == MAX // 10 and digit >= MAX % 10)):
                    return 0
                elif (res < MIN // 10 or
                    (res == MIN // 10 and digit <= MAX % 10)):
                    return 0
                res = (res * 10) + digit
            return res
    result = Solution.reverse(-2147483412)
    print(result)

#? 2. Add Two Numbers
#- My IDEA!! # Definition for singly-linked list.
    
def addTwoNumbers():
    class ListNode(object):
        def __init__(self, val=0, next=None):
            self.val = val
            self.next = next
    class Solution(object):
        def addTwoNumbers(l1, l2):
            for i in l1:
                if i >= 0 and i <= 9:
                    continue
                else:
                return False
            for j in l2:
                if j >= 0 and j <= 9:
                    continue
                else:
                return False

            while True:
                if l1[0] == 0:
                    break
                elif l1[len(l1) - 1] == 0:
                    l1.pop()
                if l2[0] == 0:
                    break
                elif l2[len(l2) - 1] == 0:
                    l2.pop()
                else:
                    break
            res_l1 = list(map(str,l1[::-1]))
            res_l2 = list(map(str,l2[::-1]))
            str_l1 = ""
            str_l2 = ""

            for i in res_l1:
                str_l1 += i
            for j in res_l2:
                str_l2 += j
            result = int(str_l1) + int(str_l2)
            return result

    output = Solution.addTwoNumbers([2,4,3],[5,6,4])
    print(output)
#- ShortVer.
def addTwoNumbers2():
    class ListNode(object):
        def __init__(self, val=0, next=None):
            self.val = val
            self.next = next
    class Solution(object):
        def addTwoNumbers(self, l1, l2):
            dummy = ListNode()
            cur = dummy

            carry = 0
            while l1 or l2 or carry:
                v1 = l1.val if l1 else 0
                v2 = l2.val if l2 else 0
                #* new digit
                val = v1 + v2 + carry
                carry =  val // 10
                val %= 10
                cur.next = ListNode(val)
                #* update ptrs
                cur = cur.next
                l1 = l1.next if l1 else None
                l2 = l2.next if l2 else None
            return dummy.next

#? 9. Palindrome Number
#* A number that reversed and still has the same value like 14941, 9999 etc.
#- My IDEA swith to string and reversed (most performance)
def isPalindrome1():
    class Solution(object):
        def isPalindrome(x):
            if str(x)[0] == '-':
                return False
            reversed = str(x)[::-1]
            if x == int(reversed):
                return True
            else:
                return False
    result = Solution.isPalindrome(1011101)
    print(result)
#- My IDEA mathematics Version
def isPalindrome2():
    class Solution(object):
        def isPalindrome(x):
            if str(x)[0] == "-":
                return False
            new_x = x
            res = 0
            while new_x:
                digit = int(math.fmod(new_x,10)) #* get last digit
                new_x = int(new_x/10) #* rounded number
                res = (res * 10) + digit #* plus them (append)
            print(res)
            if res == x:
                return True
            else:
                return False
    result = Solution.isPalindrome(1210)
    print(result)

#? 13. Roman to Integer
#* Using map
def romanToInt():
    class Solution(object):
        def romanToInt(s):
            romane_digit = {
                "I": 1,
                "V": 5,
                "X": 10,
                "L": 50,
                "C": 100,
                "D": 500,
                "M": 1000
            }
            res = 0
            for i in range(len(s)):
                #* Check if is len romane bound of bones
                #* and previous romane is less then next romane
                #* then subtract it
                if i + 1 < len(s) and romane_digit[s[i]] < romane_digit[s[i + 1]]:
                    res -= romane_digit[s[i]]
                else: #* otherwise plus next romane
                    res += romane_digit[s[i]]
            return res
    result = Solution.romanToInt("XII")
    print(result)

? 14. Longest Common Prefix
#* MyIdea [ not work :( ]
def longestCommonPrefix():
    class Solution(object):
        def longestCommonPrefix(strs):
            print(strs)
            string = ""
            for i in range(len(strs)):
                for j in range(len(strs[i])):
                    if strs[i][j] == strs[i+1][j]:
                        if strs[i][j] == "":
                            break
                        string += strs[i][j]

            return string
    result = Solution.longestCommonPrefix(["flower","flow","flight"])
    print(result)

#* WorkVer.
def longestCommonPrefix2():
    class Solution(object):
        def longestCommonPrefix(strs):
            string = ""
            for i in range(len(strs[0])):
                for s in strs:
                    #* check if it's bound of bone or is it prefix or not
                    if i == len(s) or s[i] != strs[0][i]:
                        return string
                string += strs[0][i]

            return string
    result = Solution.longestCommonPrefix(["","flow","flight"])
    print(result)

#? Check last digit
def check_lastDigit(nums):
    def check_lastDigit(nums):
        res = 0
        nums = nums.replace("-","")
        for i in range(len(nums) - 1):
            res += (int(nums[i])) * (len(nums) - i)
        remain = res % 11
        checkDigit = str(11 - remain)
        return checkDigit[len(checkDigit) - 1]

    nums = str(input())
    result = check_lastDigit(nums)
    print(result)

#? 20. Valid Parentheses
def isValid():
    class Solution(object):
        def isValid(s):
            stack = []
            lookup = { '(':')', '[':']', '{':'}' }
            for i in s:
                if i in lookup:
                    stack.append(i)
                elif len(stack) == 0 or lookup[stack.pop()] != i:
                    return False

            return len(stack) == 0
    result = Solution.isValid('{()[]]')
    print(result)

#? 3. Longest Substring Without Repeating Characters
def lengthOfLongestSubstring():
    class Solution(object):
        def lengthOfLongestSubstring(s):
            charSet = set()
            l = 0
            res = 0
            for r in range(len(s)):
                while s[r] in charSet:
                    charSet.remove(s[l])
                    l += 1
                charSet.add(s[r])
                res = max(res, len(charSet))
            return res

    res = Solution.lengthOfLongestSubstring('abcabcbb')
    print(res)

#? 5. Longest Palindromic Substring
#- Time complextiy O(n^2)
#- find every str and search from outwards to innerwards (left-right side) it would be n * n
def longestPalindrome():
    class Solution(object):
        def longestPalindrome(s):
            res = ""
            resLen = 0

            for i in range(len(s)):
                #* odd length
                l, r = i, i  #* defined left right counts
                #* check if in bound of bones and it's palindromic
                while l >= 0 and r < len(s) and s[l] == s[r]:
                    #* update str and len till the last str
                    if (r - l +1) > resLen:
                        res = s[l:r+1]
                        resLen = r - l + 1
                    l -= 1
                    r += 1
                #* even length
                l, r = i, i + 1
                while l >= 0 and r < len(s) and s[l] == s[r]:
                    if (r - l + 1) > resLen:
                        res = s[l:r+1]
                        resLen = r - l + 1
                    l -= 1
                    r += 1

            return res
    res = Solution.longestPalindrome('babad')
    print(res)

#? 21. Merge Two Sorted Lists
#! Test with leetcode test server
#- Space: O(n): A new linked list with length equal to the sum of l1 and l2.
#- Time: O(n): Passing the shorter link list to the end.
#* Definition for singly-linked list.
def mergeTwoLists():
    class ListNode(object):
        def __init__(self, val=0, next=None):
            self.val = val
            self.next = next
    class Solution(object):
        def mergeTwoLists(l1, l2):
            dummy = ListNode()
            #* curr is None
            curr = dummy
            #* both of them aren't emtpy
            while l1 and l2:
                if l1.val < l2.val:
                    #* set next curr to l1
                    curr.next = l1
                    l1 = l1.next
                else:
                    #* set next curr to l2
                    curr.next = l2
                    l2 = l2.next
                curr = curr.next #* set it back to None (default param)
    #* if l1 or l2 aren't empty(Null) yet then add them at the last of Node.
        if l1:
            curr.next = l1
        else l2:
            curr.next = l2
        return dummy.next

    res = Solution.mergeTwoLists([1,2,4],[1,3,4])
    print(res)

#? 28. Implement strStr()
#* My Idea using with index method
def strStr():
    class Solution(object):
        def strStr(haystack, needle):
            if needle == "":
                return 0
            elif needle in haystack:
                return haystack.index(needle)
            else:
                return -1
    res = Solution.strStr("hello","ll")
    print(res)
#* Not using any methods (Brute-Force Version)
def strStr2():
    class Solution(object):
        def strStr(haystack, needle):
            h = len(haystack)
            nd = len(needle)
            if h == nd and h == 0:
                return 0
            elif needle in haystack:
                for i in range(h-nd + 1):
                    j = 0
                    while j < nd and needle[j] == haystack[i+j]:
                        j += 1
                    if j == nd:
                        return i
            else:
                return -1

    res = Solution.strStr("a","a")
    print(res)

#? Find duplicates in list
def duplicateCount():

    def duplicateCount(n):
        arr1 = []
        arr2 = []
        c = 0
        for i in range(n):
            n1 = int(input())
            arr1.append(n1)
            n2 = int(input())
            arr2.append(n2)
            for i in arr1:
                for j in range(len(arr2)):
                    if i == arr2[j]:
                        c += 1
                        arr2.remove(arr2[j])
                        break
        return c

    arrSize = int(input())
    res = duplicateCount(arrSize)
    print(res)

    start_time = time.time()
    a = 0
    for i in range(1000):
        a += (i**100)
    end = time.time()
    print("The time of execution of above program is :", end-start_time)

#? 27. Remove elements
def removeElement():
    class Solution(object):
        def removeElement(nums, val):
            while nums:
                if val in nums:
                    nums.remove(val)
                else:
                    break
            return len(nums)

    res = Solution.removeElement([3,2,2,3],3)
    print(res)

#? 29. Divide Two Integers
#* My Idea math version (Brute force).
def divide():
    class Solution(object):
        def divide(dividend, divisor):
            remain = 0
            c = 0
            while remain <= math.abs(dividend):
                if (remain + math.abs(divisor)) > math.abs(dividend):
                    break
                else:
                    remain += math.abs(divisor)
                    c += 1
            if divisor < 0 and dividend < 0:
                return c
            elif divisor < 0 or dividend < 0:
                return -(c)
            return c
    res = Solution.divide(10,5)
    print(res)

#* Bitwise left shift == n*2 Ver.
def divide2():
    class Solution(object):
        def divide(dividend, divisor):
            if not dividend:
                return 0
            sign = 1 if (dividend > 0 and divisor < 0) or (dividend < 0 and divisor > 0) else 0
            dividend = abs(dividend)
            divisor = abs(divisor)
            res = 0
            while dividend >= divisor:
                k = 0
                while dividend >= divisor << (k + 1):
                    k += 1
                dividend -= (divisor << k) #* dividened remain
                res += (1 << k) #* divide remain.
            MAX_INT =  (1 << 31) - 1
            return -res if sign else (res if res <= MAX_INT else MAX_INT)

    res = Solution.divide(10,5)
    print(res)

#? 35. Search Insert Position
#! Time complexity O(log n)
def searchInsert():
    class Solution(object):
        def searchInsert(nums, target):
            l, r = 0, len(nums) - 1

            while l <= r:
                mid = (l+r)//2
                if target == nums[mid]:
                    return mid

                if target > nums[mid]:
                    l = mid + 1
                else:
                    r = mid - 1
            return l

    res = Solution.searchInsert([1,2,3,5], 4)
    print(res)

#? 6. ZigZag Conversion
#! Both time complexity O(n), n is the number of characters in the string
#* NeetCode's verison (Using only mathematic to find each chr)
def zigZag_convert():
    class Solution(object):
        def convert(s, numRows):
            if numRows == 1: return s

            res = ""
            for r in range(numRows):
                #* chr print pattern (start and end rows)
                increment = (numRows-1) * 2
                for i in range(r, len(s), increment):
                    res += s[i]
                    #* check if it was in middle of rows (0, numRows - 1)
                    #* and if it didn't bound of bones of string (s)
                    if (r > 0 and r < numRows - 1 and
                        i + increment - 2 * r < len(s)):
                        #* nested chr print pattern (between rows)
                        res += s[i + increment - 2 * r]
            return res

    res = Solution.convert("PAYPALISHIRING", 3)
    print(res)

#* Performance Version (similar to first algorithm)
def zigZag_convert2():
    class Solution(object):
        def convert(s, numRows):
            if numRows == 1: return s

            res = []
            cycle = (2 * numRows) - 2
            for r in range(numRows):
                for i in range(r, len(s), cycle):
                    res.append(s[i])
                    k = i + cycle - 2 * r
                    if r != 0 and r != numRows - 1 and k < len(s):
                        res.append(s[k])
            return ''.join(res)

    res = Solution.convert("PAYPALISHIRING", 3)
    print(res)

#? 4. Median of Two Sorted Arrays
def findMedianSortedArrays():
    class Solution(object):
        def findMedianSortedArrays(nums1,nums2):

#? 53. Maximum Subarray
#* Brute Force O(n**2)
def maxSubArray():
    class Solution(object):
        def maxSubArray(nums):
            maximun = 0
            for i in range(len(nums)):
                if maximun < sum(nums[i:len(nums)]):
                    maximun = sum(nums[i:len(nums)])
            return maximun

    res = Solution.maxSubArray([-2,-3,5,9,10,-5])
    print(res)
#* Kadane's Algorithim O(n)
def maxSubArray2():
    class Solution(object):
        def maxSubArray(nums):
            if(len(nums) == 0):
                return 0
            sum_max = nums[0]
            sum_crr = nums[0]

            for i in range(1, len(nums)):
                n = nums[i]
                sum_crr = max(sum_crr + n,n)
                if sum_crr > sum_max:
                    sum_max = sum_crr
            return sum_max
    res = Solution.maxSubArray([5,4,-1,7,8])
    print(res)

#? 58. Length of Last Word
#* My idea
def lengthOfLastWord():
    class Solution(object):
        def lengthOfLastWord(s):
            s = s.split()
            return len(s[-1])

    res = Solution.lengthOfLastWord("Hello ssadsadsasad World")
    print(res)

#? Maximum lenght substr in arr O(n)
#* Find maximum str 
def lengthOfLastWord2():
    class Solution(object):
        def lengthOfLastWord(s):
            s = s.split()
            arr = list(map(str,s))
            max_len = len(arr[0])
            for i in range(1,len(arr)):
                if len(arr[i]) > max_len:
                    max_len = len(arr[i])
            return max_len

    res = Solution.lengthOfLastWord("Hello s World")
    print(res)

#? 11. Container With Most Water
#* Brute force O(n**2)
def maxArea():
    class Solution(object):
        def maxArea(height):
            res = 0
            for l in range(len(height)):
                for r in range(l+1, len(height)):
                    #* area of container width * height
                    area = (r - l) * min(height[l], height[r])
                    #* update the result
                    res = max(res, area)
            return res


    res = Solution.maxArea([4, 3, 2, 1, 4])
    print(res)

#* Linear time O(n)
def maxArea2():
    class Solution(object):
        def maxArea(height):
            res = 0
            l, r = 0, len(height) - 1

            while l < r:
                #* area of container width * height
                area = (r - l) * min(height[l], height[r])
                #* update the result
                res = max(res, area)

                if height[l] < height[r]:
                    l += 1
                else:
                    r -= 1
            return res

    res = Solution.maxArea([4, 3, 2, 1, 4])
    print(res)

#? 217. Contains Duplicate
#* My Idea Check if lenght list is equal to set length?
#* When convert list to set, duplicate nums isn't allowed. 
#* So in set won't have any duplicate nums.
def containsDuplicate():
    class Solution(object):
        def containsDuplicate(nums):
            orginLenght = len(nums)
            numSet = len(set(nums))
            True if orginLenght != numSet else False

    res = Solution.containsDuplicate([1,2,3,4])
    print(res)

#? 121. Best Time to Buy and Sell Stock
#* My Idea
def maxProfit():
    class Solution(object):
        def maxProfit(prices):
            #* find min(buy), max(stock) in arr
            if len(prices) <= 1:
                return 0

            buy = min(prices[:-1])
            stock = max(prices)
            buyIndex, stockIndex = prices.index(buy), prices.index(stock)
            print(buy,buyIndex)
            print(stock,stockIndex)
            #* check if is possible to get profit?
            while buyIndex > stockIndex:
                prices = prices[buyIndex:]
                stock = max(prices)
                profit = stock - buy
                return profit
            if buyIndex < stockIndex:
                profit = stock - buy
                return profit
            elif prices[-1] <= buy:    
                return 0
                
    res = Solution.maxProfit([3,2,6,5,0,3])
    print(res)
#* Work Version
#* go through every elem check if index min > index max then shiff
#* by min = max and max increase by then update profit 
#! Linear time Solution O(n)

def maxProfit2():
    class Solution(object):
        def maxProfit(prices):
            if len(prices) <= 1:
                return 0
            Min, Max = 0, 1
            max_profit = 0
            while Max < len(prices):
                if prices[Min] > prices[Max]:
                    Min = Max
                else:
                    profit = prices[Max] - prices[Min]
                    max_profit = max(profit, max_profit)
                Max +=1
            return max_profit
    res = Solution.maxProfit([2,1,2,1,0,1,2])
    print(res)


#? 238. Product of Array Except Self
#- My idea (Brute Force O(n**2))
def productExceptSelf():
    import functools as ft
    class Solution(object):
        def productExceptSelf(nums):
            res = []
            for i in nums:
                nums.remove(i)
                elem = ft.reduce(lambda a,b: a*b, nums)
                res.append(elem)
                nums.insert(0,i)
            return res
    res = Solution.productExceptSelf([-1,1,0,-3,3])
    print(res)
#- Linear time O(n)
#* My idea: find mutilply maximum arr and divide by each elem.
def liner_time():
    import functools as ft
    class Solution(object):
        def productExceptSelf(nums):
            res = []
            ls = nums[:]
            while 0 in ls:
                ls.remove(0)
            ls = [0,0] if len(ls) == 0 else ls
            maximum = ft.reduce(lambda a,b: a*b, nums)
            maximum_ls =  ft.reduce(lambda a,b: a*b, ls)
            for i in nums:
                if i == 0:
                    res.append(maximum_ls)
                else:
                    elem =  maximum // i
                    res.append(elem)
            return res
    res = Solution.productExceptSelf([0,4,0])
    print(res)
#* NeetCode: find prefix and postfix then mutiply them each elem
def liner_time2():
    class Solution(object):
        def productExceptSelf(nums):
            #* in arr, make every value elem equal to 1 
            #* (make it easier to add perfix and postfix)
            res = [1] * (len(nums))

            #* compute prefix in arr
            prefix = 1
            for i in range(len(nums)):
                res[i] = prefix
                prefix *= nums[i]
            #* compute postfix in arr
            postfix = 1
            for i in range(len(nums)-1, -1, -1):
                res[i] *= postfix
                postfix *= nums[i]
            return res
    res = Solution.productExceptSelf([0,4,0])
    print(res)

#? 152. Maximum Product Subarray
#- Time complextiy O(n)
def maxProduct():
    class Solution(object):
        def maxSubArray(nums):
            res = max(nums) #* 0 -> [-1]
            currMin, currMax = 1, 1

            for n in nums:
                temp =  currMax * n
                currMax = max(n * currMax, n * currMin, n)
                currMin = min(temp, n * currMin, n)
                res = max(res, currMax)
            return res
    res = Solution.maxSubArray([0,4,0])
    print(res)

#? 153. Find Minimum in Rotated Sorted Array
#- Binary Serach Linear time O(n)
def findMin():
    class Solution(object):
        def findMin(nums):
            res = nums[0]
            l,r = 0, len(nums) - 1
            while l <= r:
                if nums[l] < nums[r]:
                    res = min(res,nums[l])
                    break
            
                m = (l + r) // 2
                res = min(res,nums[m])
                if nums[m] >= nums[l]:
                    l = m + 1
                else:
                    r = m - 1
            return res
    ls = list(map(int, input('').split()))
    print(ls)

def main(run_number):
    def run(run_number):
        switcher = {
            1: twoSum1,
            2: twoSum2,
            3: reverse_int1,
            4: reverse_int2,
            5: reverse_int3,
            6: addTwoNumbers,
            7: addTwoNumbers2,
            8: isPalindrome1,
            9: isPalindrome2,
            10: romanToInt,
            11: longestCommonPrefix,
            12: longestCommonPrefix2,
            13: check_lastDigit,
            14: isValid,
            15: lengthOfLongestSubstring,
            16: longestPalindrome,
            17: mergeTwoLists,
            18: strStr,
            19: strStr2,
            20: duplicateCount,
            21: removeElement,
            22: divide,
            23: divide2,
            24: searchInsert,
            25: zigZag_convert,
            26: zigZag_convert2,
            27: findMedianSortedArrays,
            28: maxSubArray,
            29: maxSubArray2,
            30: lengthOfLastWord,
            31: lengthOfLastWord2,
            32: maxArea,
            33: maxArea2,
            34: containsDuplicate,
            35: maxProfit,
            36: maxProfit2,
            37: productExceptSelf,
            38: liner_time,
            39: liner_time2,
            40: maxProduct,
            41: findMin
        }
        func = switcher.get(run_number, lambda: print("Invalid run number"))
        func()

    run(run_number)
    
if __name__ == "__main__":
    run_number = int(input("Enter the task number: "))
    main(run_number)
