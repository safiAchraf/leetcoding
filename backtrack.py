# Given two integers n and k, return all possible combinations of k numbers chosen from the range [1, n].
def combinations(n, k):
    result = []
    def helper(i, rs):
        if len(rs) == k:
            result.append(rs)
            return
        if i > n:
            return
        helper(i+1, rs+[i])
        helper(i+1, rs)

    return result













def generateParenthesis(n):
    result = []
    
    def helper(open, close, s):
        if close < open or close < 0 or open < 0:
            return
        if open == 0 and close == 0:
            result.append(s)
            return
        helper(open-1, close, s + '(')
        helper(open, close-1, s + ')')

    helper(n, n, '')
    return result



def cookies(cookies , k): # leetcode number 2305
    rs = float('inf')
    def helper(childs , i):
        nonlocal rs
        if i == len(cookies):
            rs = min(rs,max(childs))
        else:
            for j in range(k):
                newchilds = childs[:]
                newchilds[j] += cookies[j]
                helper(newchilds , i+1)
    helper([0 for i in range(k)],0)
    return rs


def maxscoreOfIsland(grid):
    maxscore = 0
    visited = [[False for i in range(len(grid[0]))] for j in range(len(grid))]
    moves = [[0,1],[0,-1],[1,0],[-1,0]]
    def islands( i , j):
        if (i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or visited[i][j]):
            return 0
        maxscore = 0
        maxscore = max(maxscore , grid[i][j])
        visited[i][j] = True
        for r , c in moves:
            nr , nc = i + r , j + c
            if nr >= 0 and nr < len(grid) and nc >= 0 and nc < len(grid[0]) and grid[nr][nc] <= grid[i][j]+1:
                maxscore = max(islands(nr , nc),maxscore)
        return maxscore
    
    return islands(0,0)

moves = [[0,1],[0,-1],[1,0],[-1,0]]

def magicalgrid(n,m ,q):
    grid = [[False for i in range(m)] for j in range(n)]
    visited = [[False for i in range(m)] for j in range(n)]
    def helper(i , j):
        if i<0 or i >= n or j < 0 or j >= m or visited[i][j]:
            return False
        if i == n-1 :
            return True
        visited[i][j] = True
        ans = False
        for r , c in moves:
            nr , nc = i + r , j + c
            if nr >= 0 and nr < n and nc >= 0 and nc < m and grid[nr][nc]:
                ans = helper(nr , nc) or ans
            if ans:
                break
        return ans
        
    rs = -1
    for i in range(q):
        x , y = map(int,input().split())
        grid[x-1][y-1] = True
        if rs == -1 :
            test = False
            for ind in grid[-1]:
                test = test or ind
            if test:
                for ind in range(n):
                    if grid[0][ind] and helper(ind , 0):
                        rs = i + 1
                        break

    return rs


def countAndSay(n : int):
    if n == 1 :
        return "1"
    def helper(s):
        if n == 1:
            return '1'
        rs = ""
        i = 0
        while i < len(s):
            tmp = 1
            num = s[i]
            while i + 1 < len(s) and s[i] == s[i+1]:
                tmp += 1
                i+= 1
            rs += str(tmp) + num
            i+=1
        return rs
    result = helper('1')
    for i in range(n - 2):
        result = helper(result)
    return result

def visited(grid):
    cell = [[False for i in range(len(grid[0]))] for j in range(len(grid))]
    def helper( i , j ):
        if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or cell[i][j]:
            return 0
        cell[i][j] = True

        return 1 + helper(i+1 , j) + helper(i-1 , j) + helper(i , j+1) + helper(i , j-1)
    result = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if not cell[i][j]:
                result = max(result , helper(i ,j))
    return result


def largestIsland(grid): # leetcode number 827
    visited = [[False for i in range(len(grid[0]))] for j in range(len(grid))]
    marked = set()

    neighbords = [] 
    def helper(i , j):
        if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] != 1 or visited[i][j]:
            return 0
        visited[i][j] = True
        if i-1 >= 0 and grid[i-1][j] != 1 and (i-1 , j) not in marked:
            neighbords.append([i-1 , j])
            marked.add((i-1 , j))
        if i+1 < len(grid) and grid[i+1][j] != 1 and (i+1 ,j) not in marked:
            neighbords.append([i+1 , j])
            marked.add((i+1 , j))
        if j-1 >= 0 and grid[i][j-1] != 1 and (i , j-1) not in marked:
            neighbords.append([i , j-1])
            marked.add((i , j-1))
        if j+1 < len(grid[0]) and grid[i][j+1] != 1 and (i , j+1) not in marked:
            neighbords.append([i , j+1])
            marked.add((i , j+1))
        return 1 + helper(i+1 , j) + helper(i-1 , j) + helper(i , j+1) + helper(i , j-1)
    
    zeroPresent = False
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 0:
                zeroPresent = True
                break
    if not zeroPresent :
        return len(grid) * len(grid[0])
    
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                if not visited[i][j]:
                    surface = helper(i , j )
                    for r , c in neighbords:
                        grid[r][c] = str(int(grid[r][c]) + surface)
                    marked = set()
                    neighbords = []
    result = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] != 1:
                result = max(result , int(grid[i][j]))
    
    return result + 1





def solution(A):
    N = len(A)

    # Handle edge cases:
    if N == 0:
        return 0
    if N == 1:
        return A[0]

    # Create a DP table to store maximum sums for subarrays ending at each index
    dp = [0] * N
    dp[0] = A[0]
    dp[1] = max(A[0] * 10 + A[1], A[1])  # Consider merging first two elements

    # Fill the DP table using the recurrence relation
    for i in range(2, N):
        dp[i] = max(dp[i - 1], dp[i - 2] * 10 + A[i])  # Choose to merge or not

    return max(dp)


# Python3 program to find XNOR of two numbers.
import math

# Returns XNOR of num1 and num2
def XNOR(a, b):
	numOfBits = int(math.log(max(a, b), 2))
	mask = (1 << numOfBits) - 1
	xnor = ~(a ^ b)
	return xnor & mask


arr = [9, 8, 5, 6, 3, 7, 2, 7]

