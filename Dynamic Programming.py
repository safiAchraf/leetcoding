def uniquePaths(m, n):
    memo = {}

    def unique_paths(i, j):
        if (i, j) in memo or (j, i) in memo:
            return memo[(i, j)]
        if i == 1 and j == 1:
            return 1
        if i == 0 or j == 0:
            return 0
        memo[(i, j)] = unique_paths(i-1, j) + unique_paths(i, j-1)
        return memo[(i, j)]
    return unique_paths(m, n)


def uniquePathsWithObstacles(obstacleGrid: list[list[int]]) -> int:
    memo = {}
    m = len(obstacleGrid)
    n = len(obstacleGrid[0])

    def unique_paths(i, j):
        if (i, j) in memo:
            return memo[(i, j)]

        if i == m or j == n or obstacleGrid[i][j] == 1:
            return 0

        if i == m - 1 and j == n - 1:
            return 1

        memo[(i, j)] = unique_paths(i + 1, j) + unique_paths(i, j + 1)
        return memo[(i, j)]

    return unique_paths(0, 0)


def Shortestsum(target, nums):
    memo = {}

    def summ(target, nums):
        shortest = None
        if target in memo:
            return memo[target]
        if target == 0:
            return []
        if target < 0:
            return None
        for i in nums:
            result = summ(target - i, nums)
            if result != None:
                if shortest == None or len(result) < len(shortest):
                    shortest = result+[i]
        memo[target] = shortest
        return shortest

    return summ(target, nums)


def word(target, wordBank):
    memo = {}

    def word_search(target):
        if target in memo:
            return memo[target]
        if target == '':
            return True
        for word in wordBank:
            if target.startswith(word):
                memo[target] = word_search(target[len(word):])
                if memo[target] == True:
                    return True
        memo[target] = False

        return memo[target]

    return word_search(target)


def number_word(target, wordBank):
    memo = {}

    def word_search(target):
        if target in memo:
            return memo[target]
        if target == '':
            return 1
        sum = 0
        for word in wordBank:
            if target.startswith(word):
                sum += word_search(target[len(word):])
        memo[target] = sum
        return memo[target]

    return word_search(target)


def construct_word(target, wordBank):
    final = []

    def helper(target, rs):
        if target == '':
            final.append(rs)
            return
        for word in wordBank:
            if target.startswith(word):
                helper(target[len(word):], rs + [word])

    helper(target, [])
    return final


def minCostStairs(cost):
    cost.append(0)
    cost.append(0)
    length = len(cost)-3
    for i in range(length, -1, -1):
        cost[i] += min(cost[i+1], cost[i+2])

    return min(cost[0], cost[1])


def can_sum(tab, target):
    dp = [False]*(target+1)
    dp[0] = True

    for i in range(len(dp)):
        if dp[i]:
            for j in range(len(tab)):
                ind = i+tab[j]
                if ind < len(dp):
                    dp[ind] = True
    return dp[-1]


def how_sum(arr, target):
    dp = [None]*(target+1)
    dp[0] = []
    for i in range(len(dp)):
        if dp[i] != None:
            for j in range(len(arr)):
                ind = i+arr[j]
                if ind < len(dp):
                    dp[ind] = dp[i][:]
                    dp[ind].append(arr[j])
    return dp[-1]


def best_sum(arr, target):
    dp = [None]*(target+1)
    dp[0] = []
    for i in range(len(dp)):
        if dp[i] != None:
            for j in range(len(arr)):
                ind = i+arr[j]
                if ind < len(dp):
                    tmp = dp[i][:] + [arr[j]]
                    if dp[ind] == None or len(tmp) < len(dp[ind]):
                        dp[ind] = tmp

    return len(dp[-1])


def can_construct(word, bank):
    dp = [False]*(len(word)+1)
    dp[0] = True
    for i in range(len(dp)):
        if dp[i]:
            for j in range(len(bank)):
                if word[i:].startswith(bank[j]):
                    dp[i+len(bank[j])] = True
    return dp[-1]


def countConstruct(word, bank):
    dp = [0]*(len(word)+1)
    dp[0] = 1
    for i in range(len(dp)):
        if dp[i] != -1:
            for j in range(len(bank)):
                if word[i:].startswith(bank[j]):
                    dp[i+len(bank[j])] += dp[i]
    return dp[-1]


def all_comb_Construct(word, bank):
    dp = [None] * (len(word) + 1)
    dp[0] = [[]]  # Initialize with an empty list of lists
    for i in range(len(dp)):
        if dp[i] is not None:
            for j in range(len(bank)):
                if word[i:].startswith(bank[j]):
                    if dp[i + len(bank[j])] is None:
                        dp[i + len(bank[j])] = []
                    tmp = [com + [bank[j]] for com in dp[i]]
                    dp[i + len(bank[j])].extend(tmp)
    return dp[-1]


def house_robber(arr):
    memo = {}

    def helper(i):
        if i >= len(arr):
            return 0
        if i in memo:
            return memo[i]
        one = arr[i] + helper(i+2)
        two = helper(i+1)
        memo[i] = max(one, two)
        return memo[i]

    return helper(0)


def house_robber_opti(arr):
    one, two = 0, 0
    for i in nums:
        one, two = two, max(one+i, two)
    return max(one, two)


def house_robber_tab(arr):
    dp = [0]*len(arr)
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    for i in range(2, len(arr)):
        dp[i] = max(dp[i-2]+arr[i], dp[i-1])
    return max(dp[-1], dp[-2])


nums = [1, 2, 3, 1]


def max_product(arr):
    res = nums[0]
    curMin, curMax = 1, 1

    for n in nums:
        tmp = curMax * n
        curMax = max(tmp, n * curMin, n)
        curMin = min(tmp, n * curMin, n)
        res = max(res, curMax)
    return res


def max_product_tab(arr):
    res = nums[0]
    n = len(nums)
    dp = [[nums[i], nums[i]]for i in range(n)]  # min, max

    for i in range(1, n):
        tmp = [nums[i], dp[i - 1][0] * nums[i], dp[i - 1][1] * nums[i]]

        dp[i][0] = max(tmp)
        dp[i][1] = min(tmp)

        res = max(res, dp[i][0])

    return res



def tribonacci_tab(self, n: int) -> int:
    if n == 0:
        return 0
    if n <= 2:
        return 1
    dp = [0]*(n+1)
    dp[0] = 0
    dp[1] = 1
    dp[2] = 1
    for i in range(3, n+1):
        dp[i] = dp[i-1]+dp[i-2]+dp[i-3]
    return dp[n]


def tribonacci(n):
    if n == 0:
        return 0
    if n <= 2:
        return 1
    one = 0
    two = 1
    three = 1
    for i in range(2, n):
        rs = one + two + three
        three, two, one = rs, three, two
    return rs


def decode_ways(n):  # how many ways to decode a string like "11106" to "AAJF"((1 1 10 6)) or "KJF"  ((11 10 6))
    memo = {}
    def helper(i):
        if i in memo:
            return memo[i]
        if i == len(n):
            return 1
        if n[i] == '0':
            return 0
        one = helper(i+1)
        two = 0
        if i+1 < len(n) and int(n[i:i+2]) <= 26:
            two = helper(i+2)
        memo[i] = one + two
        return memo[i]

    return helper(0)


def decode_ways_tab(n):
    dp = [0]*(len(n)+1)
    dp[-1] = 1

    for i in range(len(n)-1, -1, -1):
        if n[i] != '0':
            dp[i] = dp[i+1]
        if i+1 < len(n) and int(n[i:i+2]) <= 26:
            dp[i] += dp[i+2]

    return dp[0]


# can partition an array to 2 arrays with the same sum
def can_partition_recursion(arr):

    def helper(i, one, two):

        if i == len(arr):
            return one == two
        return helper(i+1, one+arr[i], two) or helper(i+1, one, two + arr[i])

    return helper(0, 0, 0)


def can_partition_tab(arr):
    s = 0
    for i in arr:
        s += i

    if s % 2 != 0:
        return False
    s = s//2

    dp = [False]*(s+1)
    dp[0] = True
    for num in arr:
        for i in range(s, num - 1, -1):
            if dp[i - num]:  # If the subset with sum (i - num) is possible,
                dp[i] = True  # then subset with sum i is also possible.

    return dp[-1]


def stonegame(piles, alice, bob):
    if len(piles) == 0:
        return alice > bob

    l = piles.pop(0)
    r = piles.pop()
    return stonegame(piles, alice + l, bob + r) or stonegame(piles, alice + r, bob + l)


def change(coins, amount):  # 518
    dp = [0]*(amount+1)
    dp[0] = 1
    for coin in coins:
        for i in range(coin, amount+1):
            dp[i] += dp[i-coin]
    return dp[-1]


s = "aacaba"


def numSplites(s):
    dp = [0] * (len(s))
    left = set()
    right = set()
    for i in range(len(s)-1):
        left.add(s[i])
        dp[i] = len(left)
    rs = 0
    for i in range(len(s)-1, 0, -1):
        right.add(s[i])
        if len(right) == dp[i-1]:
            rs += 1
    return rs


def climbing_stairs(n):  # 70
    dp = [0]*(n+1)
    dp[0] = 1
    for i in range(n+1):
        if i+1 < n+1:
            dp[i+1] += dp[i]
        if i+2 < n+1:
            dp[i+2] += dp[i]
    return dp


def climbing_stairs_opt(n):
    one = 1
    two = 1
    for i in range(2, n+1):
        two, one = two + one, two
    return two


def cherrypickup(grid):  # n1462 on leetcode
    r = len(grid)
    c = len(grid[0])
    movs = [1, -1, 0]
    memo = {}

    def dfs(one, two, y):
        if one == c or two == c or y == r or one < 0 or two < 0:
            return 0
        if (one, two, y) in memo:
            return memo[(one, two, y)]
        cheries = (grid[y][one] + grid[y][two]) if two != one else grid[y][one]
        max_cheries = 0
        for mov in movs:
            for jmov in movs:
                max_cheries = max(max_cheries, dfs(
                    one + mov, two + jmov, y + 1))
        memo[(one, two, y)] = max_cheries + cheries
        return memo[(one, two, y)]
    return dfs(0, c-1, 0)


# for every chr we check if it is the center of a palindrome
def longest_palindromic_string(s):
    # the seconde case of the even palindroms like 'abbd' -> bb
    res = ""

    def check(l, r):
        nonlocal res
        while l >= 0 and r < len(s) and s[l] == s[r]:
            if (r-l+1) > len(res):
                res = s[l:r+1]
            l -= 1
            r += 1
    for i in range(len(s)):
        l, r = i, i
        check(l, r)
        l, r = i, i+1
        check(l, r)
    return res


def count_palindroms(s):  # for every chr we check if it is the center of a palindrome
    # the seconde case of the even palindroms like 'abbd' -> bb
    res = 0

    def check(l, r):
        count = 0
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1
            r += 1
            count += 1
        return count

    for i in range(len(s)):
        l, r = i, i
        res += check(l, r)
        l, r = i, i+1
        res += check(l, r)
    return res


def canCross(stones):  # leetcode n403 , correct solution but it returns time limit exceeded in leetcode
    # adding @cache decorator will make it acceptable in leetcode

    def backtrack(i, k):
        if i == len(stones) - 1:
            return True
        ans = False

        for ind in range(i+1, len(stones)):
            if stones[ind] > stones[i] + k+1:
                break
            for j in range(-1, 2):
                if stones[ind] == stones[i] + k+j:
                    ans = backtrack(ind, k+j) or ans
        return ans

    if stones[1] > stones[0] + 1:
        return False

    return backtrack(1, 1)


def sub_sum_k(nums, k):
    prefix_sum = 0
    count = 0
    sum_count = {0: 1}

    for num in nums:
        prefix_sum += num
        x = prefix_sum - k
        if x in sum_count:
            count += sum_count[x]

        if prefix_sum in sum_count:
            sum_count[prefix_sum] += 1
        else:
            sum_count[prefix_sum] = 1

    return count
    

def minimum_pen(customers):  # n 2383 in leetcode n
    # not a dp problem
    arr = [[0, 0] for i in range(len(customers)+1)]
    pen = 0
    for i in range(len(customers)):
        arr[i][0] = pen
        if customers[i] == 'N':
            pen += 1
    arr[-1][0] = pen

    pen = 0
    for i in range(len(customers)-1, -1, -1):
        if customers[i] == 'Y':
            pen += 1
        arr[i][1] = pen
    ans = -1
    tmp = float('inf')
    for i in range(len(arr)):
        if sum(arr[i]) < tmp:
            tmp = sum(arr[i])
            ans = i
    return ans


def minimum_ticket_cost(days, costs):
    memo = {}

    def dp(i):
        if i >= len(days):
            return 0

        if i in memo:
            return memo[i]

        # if we buy one day pass we will jump to the next day directly
        cost1 = costs[0] + dp(i + 1)
        next_7 = skip(i, 7)
        cost7 = costs[1] + dp(next_7)
        next_30 = skip(i, 30)
        cost30 = costs[2] + dp(next_30)

        memo[i] = min(cost1, cost7, cost30)

        return memo[i]

    def skip(i, duration):
        ind = i
        while i < len(days) and days[i] < days[ind] + duration:
            i += 1
        return i

    return dp(0)


def unique_paths_two(m, n):
    dp = [[0 for i in range(m)] for i in range(n)]
    for i in range(len(dp)):
        for j in range(len(dp[0])):
            if i == 0 or j == 0:
                dp[i][j] = 1
            else:
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[-1][-1]


def out_bound_paths(m, n, maxMove, startRow, startColumn):  # n576 on leetcode
    MOD = 1000000007
    memo = {}
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def dfs(i, j, moves):
        if i < 0 or i >= m or j < 0 or j >= n:
            return 1
        if moves == 0:
            return 0
        if (i, j, moves) in memo:
            return memo[(i, j, moves)]

        total_paths = 0

        for dx, dy in moves:
            new_i, new_j = i + dx, j + dy
            total_paths += dfs(new_i, new_j, moves - 1)
            total_paths %= MOD

        memo[(i, j, moves)] = total_paths
        return total_paths

    return dfs(startRow, startColumn, maxMove)


def e(arr , k):
    newarr = sorted(arr)
    if arr == newarr:
        return True
    dic = {}
    for i in range(len(arr)):
        n = arr[i]
        if n in dic:
            dic[n].append(i)
        else:
            dic[n] = [i]
    
    for i in range(len(newarr)):
        n = newarr[i]
        ans = False
        j=0
        while dic[n] and j < len(dic[n]):
            if abs(i - dic[n][j]) <= k :
                ans = True
                dic[n].pop(j)
            j+=1
        if not ans:
            return False
    return True

def max_matrix_sum(mat):
    r= len(mat)
    c= len(mat[0])
    negative_num = 0
    min_negative = -float("inf")
    for i in range(r):
        for j in range(c):
            if mat[i][j] < 0:
                negative_num += 1
                min_negative = max(min_negative , mat[i][j])
    if negative_num % 2 == 0:
        return sum(mat)
    else:
        return sum(mat) - abs(min_negative)


def divideArray(nums, k: int):
    nums = sorted(nums)
    i = 0
    print(nums)
    res = []
    while i < len(nums):
        tmp = []

        for j in range(3):
            tmp.append(nums[i+j])
        print(tmp)
        if tmp[-1] - tmp[0] > k :
            print(f"the error array {tmp}")
            return []
        res.append(tmp)
        i+= 3
    return res

nums = [1,3,4,8,7,9,3,5,1]
k = 2
print(divideArray(nums ,k))

