rooms = [[1],[2],[3],[]]
def keysandrooms(rooms):
    rs = set()
    def helper(index , rooms):
        tmp = rooms[index]
        if len(tmp) == 0:
            rs.add(-1)
            return
        for i in tmp:
            if i not in rs:
                rs.add(i)
                helper(i,rooms)
    helper(0,rooms)
    return len(rs)==len(rooms)



def islands(grid):
    visited = set()
    def helper(i,j):
        if i<0 or j<0 or i>=len(grid) or j>=len(grid[0]) or grid[i][j] == '0' or (i,j) in visited:
            return
  
        visited.add((i,j))
        helper(i+1,j)
        helper(i-1,j)
        helper(i,j+1)
        helper(i,j-1)
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j]=='1' and (i,j) not in visited:
                count+=1
                helper(i,j)
    return count

grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]




def maxsizeIsland(grid):
    visited = set()
    def helper(i,j):
        if i<0 or j<0 or i>=len(grid) or j>=len(grid[0]) or grid[i][j] == '0' or (i,j) in visited:
            return 0
  
        visited.add((i,j))
        return 1+helper(i+1,j)+helper(i-1,j)+helper(i,j+1)+helper(i,j-1)
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j]=='1' and (i,j) not in visited:
                count = max(count,helper(i,j))
    return count


board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
def surrounded(nums):
    visited = set()
    def helper(i,j):
        if  i < 0 or j == len(nums[0]) or j<0 or i==len(nums) or (i,j) in visited or nums[i][j] == 'X':
            return
        visited.add((i,j))
        helper(i+1,j)
        helper(i-1,j)   
        helper(i,j+1)
        helper(i,j-1)
    
    for i in range(len(nums)):
        for j in range(len(nums[0])):
            if (i==0 or j==0 or i==len(nums)-1 or j==len(nums[0])-1) and nums[i][j] == 'O':
                helper(i,j)
    for i in range(len(nums)):
        for j in range(len(nums[0])):
            if nums[i][j] == 'O' and (i,j) not in visited:
                nums[i][j] = 'X'

    return nums


grid = [[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]
def islandPerimeter(grid):

    

    visited = set()
    def dfs(i,j):
        if i<0 or i==len(grid) or j<0 or j==len(grid[0]) or grid[i][j] == 0 or (i,j) in visited:
            return 0
        visited.add((i,j))
        perimeter = 0
        if i==0 or grid[i-1][j] == 0:
            perimeter+=1
        if i==len(grid)-1 or grid[i+1][j] == 0:
            perimeter+=1
        if j==0 or grid[i][j-1] == 0:
            perimeter+=1
        if j==len(grid[0])-1 or grid[i][j+1] == 0:
            perimeter+=1
        return perimeter + dfs(i+1,j) + dfs(i-1,j) + dfs(i,j+1) + dfs(i,j-1)
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                return dfs(i,j)



def allPathsSourceTarget(graph: list[list[int]]) -> list[list[int]]:
        rs= []
        visited = set()
        def helper(i , tmp):
            if i == len(graph)-1:
                rs.append(tmp) 
            visited.add(i)
            for item in graph[i]:
                helper(item , tmp + [item])
            visited.remove(i)
        helper(0,[0])
        return rs
graph = [[4,3,1],[3,2,4],[3],[4],[]]



def countConnectedComponents(n:int,  edges: list[list[int]]) -> int:
        adj = {i:[] for i in range(n)}
        for a , b in edges :
            adj[a].append(b)
            adj[b].append(a)

        visited = set()
        def dfs(i):
            visited.add(i)
            for item in adj[i]:
                if item not in visited:
                    dfs(item)
        count = 0
        for i in adj:
            if i not in visited:
                dfs(i)
                count+=1
        return count
edges = [[0,1],[0,2],[1,2],[3,4]]

def validPath(n:int, edges: list[list[int]], start: int, end: int) -> bool:
        adj = {i:[] for i in range(n)}
        for a , b in edges :
            adj[a].append(b)
            adj[b].append(a)

        visited = set()
        def dfs(i):
            visited.add(i)
            for item in adj[i]:
                if item not in visited:
                    dfs(item)
        dfs(start)
        return end in visited

    
numCourses = 4
prerequisites = [[1,0],[2,0],[3,1],[3,2]]
def courses(num , prereq):
    req = {i:[] for i in range(num)}
    for a , b in prereq:
        req[a].append(b)
    visited = set()
    rs = []
    def helper(i):
        if i in visited :
            return 
        if req[i] == []:
            rs.append(i)
        visited.add(i)
        for item in req[i]:
            helper(i)
        visited.remove(i)
        req[i] = []
            
    for a , b in prereq:
        helper(a)
    print(rs)
    return rs





from collections import deque

def nearestExit( maze: list[list[str]], start: list[int]) -> int:
    directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    q = deque()
    rs = []
    finish = False
    q.append(start)
    while q and not finish:
        length = len(q)
        for i in range(length):
            r, c = q.popleft()

            for dr, dc in directions:
                row, col = r + dr, c + dc
                if (row == len(maze)-1 or col == len(maze[0])-1 or row==0 or col == 0) and  maze[row][col] == '.' and [row,col] != start:
                    rs = [row , col]
                    finish = True
                    break
                if (
                    row >= 0 and col >= 0 and row < len(maze)
                    and col < len(maze[0]) and [row,col] != start
                    and maze[row][col] == "." 
                    ):
                        
                        q.append([row, col])
                

    return rs if finish else -1   





def nearestExit_steps(maze: list[list[str]], start: list[int]) -> int:
    directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    queue = deque()
    visited = set()
    a, b = start
    queue.append((a, b, 0))
    while queue:
        r, c, d = queue.popleft()

        for dr, dc in directions:
            row, col = r + dr, c + dc
            if 0 <= row < len(maze) and 0 <= col < len(maze[0]) and maze[row][col] == "." and (row, col) != start:
                if row == 0 or row == len(maze) - 1 or col == 0 or col == len(maze[0]) - 1:
                    return d + 1
                if (row, col) not in visited:
                    queue.append((row, col, d + 1))
                    visited.add((row, col))

    return -1


maze = [["+","+",".","+"],["+",".",".","+"],["+","+","+","."]]
entrance = [1,2]









def course_req(req,num):
    premap = {i: [] for i in range(num)}
        # map each course to : prereq list
    for crs, pre in req:
        premap[crs].append(pre)
    visited = set()
    def helper(i):
        if i in visited:
            return False
        if premap[i]==[]:
            return True
        visited.add(i)
        for j in premap[i]:
            if not helper(j):
                return False
        visited.remove(i)
        premap[i]==[]
        return True
    for i in premap:
        if not helper(i):
            return False
    return True



def walls_gates(maze: list[list[str]], start: list[int]) -> int:
    directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    queue = deque()
    visited = set()
    a, b = start
    queue.append((a, b, 0))
    while queue:
        r, c, d = queue.popleft()

        for dr, dc in directions:
            row, col = r + dr, c + dc
            #-1 is a wall / 0 is the gate(the target) / INF is empty space
            if 0 <= row < len(maze) and 0 <= col < len(maze[0]) and maze[row][col] != -1 and (row, col) != start:
                if maze[row][col] == 0:
                    return d + 1
                if (row, col) not in visited:
                    queue.append((row, col, d + 1))
                    visited.add((row, col))

    return -1


def walls_and_gates(rooms: list[list[int]]):
    # write your code here
    def walls_gates(maze: list[list[str]], start: list[int]) -> int:
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        queue = deque()
        visited = set()
        a, b = start
        queue.append((a, b, 0))
        while queue:
            r, c, d = queue.popleft()

            for dr, dc in directions:
                row, col = r + dr, c + dc
                if 0 <= row < len(maze) and 0 <= col < len(maze[0]) and maze[row][col] != -1 and (row, col) != start:
                    if maze[row][col] == 2147483647 or maze[row][col] > d + 1:
                        maze[row][col] =  d + 1

                    if (row, col) not in visited:
                        queue.append((row, col, d + 1))
                        visited.add((row, col))

        return -1
    
    for i in range(len(rooms)):
        for j in range(len(rooms[0])):
            if rooms[i][j] == 0:
                walls_gates(rooms , [i,j])
                

    return rooms

rooms = [[2147483647,-1,0,2147483647],[2147483647,2147483647,2147483647,-1],[2147483647,-1,2147483647,-1],[0,-1,2147483647,2147483647]]






from collections import deque

def matrix01(rooms: list[list[int]]):
    def bfs(matrix: list[list[int]], start: tuple[int, int]) -> None:
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        queue = deque()
        visited = set()
        a, b = start
        queue.append((a, b, 0))
        visited.add((a, b))
        while queue:
            r, c, d = queue.popleft()
            for dr, dc in directions:
                row, col = r + dr, c + dc
                if 0 <= row < len(matrix) and 0 <= col < len(matrix[0]) and (row, col) not in visited:
                    
                    if matrix[row][col] == 0:
                        matrix[a][b] = d + 1
                        return
                    queue.append((row, col, d + 1))
                    visited.add((row, col))

    for i in range(len(rooms)):
        for j in range(len(rooms[0])):
            if rooms[i][j] == 1:
                bfs(rooms, (i, j))

    return rooms



def countVisitedNodes(edges : list[int]):
    def dfs(node : int , visited : set):
        if node in visited:
            return 0
        visited.add(node)
        return 1 + dfs(edges[node], vis)
    ans = [0] * len(edges)
    counted = set()
    for i in range(len(edges)):
        if i not in counted:
            vis = set()
            ans[i] = dfs(i , vis)
            for j in vis:
                if j not in counted:
                    ans[j] = ans[i]
                    counted.add(j)
    return ans
    
print(countVisitedNodes([1,2,0,0]))






