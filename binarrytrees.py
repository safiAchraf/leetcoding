class TreeNode:
     def __init__(self, val=0, left=None, right=None):
         self.val = val
         self.left = left
         self.right = right

def rightSideView(root) -> list[int]:
    rs = []
    def fun(root,n):
        if not root:
            return
        i = 0
        r = root
        while r:
            print(r.val)
            if i>=n:
                print(r.val)
                rs.append(r.val)
            i+=1
            if r.right:
                r = r.right
            else:
                r = r.left
        fun(root.left,i-1)


    fun(root,0)
    return rs
                
             

root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.right = TreeNode(5)
root.left.right.right = TreeNode(6)
root.right.right = TreeNode(4)





def levelOrder(root):
    rs = []
    def helper(root,i):
        if not root :
            return
        if len(rs)<=i:
            rs.append([root.val])
        else:
            rs[i].append(root.val)
        helper(root.left , i+1)
        helper(root.right , i+1)
    helper(root , 0)
    return rs










def minDepth(root):
    if not root:
        return 0
    if not root.left and not root.right:
        return 1
    left = minDepth(root.left) + 1
    right = minDepth(root.right) + 1
    if not root.left:
        return right
    if not root.right:
        return left
    return min(left, right)





newroot = TreeNode(10)
newroot.left = TreeNode(5)
newroot.right = TreeNode(-3)
newroot.left.left = TreeNode(3)
newroot.left.right = TreeNode(2)
newroot.right.right = TreeNode(11)
newroot.left.left.left = TreeNode(3)
newroot.left.left.right = TreeNode(-2)
newroot.left.right.right = TreeNode(1)


class Solution:
    def pathSum(self, root: [TreeNode], targetSum: int) -> int:
        self.rs = 0
        def fun(root,cur):
            if not root:
                return
            if cur+root.val == targetSum:
                self.rs+=1
            fun(root.left,cur+root.val)
            fun(root.right,cur+root.val)
        def helper(root):
            if not root:
                return
            fun(root,0)
            helper(root.left)
            helper(root.right)
        helper(root)
        return self.rs
    def pathsumOPT(self , root , targetSum):
        self.rs = 0
        dic = {0:1}
        def dfs(root , total):
            if root:
                total += root.val
                self.rs += dic.get(total - targetSum , 0)
                dic[total] = dic.get(total , 0) + 1
                dfs(root.left , total)
                dfs(root.right , total)
                dic[total] -= 1
        dfs(root , 0)
        return self.rs



theroot = TreeNode(3)
theroot.left = TreeNode(0)
theroot.right = TreeNode(4)
theroot.left.right = TreeNode(2)
theroot.left.right.left = TreeNode(1)
theroot.right = TreeNode(4)

def trimbst(root , low , high):
    if not root:
        return None
    if  root.val < low:
        return trimbst(root.right , low , high)
    if root.val > high:
        return trimbst(root.left , low , high)
    root.left = trimbst(root.left , low , high)
    root.right = trimbst(root.right , low , high)
    return root
    



def whereTheBallEnds():
    i = 65
    j = 1
    ri = -1
    rj = 1
    lastBounce = "L"
    while [i,j] != [66,0] and [i,j] != [66,99] and [i,j] != [0,99] and [i,j] != [0,0]:
        if i == 0 :
            ri = 1
            rj = 1 if lastBounce == 'L' else -1
            lastBounce = "U"
            print(i , j)
        if j == 99:
            ri = 1 if lastBounce == "U" else -1
            rj = -1
            print(i , j)
            lastBounce = "R"
        if i == 66:
            ri = -1
            rj = -1 if lastBounce == "R" else 1
            print(i , j)
            lastBounce = "D"
        if j == 0:
            ri = -1 if lastBounce == "D" else 1
            rj = 1
            print(i , j)
            lastBounce = "L"
        
        i += ri
        j += rj
        
        

    print(i , j)

