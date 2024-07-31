#Maximum Points You Can Obtain from Cards

def solve( arr, k):
    if k==len(arr):
        return sum(arr)
    l = 0 
    summ = 0
    for i in range(k):
        summ+=arr[-1-i]
    max_sum = summ
    for r in range(len(arr)-k,len(arr)):
        summ = summ - arr[r] + arr[l]
        l+=1
        max_sum = max(summ , max_sum)

    return max_sum

cardPoints = [1,2,3,4,5,6,1]
k = 3


def groupThePeople(groupSizes):
    dic = {i : [] for i in range(len(groupSizes)+1)}
    for i in range(len(groupSizes)):
        dic[groupSizes[i]].append(i)
    rs = []
    for k in dic:
        arr = dic[k]
        if arr != []:
            while arr:
                tmp = []
                for i in range(k):
                    tmp.append(arr.pop())
                rs.append(tmp)
    return rs

def can_adhm_win(N, S, X):
    # Adhm wins if S is not divisible by X and N > 1
    if S % X != 0 and N > 1:
        return "Adhm"
    else:
        return "Methat"

# Example usage:
N = 4
S = 3
X = 3
winner = can_adhm_win(N, S, X)
print(f"The winner is: {winner}")




            
