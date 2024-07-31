#include <bits/stdc++.h>
using namespace std;
#define ll long long




const int MOD = 1000000007;
vector<pair<int, int>> moves = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

int dfs(int i, int j, int movesLeft, int m, int n, unordered_map<string, int> &memo)
{
    if (i < 0 || i >= m || j < 0 || j >= n)
    {
        return 1;
    }
    if (movesLeft == 0)
    {
        return 0;
    }
    string key = to_string(i) + "," + to_string(j) + "," + to_string(movesLeft);
    if (memo.find(key) != memo.end())
    {
        return memo[key];
    }

    int totalPaths = 0;

    for (const auto &move : moves)
    {
        int new_i = i + move.first;
        int new_j = j + move.second;
        totalPaths += dfs(new_i, new_j, movesLeft - 1, m, n, memo);
        totalPaths %= MOD;
    }

    memo[key] = totalPaths;

    return totalPaths;
}

int findPaths(int m, int n, int maxMove, int startRow, int startColumn)
{
    unordered_map<string, int> memo;
    return dfs(startRow, startColumn, maxMove, m, n, memo);
}

int coinChange(vector<int> &arr, int target)
{
    vector<int> dp(target + 1, -1);
    dp[0] = 0;
    for (int i = 0; i < target + 1; i++)
    {
        if (dp[i] == -1)
            continue;

        for (int64_t coin : arr)
        {
            int64_t ind = i + coin;
            if (ind < target + 1)
            {
                if (dp[ind] == -1 || dp[ind] > dp[i] + 1)
                {
                    dp[ind] = dp[i] + 1;
                }
            }
        }
    }
    return dp[target] == -1 ? -1 : dp[target];
}

unordered_set<string> visited;

void dfs(vector<vector<char>> &board, int i, int j)
{
    if ((i < 0) || (i >= board.size()) || (j < 0) || (j >= board[0].size()) || ((board[i][j] == 'X')))
    {
        return;
    }
    string key = to_string(i) + "," + to_string(j);
    if (visited.find(key) != visited.end())
    {
        return;
    }

    visited.insert(key);

    dfs(board, i + 1, j);
    dfs(board, i - 1, j);
    dfs(board, i, j + 1);
    dfs(board, i, j - 1);

    return;
}

void surrounded_regions(vector<vector<char>> &board)
{

    for (int i = 0; i < board.size(); i++)
    {
        for (int j = 0; j < board[0].size(); j++)
        {
            if ((i == 0) || (i == board.size() - 1) || (j == 0) || (j == board[0].size() - 1))
            {
                if (board[i][j] == 'O')
                {
                    dfs(board, i, j);
                }
            }
        }
    }
    
    for (int i = 0; i < board.size(); i++)
    {
        for (int j = 0; j < board[0].size(); j++)
        {
            string key = to_string(i) + "," + to_string(j);
            if ((board[i][j] == 'O') && (visited.find(key) == visited.end()))
            {
                board[i][j] = 'X';
            }
        }
    }
}

vector<vector<int>> Pascal(int n)
{
    vector<vector<int>> ans(n, vector<int>());
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            if ((j == 0) || (j == i))
            {
                ans[i].push_back(1);
            }
            else
            {
                ans[i].push_back(ans[i - 1][j] + ans[i - 1][j - 1]);
            }
        }
    }
    return ans;
}

vector<vector<int>> groupThePeople(vector<int> &groupSizes)
{
    vector<vector<int>> rs;
    unordered_map<int, vector<int>> dic;
    for (int i = 0; i <= groupSizes.size(); i++)
    {
        dic[groupSizes[i]].push_back(i);
    }
    for (auto x : dic)
    {
        int key = x.first;
        vector<int> value = x.second;
        int i = 0;
        while (i < value.size())
        {
            vector<int> temp;
            for (int j = 0; j < key; j++)
            {
                temp.push_back(value[i]);
                i++;
            }
            rs.push_back(temp);
            temp.clear();
        }
    }
    return rs;
}

int minRemovals(unordered_map<int, int> &A, unordered_map<int, int> &B, unordered_map<int, int> &C)
{
    int rs = 0;
    unordered_set<int> s;
    for (auto x : A)
    {
        vector<int> tmp;
        if (B.find(x.first) != B.end())
        {
            tmp.push_back(B[x.first]);
        }
        if (C.find(x.first) != C.end())
        {
            tmp.push_back(C[x.first]);
        }
        tmp.push_back(x.second);

        if (tmp.size() > 1)
        {
            sort(tmp.begin(), tmp.end());
            rs += tmp[0] + tmp[1];
        }
        s.insert(x.first);
    }
    for (auto x : B)
    {
        if (s.find(x.first) == s.end())
        {
            vector<int> tmp;
            if (C.find(x.first) != C.end())
            {
                tmp.push_back(C[x.first]);
            }
            tmp.push_back(x.second);
            if (tmp.size() > 1)
            {
                sort(tmp.begin(), tmp.end());
                rs += tmp[0] + tmp[1];
            }
        }
    }
    return rs;
}

void dremove()
{

    int X, Y, Z;
    cin >> X >> Y >> Z;
    unordered_map<int, int> A, B, C;

    for (int i = 0; i < X; i++)
    {
        int a;
        cin >> a;
        A[a]++;
    }

    for (int i = 0; i < Y; i++)
    {
        int b;
        cin >> b;
        B[b]++;
    }

    for (int i = 0; i < Z; i++)
    {
        int c;
        cin >> c;
        C[c]++;
    }

    cout << minRemovals(A, B, C) << endl;
}

string solve(int arr[], int n, int k)
{

    for (int i = 0; i < n; i++)
    {
        int r = i;
        while ((r < n) && (arr[i] >= arr[r]))
        {
            r++;
        }
        if (i + k < r - 1)
        {
            return "No";
        }
    }
    return "Yes";
}

void e()
{
    int n, k;
    cin >> n >> k;
    int arr[n];
    for (int i = 0; i < n; i++)
    {
        cin >> arr[i];
    }
    cout << solve(arr, n, k) << endl;
}

int islands(vector<vector<int>> &arr, int i, int j, int n, int m, vector<vector<bool>> &visited)
{
    if ((i < 0) || (i >= n) || (j < 0) || (j >= m) || (visited[i][j]))
    {
        return 0;
    }
    int current_max = 0;
    if (arr[i][j] > current_max)
    {
        current_max = arr[i][j];
    }
    visited[i][j] = true;

    for (const auto &move : moves)
    {
        int r = i + move.first;
        int c = j + move.second;
        if ((r >= 0) && (r < n) && (c >= 0) && (c < m) && (arr[r][c] <= arr[i][j] + 1))
        {
            current_max = max(islands(arr, r, c, n, m, visited), current_max);
        }
    }
    return current_max;
}

bool helperg(int i, int j, int n, int m, vector<vector<bool>> &grid, vector<vector<bool>> &visited)
{
    if ((i < 0) || (i >= n) || (j < 0) || (j >= m) || (visited[i][j]))
        return false;
    if (i == n - 1)
        return true;
    visited[i][j] = true;
    for (const auto &move : moves)
    {
        int r = i + move.first;
        int c = j + move.second;
        if ((r >= 0) && (r < n) && (c >= 0) && (c < m) && (grid[r][c]))
        {
            if (helperg(r, c, n, m, grid, visited))
            {
                return true;
            }
        }
    }
}

int solveg(int n, int m, int q)
{
    vector<vector<bool>> visited(n, vector<bool>(m, false));
    vector<vector<bool>> grid(n, vector<bool>(m, false));
    int rs = -1;
    int count = 1;
    while (q--)
    {
        int x, y;
        cin >> x >> y;
        x--;
        y--;
        grid[x][y] = true;
        if (x == n - 1 && rs == -1)
        {
            for (int ind = 0; ind < n; ind++)
            {
                if ((grid[0][ind]) && (helperg(0, ind, n, m, grid, visited)))
                {
                    rs = count;
                }
            }
        }
        count++;
    }
    return rs;
}

void file()
{
    freopen("test.in", "r", stdin);
    int n;
    cin >> n;
    while (n--)
    {
        int n, m;
        cin >> n >> m;
        cout << n + m << endl;
    }
}

bool self_desc(string s)
{
    unordered_map<int, int> dic;
    for (int i = 0; i < s.size(); i++)
    {
        dic[s[i] - '0']++;
    }
    int count = 0;
    for (int i = 0; i < s.size(); i++)
    {
        if (dic[count] != s[i] - '0')
        {
            return false;
        }
        count++;
    }
    return true;
}



bool sortbysec(const pair<int, int> &a,const pair<int, int> &b)
{
    return (a.first < b.first);
}
int overlapped(vector<pair<int, int>> &vec)
{
    sort(vec.begin(), vec.end(), sortbysec);
    int count = 0;
    int end = vec[0].second;
    int start = vec[0].first;
    for (int i = 1; i < vec.size(); i++)
    {
        if (vec[i].first < end)
        {
            if (vec[i].second < end) count += vec[i].second - vec[i].first;
            else count +=  vec[i].first - end ;
            end = max(end, vec[i].second);
        }
        else
        {
            end = vec[i].second;
        }
    }
    return count;
}

vector<int> dailyTem(vector<int> &temp){
    stack<int> st;
    for(int i=0; i<temp.size() ; i++){
        while((!st.empty())&&(temp[i] > temp[st.top()])){
            int ind = st.top();
            st.pop();
            temp[ind] = i - ind;
        }
        st.push(i);
    }
    while(!st.empty()){
        temp[st.top()] = 0;
        st.pop();
    }
    return temp;
}

int maxSubarray(vector<int> nums){
    int res = nums[0];
    int maxProduct = 1, minProduct = 1;

    for (auto num : nums){
        int tmp = maxProduct * num;
        maxProduct = max(max(tmp ,  minProduct*num) , num);
        minProduct = min(min(tmp ,  minProduct*num) , num);
        res = max(res,  maxProduct);
    }
    return res;
}

