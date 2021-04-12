
def bagQuestion(n, m, w, v):
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if j - w[i] >= 0:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - w[i]] + v[i])
            else:
                dp[i][j] = dp[i - 1][j]
    return dp[m][n]