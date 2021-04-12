

def coinChange(coin, M):
    
    f = list([0] * (M + 1))
    # print(f)
    n = len(coin)

    f[0] = 0
    for i in range(1, M + 1):
        f[i] = float('inf')
        for j in range(n):
            if i >= coin[j] and f[i - coin[j]] != float('inf'):
                f[i] = min(f[i - coin[j]] + 1, f[i])

    if f[M] == float('inf'):
        f[M] = -1
    return f[M]


coin = [2, 5, 7]
print(coinChange(coin, 9))
