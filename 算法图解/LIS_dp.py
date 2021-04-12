def lengthOfLIS(nums):
    dp = [1] * len(nums)
    res = 1  # 记录最大长度
    for i in range(len(nums)):
        for j in range(0, i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
        res = max(res, dp[i])

    return res


nums = [186, 186, 150, 200, 160, 130, 197, 200]
print(lengthOfLIS(nums))
while True:
    try:
        people_num = int(input())
        height_list = list(map(int, input().split()))
        result_1 = max_order(height_list)
        print('result_1: ', result_1)
        result_2 = max_order(height_list[::-1])[::-1]
        print('result_2: ', result_2)
        print(people_num - max(map(sum, zip(result_1, result_2))) + 1)
    except BaseException:
        # print("fault line is", er.__traceback__.tb_lineno)
        break