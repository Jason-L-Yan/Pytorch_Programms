import bisect


def max_order(lists):
    list_num = []
    list_max = []
    for i in lists:
        local = bisect.bisect_left(list_num, i)
        if local == len(list_num):
            list_num.append(i)
            list_max.append(local + 1)
        else:
            list_num[local] = i
            list_max.append(local + 1)
    return list_max


while True:
    try:
        people_num = 8
        height_list = [186, 186, 150, 200, 160, 130, 197, 200]
        # people_num = int(input())
        # height_list = list(map(int, input().split()))
        result_1 = max_order(height_list)
        print('result_1: ', result_1)
        result_2 = max_order(height_list[::-1])[::-1]
        print('result_2: ', result_2)
        print(people_num - max(map(sum, zip(result_1, result_2))) + 1)
    except BaseException:
        # print("fault line is", er.__traceback__.tb_lineno)
        break