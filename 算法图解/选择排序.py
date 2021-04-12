# 从大到小排序

def findLargest(arr):
    # 初始化最大值等于第一个元素
    largest = arr[0]
    largest_index = 0
    length = len(arr)
    for x in range(0, length):
        if arr[x] > largest:
            largest = arr[x]
            largest_index = x

    return largest_index


def selectSort(arr):
    new_arr = []
    length = len(arr)
    for x in range(0, length):
        largest = arr[findLargest(arr)]
        new_arr.append(largest)
        arr.pop(findLargest(arr))

    return new_arr


print(selectSort([5, 3, 6, 2, 10]))
