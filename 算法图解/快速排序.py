
def quickSort(arr):
    if len(arr) < 2:
        return arr
    pivot = arr[0]
    # 所有小于基准值的元素组成的数组
    pivotBefore = [i for i in arr[1:] if i <= pivot]
    # 所有大于基准值的元素组成的数组
    pivotAfter = [i for i in arr[1:] if i > pivot]
    return quickSort(pivotBefore) + [pivot] + pivotAfter


arr = [10, 5, 2, 3]
print(quickSort(arr))