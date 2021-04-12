# def func():
#     num = int(input())

#     def getprime(n):
#         i = 2
#         while i * i <= n and n >= i:
#             while n % i == 0:
#                 n = n // i
#                 print(i, end=" ")
#             i = i + 1
#         if n - 1:
#             print(n, end=" ")
#     getprime(num)


# if __name__ == "__main__":
#     func()


# n = int(input())
# a= []
# while n > 0:
#     a.append(n % 2)
#     n = int(n / 2)
    
# print(a[::-1])


# def frogJump(path):
    
#     length = len(path)
#     f = [False] * length
#     for i in range(1, length):
#         f[0] = True
#         for j in range(i):
#             if f[j] and j + path[j] >= i:
#                 f[i] = True
#     return f[length - 1]


# # frogJump([2, 3, 1, 1, 4])
# frogJump([3, 2, 1, 0, 4])


# a = [1, 2, 3]
# b = [1, 2, 3]
# print(a - b)


# a = list(map(int, input().split(';')))
# a = list(input().split(';'))
# print(a)
# b = list(a[0])
# print(b)
# b = ['6', '9']
# c = ord(b[1])
# print(c)

# a = list(input().split(';'))
# xy = [0, 0]
# # m = 0
# for i in a:
#     b = list(i)
#     if len(b) == 3:
#         if ord(b[1]) >= 48 and ord(b[1]) <= 57 and ord(b[2]) >= 48 and ord(b[2]) <= 57:
#             if b[0] == 'A':
#                 xy[0] = xy[0] - (int(b[1]) * 10 + int(b[2]))
#             if b[0] == 'D':
#                 xy[0] = xy[0] + (int(b[1]) * 10 + int(b[2]))
#             if b[0] == 'W':
#                 xy[1] = xy[1] + (int(b[1]) * 10 + int(b[2]))
#             if b[0] == 'S':
#                 xy[1] = xy[1] - (int(b[1]) * 10 + int(b[2]))
#     if len(b) == 2:
#         if ord(b[1]) >= 48 and ord(b[1]) <= 57:
#             if b[0] == 'A':
#                 xy[0] = xy[0] - (int(b[1]))
#             if b[0] == 'D':
#                 xy[0] = xy[0] + (int(b[1]))
#             if b[0] == 'W':
#                 xy[1] = xy[1] + (int(b[1]))
#             if b[0] == 'S':
#                 xy[1] = xy[1] - (int(b[1]))
#     else:
#         continue
# print(xy)


"""
bisect 为可排序序列提供二分查找算法
"""
import bisect

# 使用bisect函数前需要对列表进行排序，否则虽然可以输出数值，但没有意义
a = [1, 5, 6, 10, 9]
a.sort()
print("最初的列表：", a)  # [1, 5, 6, 9, 10]

# bisect.bisect 返回某个数在列表中可以插入的位置，但不会插入该数。
# 如果这个数与列表中的元素相同，则返回元素后面的位置
print("6在列表中可以插入的位置：", bisect.bisect(a, 6))

# bisect.insort 将某个数插入列表
bisect.insort(a, 7)
print("在列表中插入7：", a)

# 处理插入数值与列表元素相同的情况，返回位置，但不会插入该数
# bisect.bisect_left 插入元素左侧位置；bisect.bisect_right 插入元素右侧位置
print("9在列表中可以插入的位置：", bisect.bisect_left(a, 9))
print("9在列表中可以插入的位置：", bisect.bisect_right(a, 9))

# 处理插入数值与列表元素相同的情况，插入该数
# bisect.insort_left 插入元素左侧位置；bisect.insort_right 插入元素右侧位置
bisect.insort_left(a, 9)
print("在列表中插入10：", a)
bisect.insort_right(a, 10)
print("在列表中插入10：", a)





























