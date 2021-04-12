flag = 1
while True:
    try:
        # s = []
        # s = s.append(input())
        # print(s)
        print('中文')
        s = list(input())
        if len(s) == 0:
            flag = 0
        elif len(s) % 8 != 0:
            m = 8 - (len(s) % 8)
            for i in range(m):
                s.append(0)
        if flag == 1:
            for i in range(8, len(s) + 1, 8):
                for j in range(i - 8, i):
                    print(s[j], end='')
                print()
    except:
        break;


# flag = 1
# while True:
#     # s = []
#     # s = s.append(input())
#     # print(s)
#     s = list(input())
#     if len(s) == 0:
#         flag = 0
#     elif len(s) % 8 != 0:
#         m = 8 - (len(s) % 8)
#         for i in range(m):
#             s.append(0)
#     if flag == 1:
#         for i in range(8, len(s) + 1, 8):
#             for j in range(i - 8, i):
#                 print(s[j], end='')
#             print()        
        