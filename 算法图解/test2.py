# import sys
# from collections import defaultdict
 
# data = map(lambda x:x.split('\\')[-1] ,sys.stdin.readlines())
# # for d in data:
# #     print(d)
# # # print(data)
# errors = defaultdict(int)  # https://www.jianshu.com/p/bbd258f99fd3
# result = list()
 
# for d in data:
#     # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
#     # 注意：该方法只能删除开头或是结尾的字符，不能删除中间部分的字符。
#     name,line = d.strip().split()  
# #     print(name)
#     error = ' '.join([name[-16:],line])
# #     print(error)
#     errors[error] += 1
#     if errors[error] == 1:
#         result.append(error)
        
# for r in result[-8:]:
#     print(r,errors[r])

import sys
data = map(lambda x: int(x.strip()), sys.stdin.readlines())
data = list(data)
number = 0
for d in data[:-1]:
    number = 0
    while True:
        n = d // 3
        number += n
        m = d % 3
        if (m + n) == 2:
            number += 1
            break
        if (m + n) < 2:
            break
        # number += n
        d = n + m
    print(number)