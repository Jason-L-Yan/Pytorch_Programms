def binSearch(arr, item):
    low = 0;
    high = len(arr) - 1;
    while low <= high:
        mid = int((low + high) / 2);
        guess = arr[mid];
        if guess == item:
            return mid;

        if guess < item:
            low = mid + 1;
        
        if guess > item:
            high = mid - 1;
    return None


my_list = [1, 3, 5, 7, 9]
print(binSearch(my_list, 7))
print(binSearch(my_list, -1))