
class Node():

    """"节点"""
    def __init__(self, elem):
        self.elem = elem
        self.next = None


class SingleLinkList():

    """单链表"""
    def __init__(self, node=None):
        self.__head = node  # __head前加下划线表示私有(private),类外不能调用。
    
    def is_empty(self):
        """链表是否为空"""
        return self.__head == None
    
    def length(self):
        """链表长度,即节点数目"""
        current = self.__head
        count = 0
        while current != None:
            count += 1
            current = current.next
        return count
        
    def travel(self):
        """遍历链表"""
        current = self.__head
        while current != None:
            print(current.elem)
            current = current.next
     
    def add(self, item):
        """链表头部插入节点，头插法"""
        node3 = Node(item)
        node3.next = self.__head
        self.__head = node3
        
    def append(self, item):  # item不是节点，是一个具体的元素
        """尾部插入节点, 尾插法"""
        node2 = Node(item)
        if self.is_empty():
            self.__head = node2
        else:
            current = self.__head
            while current.next != None:  # 当为空列表时，没有current.next，会报错，需要加上判断语句
                current = current.next
            current.next = node2
            
    def insert(self, pos, item):
        """指定位置添加元素
        :param  pos 从0开始
        """
        if pos <= 0:  # 此时默认使用头插法
            self.add(item)
        elif pos > (self.length() - 1):
            self.append(item)
        else:
            pre = self.__head
            node4 = Node(item)
            count = 0
            while count < (pos - 1):
                count += 1
                pre = pre.next
            # 当循环退出后， pre point at pos-1
            node4.next = pre.next
            pre.next = node4
 
    def remove(self, item):
        """删除节点"""
        current = self.__head
        pre = None
        while current != None:
            if current.elem == item:
                # 先判断删除此节点是否是头结点
                # 头结点
                if current == self.__head:  
                    self.__head = current.next
                else:
                    # 如果是尾节点，可以正常处理
                    pre.next = current.next
                break
            else:
                pre = current
                current = current.next

    def search(self, item):
        """查找节点是否存在"""
        current = self.__head
        while current != None:
            if current.elem == item:
                return True
            else:
                current = current.next
        return False


if __name__ == '__main__':
    # node = Node(100) 
    # SingleLinkList(node)
    ll = SingleLinkList()
    print(ll.is_empty())
    print(ll.length())

    ll.append(1)
    print(ll.is_empty())
    print(ll.length())
    
    ll.append(2)
    ll.append(3)
    ll.append(4)
    ll.append(5)
    ll.append(6)
    print('*' * 50)
    ll.travel()

    print('*' * 50)
    ll.add(10)
    ll.travel()
    print('*' * 50)
    ll.insert(-1, 9)
    ll.travel()
    print('*' * 50)
    ll.insert(1, 0)
    ll.travel()
    print('*' * 50)
    ll.insert(12, 100)
    ll.travel()
    print('*' * 50)
    ll.remove(6)
    ll.remove(9)
    ll.remove(100)
    ll.travel()

    

    
      
    

        
