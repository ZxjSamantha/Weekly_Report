# Find the duplicated number in a given array

## Related: 

[Find All Duplicates in an Array](https://leetcode.com/problems/find-all-duplicates-in-an-array/)

[Remove Duplicates from Sorted Array](https://leetcode.com/problems/remove-duplicates-from-sorted-array/)

## Approach 1: Sorting 

```
class SolutionSorting():
    def findDuplicate(self, nums):
        n = len(nums)
        # null case 0: invalid input
        if(nums == None or len(nums) <= 0): return 0

        # null case 1: invalid element
        for num in nums:
            if num < 0 or num > n-1:
                return 0

        nums.sort()

        for i in range(n):
            if nums[i] == nums[i-1]:
                return nums[i]
```

## Approach 2: Set

```
class SolutionSet():
    def findDuplicate(self, nums):
        visited = set()

        n = len(nums)
        # null case 0: invalid input
        if(nums == None or len(nums) <= 0): return 0

        # null case 1: invalid element
        for num in nums:
            if num < 0 or num > n-1:
                return 0

        for num in nums:
            if num in visited:
                return num
            visited.add(num)
```

## Approach 3: Floyd's Tortoise and Hare (Cycle Detection) 

[Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/solution/)

Given a linked list, return the node where the cycle begins. 

1. Where does the cycle come from?

`f(x) = nums[x]` to construct the sequence: `x, nums[x], nums[nums[x]], ...`

Each new element in the sequence is an element in nums at the index of the previous element. 

The cycle appears because nums contains duplicates. The duplicate node is a cycle entrance. 

test case 1: `[2, 6, 4, 1, 3, 1, 5]` 

The constructed sequence: 2->4->3->**1->6->5->1**

test case 2: `[2, 5, 9, 6, 9, 3, 8, 9, 7, 1]`

The constructed sequence: 2->**9->1->5->3->6->8->7->9**

Problem: Find the entrance of the cycle. 

[Floyd's algorithm](https://en.wikipedia.org/wiki/Cycle_detection#Floyd.27s_Tortoise_and_Hare)

2 phases and 2 pointers

### Phase 1:

`hare = nums[nums[hare]]` is twice as fast as `tortoise = nums[tortoise]`
Hare goes fast and it would be the first one who enters the cycle and starts to run around the cycle. 

**The intersection point is not the cycle entrance in the general case.**

*2d(tortoise) = d(hare)* == *2(F+a) = F + nC + a* where *n* is some integer. 

**Hence the coordinate of the intersection point is *F+a = nC* **

### Phase 2: 

We give the tortoise a second chance by slowing down the hare, so that it now moves with the speed of tortoise:

`tortoise = nums[tortoise], hare = nums[hare]`

```
class SolutionFloyd:
    def findDuplicate(self, nums):
        # Find the intersection point of the two runners
        tortoise = hare = nums[0]
        while True:
            tortoise = nums[tortoise] # nums[0]
            hare = nums[nums[hare]] # nums[nums[nums[0]]]
            if tortoise == hare:
                break
        
        # Find the "entrance" to the cycle
        tortoise = nums[0]
        while tortise != hare:
            tortoise = nums[tortise] # F+a = nC 
            hare = nums[hare]

        return hare
```


```
def floyd(f, x0):
    # Main phase of algorithm: finding a repetition x_i = x_2i.
    # The hare moves twice as quickly as the tortoise and
    # the distance between them increases by 1 at each step.
    # Eventually they will both be inside the cycle and then,
    # at some point, the distance between them will be
    # divisible by the period λ.
    tortoise = f(x0) # f(x0) is the element/node next to x0.
    hare = f(f(x0))
    while tortoise != hare:
        tortoise = f(tortoise)
        hare = f(f(hare))
  
    # At this point the tortoise position, ν, which is also equal
    # to the distance between hare and tortoise, is divisible by
    # the period λ. So hare moving in circle one step at a time, 
    # and tortoise (reset to x0) moving towards the circle, will 
    # intersect at the beginning of the circle. Because the 
    # distance between them is constant at 2ν, a multiple of λ,
    # they will agree as soon as the tortoise reaches index μ.

    # Find the position μ of first repetition.    
    mu = 0
    tortoise = x0
    while tortoise != hare:
        tortoise = f(tortoise)
        hare = f(hare)   # Hare and tortoise move at same speed
        mu += 1
 
    # Find the length of the shortest cycle starting from x_μ
    # The hare moves one step at a time while tortoise is still.
    # lam is incremented until λ is found.
    lam = 1
    hare = f(tortoise)
    while tortoise != hare:
        hare = f(hare)
        lam += 1
 
    return lam, mu
```






