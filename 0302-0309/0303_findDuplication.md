# Find the duplicated number in a given array

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

**The intersection point is not the cycle entrance in the general case. **

*2d(tortoise) = d(hare)* == *2(F+a) = F + nC + a* where *n* is some integer. 

**Hence the coordinate of the intersection point is *F+a = nC* **







