Textbook: [Problem Solving with Python](https://runestone.academy/runestone/books/published/pythonds/index.html)

Algorithm [Code for Lectures](https://algs4.cs.princeton.edu/code/)

Google Dev [Tech Guide with Google](https://techdevguide.withgoogle.com/paths/data-structures-and-algorithms/)

Google Interview [Interview with Google](https://techdevguide.withgoogle.com/paths/interview/)

- Lecture slides
- Download code
- Summary of content 

Data types: stack, queue, bag, union-find, priority quene

Sorting: quicksort, mergesort, heapsort, radix sorts

Searching: BST, red-black BST, hash table

Graphs: BFS, DFS, Prim, Kruskal, Dijkstra

Strings: KMP, regular expressions, TST, Huffman, LZW

Advanced 

---

Problem: Dynamic Connectivity

Implementing the operations: 

Find query: Check if two objects are in the same component

Union command: Replace components containing two objects with their union. 

Goal: Design efficient data structure for union-find

- Number of objects *N* can be huge.

- Number of operations *M* can be huge. 

- Find queries and union commands may be intermixed. 

We need:

public class UF

void union(int p, int q)

boolean connected (int p, int q)

int find(int p)

int count()

Dynamic-connectivity client 

- Read in number of objects *N* from standard input 

Always remember to check our design of API

```
public static void main(String[] args)
{
  int N = StdIn.readInt();
  UF uf = new UF(N);
  
  while (!StdIn.isEmpty()){
    int p = StdIn.readInt();
    int q = StdIn.readInt();
    
    if (!uf.connected(p,q)){
      uf.union(p, q);
      StdOut.printIn(p + " " +q); 
    }
  }
}
```



















