# Why modular everything

One of the reasons for the `Embedding`, `KWeight`, and `root` classes and 
functions being separated and passed to the {{RISB}} `Solver` is because these 
are the parts of the algorithm that afford the most meaningful changes. The 
parts that make up the self-consistent loop are also easily accessed through 
the `helper` functions. In the [{{RISB}} loop](../tutorials/self-consistent.md) 
tutorial the self-consistent cycle is even made bare, and a user can easily 
write their own class and implementation. This is also largely the reason for 
using python for most aspects of the implementation, even though Fortran or 
C++ would be faster. Older versions of this code were written entirely in C++.

Doing it the way we have done is in our opinion better than forcing a specific 
immplementation because it allows anyone to easily make changes that are 
particular to their needs. Some people would prefer a black box where they can 
input some parameters and have an answer, like many {{DFT}} codes. There are 
advantages to such an approach, like ease of use and it likely also will run 
faster. But a black box prevents users from actually understanding how 
algorithms work, and prevents them from making meaningful changes and 
contributing their valuable ideas.

One of the goals of our implementation is to be a good teaching tool for 
new students in strongly correlated physics. But also to be an easy to use 
tool to build upon in our own research.