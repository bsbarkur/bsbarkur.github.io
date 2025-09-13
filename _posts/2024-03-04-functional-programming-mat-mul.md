# Programming Matrix multiplication in a functional level programming

So, thanks to one of the connections I met recently named Sasank, I have been getting a healthy dose of lisp and different way of looking at current systems. 

Sasank has been implementing few nifty things which he hopes will solve problems caused by Von Neumann Bottleneck. For more details the reader, is requested to check out his repo at [llama.lisp](https://chsasank.com/llama.lisp/). This post will to distill some of my learnings from perusal of the projects he has been working on in exhaustive style.

## So what is von Neumann Bottleneck ?
If one checks the Turing lecture paper given by Backus of BNF fame, they will understand why current imperative language paradigms do not scale and generalize well to certain mathematical properties. Backus shows with very clear descriptions and bulleted points on the shortcomings of so called Von Neumann programming languages.

The famed landmark lecture by John Backus can be read here. [Can Programming Be Liberated from the von 
Neumann Style? A Functional Style and Its 
Algebra of Programs ](https://cs.wellesley.edu/~cs251/s19/notes/backus-turing-lecture.pdf)

So a von Neumann computer simplistically is nothing but a computer having
1. Central Processing Unit (CPU)
2. A store
3. And a tube that transfers the data between teh store and the CPU.

Backus shows that how an imperative program that has assignment operations causes lots of wastage owing to lot of data fro and to in the tube in the above design. He states that "the assignment statement is the von Neumann bottleneck of programming languages and keeps us thinking 
in word-at-a-time terms in much the same way the 
computer's bottleneck does".

## Defining Inner product in functional level programming style
So inner product is nothing but this formulation

Sasank has written a neat introduction where he shows via python how this form application works for those of us who are wired to think in terms of modern languages [here](https://chsasank.com/llama.lisp/dev/interpreter.html).

So in pythonic code below:

```
## Inner Product
IP = comp(
    insert(add),
    alpha(mul),
    trans
)
```

## comp function
To dig this further this let us see how he explains `comp`.

Comp is defined like this

```
def comp(*fn_list):
    return lambda X: fn_list[0](X) if len(fn_list) == 1 else comp(*fn_list[:-1])(fn_list[-1](X))
```

* When you define a function like comp(*fn_list), the 
before fn_list means "collect all positional arguments into a tuple named fn_list".
* Inside the function, you can use fn_list like a regular tuple, and it will contain all the arguments you passed to the function.
* When you call a function with an argument like *fn_list, it’s doing the opposite: it’s unpacking the tuple (or list) fn_list into separate arguments.

### Lambda function

Let us analyse this: `lambda X: fn_list[0](X)`


The expression `lambda X: fn_list[0](X)` is a lambda function in Python. A lambda function is a small anonymous function that is defined with the lambda keyword, and it can take any number of arguments, but can only have one expression.

Here’s a breakdown of this lambda function:

* lambda is the keyword that starts the definition of the lambda function.
* X is the argument to the lambda function. You can pass a value to the function using this argument.
* : separates the arguments from the body of the lambda function.
* fn_list0 is the body of the lambda function. It calls the first function in fn_list with X as the argument.

This lambda function takes one argument X, and applies the first function in fn_list to X.

### How comp works?

So if the length of the fn_list argument is 1, then we return the lambda expressions detailed above.

Else, we branch to
`comp(*fn_list[:-1])(fn_list[-1](X))`

The expression comp(*fn_list[:-1])(fn_list-1) is part of the recursive function comp shown earlier. This expression is used to create a composition of functions from the list fn_list. 

Here's a breakdown:

* `fn_list[:-1]` gives all elements in fn_list except for the last one. The * operator is used to unpack these elements as separate arguments to the comp function.
* `comp(*fn_list[:-1])` recursively calls the comp function with the unpacked elements, which results in a composition of all functions in fn_list except for the last one.
* `fn_list[-1]` applies the last function in fn_list to X.
* `comp(*fn_list[:-1])(fn_list-1)` applies the composition of all functions except for the last one to the result of applying the last function to X.
In other words, if fn_list is a list of functions [f, g, h], then comp(*fn_list[:-1])(fn_list-1) would be equivalent to `f(g(h(X)))`. This is a way to create a composition of functions.


## Trace back to inner product definition
So, now we understand inner product is composition of these functions:

1. insert(add)
2. alpha(mul)
3. transpose

### Transpose of a matrix
Transpose function implementation is straightforward.


```
def trans(X):
    return [
        [X[n][m] for n in range(len(X))]
        for m in range(len(X[0]))
    ]
```

### Mul and Add functions

The below are also straightfoward.
```
def mul(X):
    return X[0] * X[1]

def add(X):
    return X[0] + X[1]
```


### alpha function
```
def alpha(fn):
    return lambda X: [fn(x) for x in X]
```

This is nothing but applying a function for X as argument to the lambda function.

### insert function

```
def insert(f):
    return lambda X: X[0] if len(X) == 1 else f([X[0], insert(f)(X[1:])])
```

* lambda X:: This defines an anonymous function (a function without a name), which takes one argument X. The X is expected to be a list.

* X[0] if len(X) == 1: If the length of X is 1 (i.e., X contains only one element), the function returns the first (and only) element of X.

* else f([X[0], insert(f)(X[1:])]): If X contains more than one element, the function calls another function f with a list as an argument. This list contains the first element of X and the result of calling the function insert(f) with the rest of X (i.e., X without its first element) as an argument.

### index functions

```
def idx_0(X):
    return X[0]

def idx_1(X):
    return X[1]
```

The indexing functions given a list, X as shown above are simple to understand.

### distl function

```
def distl(X):
    assert len(X) == 2
    return [[X[0], z] for z in X[1]]

```

distl(X), is designed to distribute the first element of a list X with each element in the second element of X, which is expected to be a list itself. 

Here’s a breakdown of how the function works:

* The function takes one argument, X, which is expected to be a list of two elements. The first element can be of any type, and the second element should be a list.
* The assert len(X) == 2 statement ensures that X has exactly two elements. If X has more or fewer than two elements, the function will raise an AssertionError.
* The list comprehension `[[X[0], z] for z in X[1]]` creates a new list. For each element z in the second element of X (which is a list), it creates a new list [X[0], z] and adds this list to the new list.

### distr function

```
def distr(X):
    assert len(X) == 2
    return [[y, X[1]] for y in X[0]]
```

This must be obvious to decipher after reading the above explianation.


### cat function

```
def cat(*fns):
    return lambda X: [fn(X) for fn in fns]
```

This Python function, cat(*fns), is designed to create a new function that applies a list of functions to an input. Here’s a breakdown of how it works:

* The function takes any number of arguments, *fns, which are expected to be functions. The * operator in the function definition allows for a variable number of arguments to be passed.

* It returns a lambda function, which takes one argument X.

* This lambda function applies each function in fns to X and returns a list of the results.

Example:
```
def square(x):
    return x**2

def cube(x):
    return x**3

f = cat(square, cube)
print(f(2))  # prints [4, 8]
```


## understanding execution of mat mul

So, now we can see how it all ties up to implement mat mul, which is the basic building block of all machine learning and deep learning algorithms and model inferencing.

```
MM = comp(
    alpha(alpha (IP)),
    alpha(distl),
    distr,
    cat(idx_0, comp(trans, idx_1))
)
```

cat(idx_0, comp(trans, idx_1)): This function takes a list X as input and returns a new list. 

The first element of the new list is the first element of X (obtained by idx_0(X)), and the second element is the transpose of the second element of X (obtained by comp(trans, idx_1)(X)).

distr: This function takes a list X as input, where X is expected to have exactly two elements. It returns a new list where each element is a list containing an element from the first element of X and the second element of X.

alpha(distl): This function applies the distl function to each element in a list X.

alpha(alpha (IP)): This function applies the IP function twice to each element in a list X.



