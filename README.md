# cvx

The project intends to implement a dense solver for convex minimization along the lines described   
in the book Convex Optimization by Boyd-Vandenberghe.

This is not a state of the art approach to convex minimization. The project is conducted mainly for my own  
amusement. I am primarily interested in Kullback-Leibler distance minimization and this is simple enough to  
be solved by this code (especially via convex duality).  

Use this code only if you cannot use the alternatives described below. It has the following drawbacks:

1. The approach is too general.
2. Suboptimal generalization of linear problems.
3. Uses only dense equation solvers.
4. Scala (Java) is not the preferred approach to numerical computing.

Starting with the last point I have written this code mainly because I like Scala and the Breeze numerical   
library but we do not have many of the facilities which C++ compilers offer (long long double, directional rounding,   
the ability to make use of SIMD extensions).

Be this at it may Scala is an extremely elegant language and Breeze is a great pleasure to use.

Now to point 1. Here we generalize the linear programming problem 

   \[ ? = argmin a'x\ \text{ subject to constraints }b_i'x\leq u_i \text{ and }Ax=b\],

where $a'$ denotes the transpose of the (column) vector $a$ and hence $a'x$ and $b_i'x$ are the dot products  
of the vectors $a$ and $x$ respectively $b_i$ and $x$. by passing from linear to general twice continuously   
differentiable convex functions (objective and constraints). More explicitly we try to solve the following problem:   

   \[ ? = argmin f(x) \text{ subject to constraints }g_i(x)\leq u_i  \text{ and }Ax=b \],

where now $f(x)$ and the $g_i(x)$ are arbitrary twice continuously differentiable convex functions.   
In this generality the problem is not solvable in practice with performance guarantees. Nemirovskii   
[# Nem_1] makes the case that this is not the most fruitful way to generalize the basic linear program
(leading to a ``highly unpredictable process'').

A more fruitful approach is to limit the scope of admissible problems to those for which reliable methods   
of solution  are known. As a first step we generalize the basic linear program by maintaining linearity  
in the objective function and constraints but generalize the order relation $\leq$ to an arbitrary order  
compatible with vector space operations. Such an order is completely defined by the cone of nonnegative  
elements which is why this approach is called _conic_ _programming_, see [#Nem_2], [#Nem_3].

The user of such software need not be aware of this and may still deal with ``general''convex functions  
provided the corresponding optimization problem can be transformed into an equivalent conic problem  
(with the same set of minimizers).

This leads to the approach known as _disciplined_ _convex_ _programming_ (Boyd, Grant, see [#dcvx]).   
Here the user is provided with a library of basic functions and composition rules to construct new functions.   
Functions constructed along these lines can be checked for convexity. Once convexity has been verified,   
the general convex problem above can be transformed into an equivalent standard conic problem which is then   
handed off to a solver which specializes in solving conic problems of this standard structure. 

This ensures reliable solution in fairly high dimensions (thousands to tens of thousands of variables).  
Needless to say this operates on an entirely different level of technical complexity compared to the code in  
this project (both parsing the problem definition and subsequent transformation to a conic problem as well  
as in the specialized solver for the conic problem).

Incredibly such software is available as open source and free of charge. For example in the R-package `cvxr' [#cvxr]  
you can specify your problem in the R-language itself and it will then be parsed to see if it is an admissible   
program. The conic solver behind it is the ECOS solver [#ECOS]. Similar systems exist with Python and Julia frontends.   

Our code uses only dense equation solvers (sparsity of matrices not taken advantage of). With this we loose the ability  
to apply problem transformations (efficiently) which often involve enlarging the problem dimension significantly  
(with the introduction of new variables, so called _slack_ _variables_). This leads to much larger matrices and systems  
of linear equations which however have only little more nonzero entries than the original matrices and are thus sparse.  
A sparse matrix code (highly nontrivial, especially sparsity preserving Cholesky factorization) will then be not much  
slower on this new and bigger problem. A dense code will be significanlty slower (the operations are $O(n^3)$ and once   
the problem no longer fits into the processor cache the slowdown will be worse still).

In the file `docs(cvx_tricks.pdf` you can find some examples of the surprising effects that can be achieved with the  
use of slack variables. For example some problems containing nondifferentiable objective functions can be transformed  
into equivalent problems with an objective function that is linear. It may even be possible to transform a nonconvex  
problem into convex ones.

Summary: use this code if you must or want to use the Scala language. In the files `test/cvx/OptimizationProblems.scala`   
and `test/cvx/SimpleOptimizationProblems.scala` you can find many examples of how to set up an optimzation problem.  
Be pepared that the solution will be a _highly_ _unpredictable_ process. If you have a serious problem, attack it via  
disciplined convex programming [#dcvx].

The best introduction to convex optimization is [#bv_book]. Available from Stanford University as PDF, see link in the  
citation below. Highly recommended are the notes and papers by Ben-Tal and Nemirovskii at Georgia Tech University, see   
citations below.







## Citations

[#Boyd]
Boyd at Stanford, [website](stanford.edu/~boyd/)  

[#Nem]
Nemirovski at Georgia Tech, [website](https://www.isye.gatech.edu/users/arkadi-nemirovski)  

[#bv_book] 
Boyd, Vandenberghe, 
[Convex Optimization](https://www.stanford.edu/~boyd/cvxbook/)

[#Nem_1:2013] 
Arkadi Nemirovskii, 
[Mini-Course on Convex Programming Algorithms](www2.isye.gatech.edu/~nemirovs/BrazilTransparenciesJuly4.pdf)  

[#Nem_2] Arkadi S. Nemirovskii and Michael J. Todd, _Interior-point_ _Methods_ _For_ _Optimization_,  
Acta Numerica (2008), pp. 191–234, [link](https://people.orie.cornell.edu/miketodd/selfconcN.pdf)  
 
[#Nem_3] 
Aharon Ben-Tal and Arkadii Nemirovskii, 
[Lectures on Modern Convex Optimization](https://www2.isye.gatech.edu/~nemirovs/Lect_ModConvOpt.pdf)

[#dcvx]
Disciplined Convex Programming, [website](dcp.stanford.edu/)



[#cvxr] CVXR: An R package for Disciplined Convex Optimization
[link](web.stanford.edu/~boyd/papers/cvxr_paper.html)

[#ECOS]
ECOS solver, [link](https://www.stanford.edu/~boyd/papers/ecos.html)

























