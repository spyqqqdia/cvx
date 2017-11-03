#-----------------------------------------------------------------------------------------
DEPENDENCIES:|
---------------

resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"


#-----------------------------------------------------------------------------------------
Singular KKT systems:|
-----------------------

The barrier solver increases the parameter t until the duality gap which is known
to satisfy
                gap <= number_of_inequality_Constraints/t

is less than the given tolerance tol. I.e. the termination criterion is

                number_of_inequality_Constraints/t < tol

If this tolerance is set too small t will have to be increased to very large values
for which the KKT system will be singular (up to rounding error).
Regardless of the problem this computational branch is entered always if we do not have
a feasible starting point and have to do a feasibility analysis.

If the computation terminates with singular KKTsystem exception, the solution can be
to set the tolerance lower. In our example problems (infeasible dist_KL problem) this
already occurs for a small number of constraints (on the order of 20) when the tolerance is
set to 1e-12.
The simple feasibility analysis is then not able to determine that the problem is infeasible,
but rather terminates with an unsolvable KKTsystem.
Setting the tolerance lower (to 1e-9) solves this problem and the feasibility analysis
correctly determines that the problem is infeasible.


_Hessians with zero rows_:

In simple fesibility analysis the Hessian of the barrier function can contain zero rows
if the constraints do not depend on some variables (i.e. if some variables are unconstrained).
Currently the Ruiz-equlibration algorithm used in preconditioning cannot handle that and will
throw an IllegalArgumentException.

We deal with that in MatrixUtils.solveWithPreconditioning by setting the diagonal elements
of zero rows equal to one. This is a dangerous hack. Symptom would be Newtons steps which are
not descent directions and fail to be so significantly.
In the current examples this is not observed.

The alternative would be to either let the Ruiz-algo digest the zero row (which remains a zero
row) or to throw a LinSolveExcpetion. In this case we go to the catch block which tries to solve
the system using SVD which can handle this but is much more expensive. This would be the safer 
approach. 
