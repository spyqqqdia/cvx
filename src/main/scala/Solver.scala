package cvx

import breeze.linalg.{DenseMatrix, DenseVector, NotConvergedException, _}

/**
  * Created by oar on 12/4/16.
  *
  * Class to minimize a convex function subject to equality and inequality constraints
  * using gradient and Hessian information
  *
  * In addition there might be an additional abstract constraint $x\in C$, where $C$ is an
  * open convex set. The intention is that $C$ is the full space (thus the constraint $x\in C$
  * vacuous) or it is known that the objective function approaches +oo as x approaches the boundary
  * of $C$.
  *
  * This is the case in the barrier method and can be treated similarly to the case where C
  * is the full space.
  *
  * @param C open convex set implying constraint $x\in C$.
  */
abstract class Solver(val x0:DenseVector[Double], val C:ConvexSet) {

    val n = x0.length
    assert(x0.length==C.dim, "dim(x0)="+n+" != dim(C)="+C.dim)
    assert(C.isInSet(x0),"x0 not in domain C (strictly feasible??).")

    val y0 = gradF(x0)
    val H0 = hessF(x0)
    assert(
        n == H0.cols && n == y0.length,
        "dim(x0)="+n+", hessF(x0).cols="+H0.cols+", dim(gradF(x0))="+y0.length+" not all equal!"
    )

    /** Objective function. */
    def objF(x:DenseVector[Double]):Double
    /** Gradient of objective function. */
    def gradF(x:DenseVector[Double]):DenseVector[Double]
    /** Hessian of objective function, must be positive semidefinite. */
    def hessF(x:DenseVector[Double]):DenseMatrix[Double]


}
