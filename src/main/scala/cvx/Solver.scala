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
  */
trait Solver {

    /** Find the location $x$ of the minimum of f=objF over C by the newton method
      * starting from the starting point x0.
      *
      * @param maxIter : maximal number of Newton steps computed.
      * @param alpha   : line search descent factor
      * @param beta    : line search backtrack factor
      * @param tol     : termination as soon as both the norm of the gradient is less than tol and
      *                the Newton decrement satisfies $l=\lambda(x)$ satisfies $l^2/2 < tol$.
      *                Recall that $l^2/2$ indicates the distance of f(x) from the optimal value.
      * @param delta   : if hessF(x) is close to singular, then the regularization hessF(x)+delta*I
      *                is used to compute the Newton step
      *                (this can be interpreted as restricting the step to a trust region,
      *                See docs/cvx_notes.tex, section Regularization).
      *
      *                Distance from singularity of the Hessian will be determined from the size of the smallest
      *                diagonal element of the Cholesky factor hessF(x)=LL'.
      *                If this is smaller than sqrt(delta), the regularization will be applied.
      *
      * @return Solution object (minimizer with additional info.
      */
    def solve(maxIter: Int, alpha: Double, beta: Double, tol: Double, delta: Double):Solution


}

