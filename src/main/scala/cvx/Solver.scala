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

  def startingPoint:DenseVector[Double]
  /** @return Solution object (minimizer with additional info.
    */
  def solve(debugLevel:Int=0):Solution
  def dim = startingPoint.length
}

/**
  * @param maxIter maximal number of Newton steps computed.
  * @param alpha   line search descent factor
  * @param beta    line search backtrack factor
  * @param tol     termination as soon as both the norm of the gradient is less than tol and
  *                the Newton decrement or duality gap is less than tol.
  *                Recall that $l*l/2$ indicates the distance of f(x) from the optimal value.
  * @param tolEqSolve tolerance in the solution of the KKT system.
  *                   For an equation of the form Mw=q the condition has the form
  *                   ||Mw-q||<tolEqSolve*M.rows. If this is not satisfied, a LInSolveException is
  *                   thrown which then leads to further attempts ending up with an attempted
  *                   solution via SVD and regularization. If this also fails we are dead.
  * @param tolFeas  tolerance in the inequality constraints, g_i(x) < ub_i + tolFeas is accepted.
  * @param delta   : if hessF(x) is close to singular, then the regularization hessF(x)+delta*I
  *                is used to compute the Newton step
  *                (this can be interpreted as restricting the step to a trust region,
  *                See docs/cvx_notes.tex, section Regularization).
  *
  *                Distance from singularity of the Hessian will be determined from the size of the smallest
  *                diagonal element of the Cholesky factor hessF(x)=LL'.
  *                If this is smaller than sqrt(delta), the regularization will be applied.
  */
case class SolverParams(
                             maxIter:Int,
                             alpha:Double,
                             beta:Double,
                             tol:Double,
                             tolEqSolve:Double,
                             tolFeas:Double,
                             delta:Double
                       )
