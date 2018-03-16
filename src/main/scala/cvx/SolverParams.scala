package cvx

import breeze.linalg.DenseVector

/** Created by oar on 2017-12-21.
  *
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
  * @param delta    if the PD matrix H is close to singular we will regularize
  *                    H -> H + delta*||H||*I
  *                (this can be interpreted as restricting the step to a trust region,
  *                See docs/cvx_notes.tex, section Regularization).
  *
  *                Distance from singularity of H will be determined from the size of the smallest
  *                diagonal element of the Cholesky factor H=LL'.
  *                If this is smaller than sqrt(delta), the regularization will be applied.
  * @param K       constant in the relaxed objective function h(x,s)=f(x)+Ks of the primal dual solver
  *                with one additional slack variable, see docs/primaldual.pdf.
  * @param vec_K   constant in the relaxed objective function h(x,s)=f(x)+(K dot s) of the primal dual
  *                solver with one additional slack variable for each inequality constraint, see
  *                docs/primaldual.pdf.
  */
case class SolverParams(
                         maxIter:Int,
                         alpha:Double,
                         beta:Double,
                         tol:Double,
                         tolEqSolve:Double,
                         tolFeas:Double,
                         delta:Double,
                         K:Option[Double]=None,
                         vec_K:Option[DenseVector[Double]]=None
)
object SolverParams {


  def standardParams(dim:Int):SolverParams = {

    // solver parameters
    val maxIter = 1000          // max number of Newton steps computed
    val alpha = 0.07            // line search descent factor
    val beta = 0.8              // line search backtrack factor
    val tolSolver = 1e-8        // tolerance for norm of gradient, duality gap
    val tolEqSolve = 1e-2       // tolerance in the solution of the KKT system
    val tolFeas = 1e-7          // tolerance in inequality and equality constraints
    val delta = 1e-7            // regularization A -> A+delta*I if ill conditioned
    val K = 1e7                 // constant K for primal dual solver, see docs/primaldual.pdf
    val vec_K = DenseVector.fill[Double](dim)(K)
    SolverParams(maxIter,alpha,beta,tolSolver,tolEqSolve,tolFeas,delta,Some(K),Some(vec_K))
  }
}