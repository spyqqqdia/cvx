package cvx

import breeze.linalg.DenseVector

/** Created by oar on 2017-12-21.
  *
  * @param maxIter maximal number of Newton steps computed.
  * @param alpha   line search descent factor
  * @param beta    line search backtrack factor
  * @param tolSolver     tolerance for norm of gradient, Newton decrement and
  *                      duality and equality gap.
  *                Recall that $l*l/2$ indicates the distance of f(x) from the optimal value.
  * @param tolEqSolve tolerance (relative size of error) in the solution of the KKT system.
  * @param tolFeas  tolerance for the inequality constraints, g_i(x) < ub_i + tolFeas is accepted.
  * @param delta    if the PD matrix H is close to singular we will regularize
  *                    H -> H + delta*||H||*I
  *                (this can be interpreted as restricting the step to a trust region,
  *                See docs/cvx_notes.tex, section Regularization).
  *
  *                Distance from singularity of H will be determined from the size of the smallest
  *                diagonal element of the Cholesky factor H=LL'.
  *                If this is smaller than sqrt(delta), the regularization will be applied.
  */
case class SolverParams(
                         maxIter   :Int,
                         alpha     :Double,
                         beta      :Double,
                         tolSolver :Double,   // for norm gradient, duality and equality gap
                         tolEqSolve:Double,   // for KKT equations
                         tolFeas   :Double,
                         delta     :Double
)
object SolverParams {

  /**
    * @param numSlacks number of slack variables assigned to the inequality
    *  constraints (this is equal to the number of inequality constraints).
    */
  def standardParams(numSlacks:Int):SolverParams = {

    // solver parameters
    val maxIter = 1000          // max number of Newton steps computed
    val alpha = 0.04            // line search descent factor
    val beta = 0.8              // line search backtrack factor
    val tolSolver = 1e-8        // tolerance for norm of gradient, duality gap
    val tolEqSolve = 1e-1       // tolerance in the solution of the KKT system
    val tolFeas = 1e-7          // tolerance in inequality and equality constraints
    val delta = 1e-6            // regularization A -> A+delta*I if ill conditioned
    SolverParams(maxIter,alpha,beta,tolSolver,tolEqSolve,tolFeas,delta)
  }
}