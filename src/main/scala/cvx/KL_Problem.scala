/**
  * Created by oar on 10.10.17.
  */
package cvx


/** Optimization problem where the Kullback-Leibler distance d_KL(x,p) from a discrete
  * uniform distribution p on Omega={1,2,...,n} (p_j=1/n; j=1,2,...,n) is minimized
  * subject to expectation constraints.
  *
  * Here the variable x denotes another discrete probability distribution on the set
  * Omega. Hence we always have the constraints
  *     x_j>0 and sum(x_j)=1.
  * If one of the x_j=0, then d_KL(x,p)=+oo and we will rule this case out.
  * Since the p_j are all equal the KL-distance
  *
  *        d_KL(x,p) = sum_jp_j\log(p_j/x_j) = c-sum_jp_j\log(x_j)
  *                  = c-sum_j\log(x_j)/n
  *
  * (where c is a constant) is symmetric in the variables x_j.
  * Thus, if the constraints also have some symmetry in the variables x_j the solution must
  * have at least that symmetry. This allows us to compute analytic solutions in certain special
  * cases.
  *
  * In other cases probabilistic inequalities can be used to show that a problem is infeasible.
  * Thus these types of problems are a good source of test problems with known answers.
  *
  */
class KL_Problem(
  val n:Int, override val id:String, val constraints:Seq[Constraint],
  override val solver:Solver, override val logger:Logger
)
extends OptimizationProblem(id,Dist_KL(n),solver,logger)
