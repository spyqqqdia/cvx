package cvx

import breeze.linalg.{DenseVector, Vector, max}

/** Result of phase_I feasibility analysis.*/
case class FeasibilityReport(

    /** Point which satisfies many constraints.*/
    x0:DenseVector[Double],
    /** Vector of the additional variables s_j (subject ot g_j(x)-s_j <= ub_j)
      * at the optimum of the feasibility solver.
      *
      * In case of a SOI feasibility analysis the s_j are constrained to be non negative
      * and we can classify the constraint g_j(x) <= ub_j to be feasible if s_j is sufficiently
      * close to zero.
      *
      * In case of a simple feasibility analysis there is only one additional variable
      * which is unconstrained and the vector s has dimension one.
      * In this case we only get the following information: all the constraints
      * are strictly feasible if and only if s(0)<0.
      */
    s:Vector[Double],
    /** Flag if x0 satisfies all constraints strictly. */
    isStrictlyFeasible:Boolean,
    /** List of constraints which x0 does not satisfy strictly.*/
    constraintsViolated:Seq[Constraint]
)   {

  /** Report if a feasible point with feasibility tolerance tol has been found.
   */
  def toString(tol:Double):String = {

      if(isStrictlyFeasible)
                  "\nStrictly feasible point found:\n"+x0
      else if (max(s)>tol) {

        val str:String = "\nProblem not feasible within tolerance "+tol+
          "\nFound point x0:\n" + x0 + "\nviolates constraints:\n"
        str+constraintsViolated.foldLeft("")((acc: String, ct) => acc + "\n" + ct.id) + "\n"

      }
      else "\nCannot determine if problem is feasible.\n"
  }
}