package cvx

import breeze.linalg.{DenseMatrix, DenseVector, sum}

/** Holder for a sequence of constraints with some additional methods.
  */
abstract class ConstraintSet(val dim:Int, val constraints:Seq[Constraint]) {

    assert(constraints.forall(cnt => cnt.dim == dim))

    /** A point x where all constraints in a set of constraints are defined,
      * i.e. the functions g_j(x) defining the constraints a g_j(x)<=ub_j are all defined.
      * The constraints do not have to be not satisfied at the point x.
      * Will be used as starting point for phase_I feasibility analysis.
      */
    def pointWhereDefined: DenseVector[Double]

    def numConstraints = constraints.size

    /** @return null. */
    def samplePoint = null

    def isSatisfiedStrictlyBy(x:DenseVector[Double]):Boolean = constraints.forall(_.isSatisfiedStrictly(x))

    /** Set of points where the constraints are satisfied strictly. */
    def strictlyFeasibleSet = new ConvexSet(dim) {

        def isInSet(x: DenseVector[Double]) = {

            assert(x.length == dim)
            constraints.forall(cnt => cnt.isSatisfiedStrictly(x))
        }
    }
    /** Turn this constraint set into a constraint set with a fesible point.*/
    def addFeasiblePoint(x0:DenseVector[Double]): ConstraintSet with FeasiblePoint = {

        // check if x0 is strictly feasible
        assert(x0.length==dim,"Feasible point does not have the right dimension")
        assert(constraints.forall(_.isSatisfiedStrictly(x0)))
        new ConstraintSet(dim,constraints) with FeasiblePoint {

            def pointWhereDefined = x0
            def feasiblePoint = x0
        }
    }

    /** Perform a sum of infeasibilities (SOI) feasibility analysis on this constraint set
      * with the feasibility solver [BarrierSolver.phase_I_Solver_SOI].
      *
      * If g_j(x) <= ub_j are the constraints in this constraint set this will
      * minimize the function f(x,s)=s_1+...+s_p subject to s_j>=0 and g_j(x)-s_j <= ub_j.
      *
      * This gives an indication which constraints might be feasible (s_j=0) although the
      * solution (x,s) of the feasibility solver is not guaranteed to yield a point x
      * satisfying all or even many constraints (due to the constraint s_j>=0).
      * However often x does solve many or even all constraints. This is simply a matter of luck.
      *
      * To get around this problem replace the upper bounds ub_j in the constraints with
      * ub_j-epsilon and run the SOI feasibility analysis on this new constraint set.
      * Then the point x in the solution (x,s) of the feasibility solver will satisfy the
      * original constraint g_j(x) <= ub_j strictly whenever s_j<epsilon.
      *
      * @param pars solver parameters for the feasiblity solver ([BarrierSolver.phase_I_Solver_SOI]).
      * @return a feasibility report containing both the vectors x and s above as well as other information.
      */
    def doSOIAnalysis(pars:SolverParams):FeasibilityReport = BarrierSolver.phase_I_Analysis_SOI(this,pars)
}


object ConstraintSet {

    /** Factory function,
      *
      * @param dim common dimension of all constraints in the list constraints
      * @param constraints list of constraints
      * @param x0 point at which all constraints are defined.
      * @return
      */
    def apply(dim:Int, constraints:Seq[Constraint],x0:DenseVector[Double]) =
        new ConstraintSet(dim,constraints){ def pointWhereDefined = x0 }

    //--- Objective function and constraints for basic feasibility analysis ---//

    /** Objective function for basic feasibility analysis, see [boyd], 11.4.1, p579.
      * Recall: one new variable s and the function is f(x,s)=s.
      */
    def phase_I_ObjectiveFunction(cnts:ConstraintSet):ObjectiveFunction = {

        val n = cnts.dim
        new ObjectiveFunction(n+1){

            def valueAt(x:DenseVector[Double]):Double = x(n)
            def gradientAt(x:DenseVector[Double]):DenseVector[Double] =
                DenseVector.tabulate[Double](n+1)(j => if (j<n) 0 else 1 )

            /** Is the zero matrix in dimnsion 1+cnts.dim.*/
            def hessianAt(x:DenseVector[Double]):DenseMatrix[Double] = DenseMatrix.zeros[Double](n+1,n+1)
        }
    }


    /** Set of these constraints for basic phase I analysis, [boyd], 11.4.1, p579.
      * Recall: one new variable s and each constraint g_j(x) <= ub_j replaced with g_j(x)-s <= ub_j.
      *
      * For these we can always find a point at which all these constraints are satisfied.
      * (member function [feasiblePoint]).
      */
    def phase_I_Constraints(cnts:ConstraintSet):ConstraintSet with FeasiblePoint = {

        val n = cnts.dim
        new ConstraintSet(n+1,cnts.constraints.map(cnt => Constraint.phase_I(cnt))) with FeasiblePoint {

            val x0:DenseVector[Double] = cnts.pointWhereDefined
            val y0:Double = cnts.constraints.map(_.valueAt(x0)).max  // max_j g_j(x0)
            def feasiblePoint = DenseVector.tabulate[Double](dim)(j => if (j<dim-1) x0(j) else 1+y0 )
            def pointWhereDefined = feasiblePoint
        }
    }


    //--- Objective function and constraints for sum of infeasibilities feasibility analysis ---//

    /** Objective function for sum of infeasibilities feasibility analysis, see
      * [boyd], 11.4.1, p579.
      * Recall: one new variable s_j for each constraint g_j(x) <= ub_j and the objective
      * function is f(x,s) = s_1+...+s_p, where p is the number of constraints.
      */
    def phase_I_SOI_ObjectiveFunction(cnts:ConstraintSet):ObjectiveFunction = {

        val n = cnts.dim
        val p = cnts.numConstraints
        val dim = n + p

        new ObjectiveFunction(dim){

            def valueAt(u:DenseVector[Double]):Double = sum(u(n until dim))
            def gradientAt(x:DenseVector[Double]):DenseVector[Double] =
                DenseVector.tabulate[Double](dim)(j => if (j<n) 0.0 else 1.0)
            /** Is the zero matrix in dimnsion 1+cnts.dim.*/
            def hessianAt(x:DenseVector[Double]):DenseMatrix[Double] = DenseMatrix.zeros[Double](dim,dim)
        }
    }



    /** Set of constraints cnts modified for sum of infeasibilities phase I analysis with added
      * positivity constraints for the additional variables, see [Constraint..phase_I_SOI_Constraints].
      * [boyd], 11.4.1, p580.
      * Recall: one new variable s_j for each constraint g_j(x) <= ub_j which is then replaced with
      * g_j(x)-s<=ub.
      *
      * For these we can always find a point at which all these constraints are satisfied.
      * The member function _samplePoint_ returns such a point.
      */
    def phase_I_SOI_Constraints(cnts:ConstraintSet):ConstraintSet with FeasiblePoint =  {

        val n = cnts.dim
        val p = cnts.numConstraints
        val cnts_SOI = Constraint.phase_I_SOI_Constraints(n,cnts.constraints)

        val x = cnts.pointWhereDefined

        new ConstraintSet(n+p,cnts_SOI) with FeasiblePoint {

            // new variable u = (x,s) = (x_1,...,x_n,s_1,...,s_p), where n=dim
            // feasibility if s_j > g_j(x) where g_j(x) <= ub_j is the jth original constraint
            def feasiblePoint =
                DenseVector.tabulate[Double](dim)(j => if (j<n) x(j) else 1+cnts.constraints(j-n).valueAt(x))

            def pointWhereDefined = feasiblePoint
        }
    }

}
