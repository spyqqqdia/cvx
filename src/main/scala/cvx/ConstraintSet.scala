package cvx

import breeze.linalg.{DenseMatrix, DenseVector, norm, sum}

/** Holder for a sequence of constraints with some additional methods.
  */
abstract class ConstraintSet(val dim:Int, val constraints:Seq[Constraint]) {

  self:ConstraintSet =>
  assert(constraints.forall(cnt => cnt.dim == dim))

  /** A point x where all constraints in a set of constraints are defined,
    * i.e. the functions g_j(x) defining the constraints a g_j(x)<=ub_j are all defined.
    * The constraints do not have to be not satisfied at the point x.
    * Will be used as starting point for phase_I feasibility analysis.
    */
  def pointWhereDefined: DenseVector[Double]

  def numConstraints:Int = constraints.size
  def isSatisfiedStrictlyBy(x:DenseVector[Double]):Boolean = constraints.forall(_.isSatisfiedStrictly(x))

  /** Set of points where the constraints are satisfied strictly. */
  def strictlyFeasibleSet = new ConvexSet(dim) {

    def isInSet(x: DenseVector[Double]):Boolean = {

      assert(x.length == dim)
      constraints.forall(cnt => cnt.isSatisfiedStrictly(x))
    }
    def samplePoint = None
  }
  /** Turn this constraint set into a constraint set with a feasible point.*/
  def addFeasiblePoint(x0:DenseVector[Double]): ConstraintSet with FeasiblePoint = {

    // check if x0 is strictly feasible
    assert(x0.length==dim,"Feasible point does not have the right dimension")
    assert(constraints.forall(_.isSatisfiedStrictly(x0)))
    new ConstraintSet(dim,constraints) with FeasiblePoint {

      def pointWhereDefined:DenseVector[Double] = x0
      def feasiblePoint:DenseVector[Double] = x0
    }
  }


  /**---------------------- FEASIBILITY ANALYSIS ---------------------------**/


  /**------------ Objective function and constraints for basic feasibility analysis --------**/

  /** Objective function for basic feasibility analysis, see [boyd], 11.4.1, p579.
    * Recall: one new variable s and the function is f(x,s)=s.
    */
  def phase_I_ObjectiveFunction:ObjectiveFunction = {

    val n = dim
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
  def phase_I_Constraints:ConstraintSet with FeasiblePoint = {

    val n = dim
    new ConstraintSet(1+n,constraints.map(cnt => Constraint.phase_I(cnt))) with FeasiblePoint {

      val x0:DenseVector[Double] = self.pointWhereDefined
      val y0:Double = self.constraints.map(_.valueAt(x0)).max  // max_j g_j(x0)
      def feasiblePoint:DenseVector[Double] = DenseVector.tabulate[Double](1+n)(j => if (j<n) x0(j) else 1+y0 )
      def pointWhereDefined:DenseVector[Double] = feasiblePoint
    }
  }


  //--- Objective function and constraints for sum of infeasibilities feasibility analysis ---//

  /** Objective function for sum of infeasibilities feasibility analysis, see
    * [boyd], 11.4.1, p579.
    * Recall: one new variable s_j for each constraint g_j(x) <= ub_j and the objective
    * function is f(x,s) = s_1+...+s_p, where p is the number of constraints.
    */
  def phase_I_SOI_ObjectiveFunction:ObjectiveFunction = {

    val n = dim
    val p = numConstraints
    val newDim = n + p

    new ObjectiveFunction(newDim){

      def valueAt(u:DenseVector[Double]):Double = sum(u(n until newDim))
      def gradientAt(x:DenseVector[Double]):DenseVector[Double] =
        DenseVector.tabulate[Double](newDim)(j => if (j<n) 0.0 else 1.0)
      /** Is the zero matrix in dimension newDim.*/
      def hessianAt(x:DenseVector[Double]):DenseMatrix[Double] = DenseMatrix.zeros[Double](newDim,newDim)
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
  def phase_I_SOI_Constraints:ConstraintSet with FeasiblePoint =  {

    val n = dim
    val p = numConstraints
    val cnts = constraints
    val x = pointWhereDefined
    val newDim = n + p

    new ConstraintSet(newDim,Constraint.phase_I_SOI_Constraints(n,this.constraints)) with FeasiblePoint {

      // new variable u = (x,s) = (x_1,...,x_n,s_1,...,s_p), where n=dim
      // feasibility if s_j > g_j(x) where g_j(x) <= ub_j is the jth original constraint
      def feasiblePoint:DenseVector[Double] = DenseVector.tabulate[Double](newDim)(
        j => if (j<n) x(j) else 1+cnts(j-n).valueAt(x)
      )
      def pointWhereDefined:DenseVector[Double] = feasiblePoint
    }
  }





  /**--------------------- Solvers for the feasibility problem -------------------------**/


  /** Barrier solver for phase I feasibility analysis according to basic algorithm,
    *  [boyd], 11.4.1, p579.
    *
    * @param eqs optional equality constraint(s) in the form Ax=b.
    * @param pars solver parameters, see [SolverParams].
    */
  private def phase_I_Solver(
    eqs: Option[EqualityConstraint], pars:SolverParams
  ):BarrierSolver = {

    val feasObjF = phase_I_ObjectiveFunction
    val feasCnts = phase_I_Constraints
    val logFilePath = "logs/ConstraintSet_phase_I_log.txt"
    BarrierSolver(feasObjF,feasCnts,eqs,pars,Logger(logFilePath))
  }


  /** Phase I feasibility analysis according to basic algorithm, [boyd], 11.4.1, p579
    *  using a BarrierSolver on the feasibility problem.
    *
    * @param eqs optional equality constraint(s) in the form Ax=b.
    * @param pars solver parameters, see [SolverParams].
    */
  def phase_I_Analysis(
      eqs: Option[EqualityConstraint], pars:SolverParams, debugLevel:Int
  ): FeasibilityReport = {

    if(debugLevel>0) println("\nConstraintSet: doing phase_I_Analysis.")
    Console.flush()
    // map the equalities to the dimension of the phase_I analysis:
    val phase_I_eqs:Option[EqualityConstraint] = eqs.map(eq => eq.phase_I_EqualityConstraint)
    val feasBS = phase_I_Solver(phase_I_eqs,pars)
    val sol = feasBS.solve(debugLevel)

    val w_feas = sol.x                  // w = c(x,s)
    val x_feas = w_feas(0 until dim)    // unclear how many constraints that satisfies, so we check
    val s_feas = w_feas(dim)            // if s_feas < 0, all constraints are strictly satisfied.
    val s = DenseVector[Double](1)
    s(0)=s_feas
    val isStrictlySatisfied = s_feas < 0
    val violatedCnts = constraints.filter(!_.isSatisfiedStrictly(x_feas))

    FeasibilityReport(x_feas,s,isStrictlySatisfied,violatedCnts)
  }


  /** Phase I feasibility analysis according to basic algorithm, [boyd], 11.4.1, p579
    *  using a BarrierSolver on the feasibility problem.
    *  With equality constraints parameterized as x=z0+Fu.
    *  Reports feasible candidate as u0 not as x0 = z0+Fu0.
    *
    * @param eqs equality constraints in the form Ax=b.
    * @param pars solver parameters, see [SolverParams].
    */
  def phase_I_Analysis_by_reduction(
    eqs:EqualityConstraint, pars:SolverParams,debugLevel:Int): FeasibilityReport = {

    if(debugLevel>0) println("\nConstraintSet: doing phase_I_Analysis by reduction.")
    Console.flush()

    val n = dim
    val solEqs = eqs.solutionSpace
    val F:DenseMatrix[Double] = solEqs.F
    val z0 = solEqs.z0
    assert(n==F.rows,"Dimension mismatch F.rows="+F.rows+" not equal to n=dim(problem)="+n)

    val k=F.cols     // dimension of variable u in x=z0+Fu

    val solverNoEqs = phase_I_Solver(Some(eqs),pars)
    // change variables x = z0+Fu
    val logFilePath = "logs/ConstraintSet_phase_I_by_dimension_reduction_log.txt"
    val solver = BarrierSolver.reducedSolver(solverNoEqs,solEqs,Logger(logFilePath))
    val sol = solver.solve(debugLevel)

    val w_feas = sol.x             // w = c(u,s), dim(u)=m, s scalar
    val u_feas = w_feas(0 until k)
    val x_feas = z0+F*u_feas       // unclear how many constraints that satisfies, so we check
    val s_feas = w_feas(k)         // if s_feas < 0, all constraints are strictly satisfied.
    val s = DenseVector[Double](1)
    s(0)=s_feas
    val isStrictlySatisfied = s_feas < 0
    val violatedCnts = constraints.filter(!_.isSatisfiedStrictly(x_feas))

    FeasibilityReport(u_feas,s,isStrictlySatisfied,violatedCnts)
  }

  //--------- Feasibility Analysis via sum of infeasibilities ----------------//


  /** Barrier solver for phase I feasibility analysis according to more detailled sum of infeasibilities
    *  algorithm, [boyd], 11.4.1, p579. No equality constraints.
    *
    * @param eqs optional equality constraints in the form Ax=b.
    * @param pars solver parameters, see [SolverParams].
    */
  private def phase_I_Solver_SOI(
    eqs:Option[EqualityConstraint], pars:SolverParams
  ):BarrierSolver = {

    val feasObjF = phase_I_SOI_ObjectiveFunction
    val feasCnts = phase_I_SOI_Constraints
    val logFilePath = "logs/ConstraintSet_phase_I_SOI_log.txt"
    BarrierSolver(feasObjF,feasCnts,eqs,pars,Logger(logFilePath))
  }

  /** Phase I feasibility analysis according to soi (sum of infeasibilities) algorithm,
    *  [boyd], 11.4.1, p580 using a BarrierSolver on the feasibility problem. This approach
    *  generates points satisfying more constraints than the basic method, if the problem is
    *  infeasible.
    *
    * To get around this problem replace the upper bounds ub_j in the constraints with
    * ub_j-epsilon and run the SOI feasibility analysis on this new constraint set.
    * Then the point x in the solution (x,s) of the feasibility solver will satisfy the
    * original constraint g_j(x) <= ub_j strictly whenever s_j<epsilon.
    *
    * @param eqs optional equality constraints in the form Ax=b.
    * @param pars solver parameters, see [SolverParams].
    */
  def phase_I_Analysis_SOI(
    eqs:Option[EqualityConstraint], pars:SolverParams, debugLevel:Int
  ): FeasibilityReport = {

    if(debugLevel>0) println("\nConstraintSet: doing phase_I_SOI_Analysis.")
    Console.flush()

    val p = numConstraints
    val n = dim
    val phase_I_SOI_eqs:Option[EqualityConstraint] = eqs.map(eq => eq.phase_I_SOI_EqualityConstraint)
    val solver = phase_I_Solver_SOI(phase_I_SOI_eqs,pars)
    val sol = solver.solve(debugLevel)

    val w_feas = sol.x                     // w = c(x,s),  dim(x)=n, s_j=g_j(x), j<p
    val x_feas = w_feas(0 until n)         // unclear how many constraints that satisfies, so we check
    val s_feas = w_feas(n until n+p)       // if s_feas(j) < 0, for all j<p, then all constraints are strictly satisfied.
    val isStrictlySatisfied = (0 until n).forall(j => s_feas(j) < 0)
    val violatedCnts = constraints.filter(!_.isSatisfiedStrictly(x_feas))

    FeasibilityReport(x_feas,s_feas,isStrictlySatisfied,violatedCnts)
  }

  /** Returns a version of itself with a feasible point added.
    * If it already has one, returns itself, otherwise performs a simple feasibility
    * analysis and adds a feasible point if possible, otherwise throws InfeasibleException.
    * This feasible point satisfies the constraints only up to tolerance pars.tol.
    *
    * @param eqs optional equality constraints in the form Ax=b.
    * @param pars solver parameters, see [SolverParams].
    */
  def withFeasiblePoint(
    eqs:Option[EqualityConstraint], pars:SolverParams, debugLevel:Int
  ):ConstraintSet with FeasiblePoint = {

    type ConstraintType = ConstraintSet with FeasiblePoint
    if(this.isInstanceOf[ConstraintType]) this.asInstanceOf[ConstraintType] else {
        // perform a feasibility analysis and add a feasible point

        val tol = pars.tol
        val feasibilityReport: FeasibilityReport = phase_I_Analysis(eqs,pars,debugLevel)
        val x0 = feasibilityReport.x0
        val s = feasibilityReport.s
        val violatedConstraints = feasibilityReport.constraintsViolated
        val eqResidual = eqs.map(eqCnt => norm(eqCnt.A*x0-eqCnt.b))

        if (s(0) > tol) throw InfeasibleException(x0, violatedConstraints, eqResidual)
        addFeasiblePoint(x0)
    }
  }
}


object ConstraintSet {

  /** Factory function,
    *
    * @param dim common dimension of all constraints in the list constraints
    * @param theConstraints list of constraints
    * @param x0 point at which all constraints are defined.
    * @return the ConstraintSet with constraints in theConstraints
    */
  def apply(dim:Int, theConstraints:Seq[Constraint],x0:DenseVector[Double]) =
  new ConstraintSet(dim,theConstraints){ def pointWhereDefined:DenseVector[Double] = x0 }


}