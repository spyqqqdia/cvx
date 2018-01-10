package cvx

import breeze.linalg.{DenseMatrix, DenseVector, max, norm, sum}
import breeze.numerics.abs

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

  def numConstraints = constraints.size
  def isSatisfiedStrictlyBy(x:DenseVector[Double]):Boolean = constraints.forall(_.isSatisfiedStrictly(x))

  /** Set of points where the constraints are satisfied strictly. */
  def strictlyFeasibleSet = new ConvexSet(dim) {

    def isInSet(x: DenseVector[Double]) = {

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

      def pointWhereDefined = x0
      def feasiblePoint = x0
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

  /** Set of these constraints for basic phase I analysis when no equalities are present,
    * [boyd], 11.4.1, p579.
    *
    * Recall: one new variable s and each constraint g_j(x) <= ub_j replaced with g_j(x)-s <= ub_j.
    * For these we can always find a point at which all these constraints are satisfied.
    * (member function [feasiblePoint]).
    */
  def phase_I_Constraints_noEqs:ConstraintSet with FeasiblePoint = {

    val n = dim
    new ConstraintSet(1+n,constraints.map(cnt => Constraint.phase_I(cnt))) with FeasiblePoint {

      val x0:DenseVector[Double] = self.pointWhereDefined
      // recall inequality constraints have the form g_j(x) <= ub_j and are replaced with
      // g_j(x)-s <= ub_j. This can be satisfied with any s > g_j(x)-ub_j
      val y0:Double = self.constraints.map(cnt => cnt.valueAt(x0)-cnt.ub).max // max_j g_j(x0)
      def feasiblePoint = DenseVector.tabulate[Double](1+n)(j => if (j<n) x0(j) else 1+y0 )
      def pointWhereDefined = feasiblePoint
    }
  }

  /** Set of these constraints for basic phase I analysis when equalities are present,
    * [boyd], 11.4.1, p579.
    *
    * Recall: one new variable s and each constraint g_j(x) <= ub_j replaced with g_j(x)-s <= ub_j.
    * For these we can always find a point at which all these constraints are satisfied
    * but the feasible point should also satisfy the equalities Ax=b.
    * We try to find such a point by minimizing 0.5*||Ax-b||² subject to the inequality
    * constraints.
    */
  def phase_I_Constraints_withEqs(
    eqs:EqualityConstraint, pars:SolverParams, logger:Logger, debugLevel:Int
  ):ConstraintSet with FeasiblePoint = {

    val delta = 1e-9  // PD regularization H -> H+delta*||H||*I
    val objF = ObjectiveFunctions.regularizedEquationResidual(eqs.A,eqs.b,delta)
    val id = "phase_I_Constraints_withEqs::feasibilityProblem"
    val feasibilityProblem = OptimizationProblem(id,objF,this,None,pars,logger,debugLevel)
    val sol:Solution = feasibilityProblem.solve(debugLevel)

    val n = dim
    new ConstraintSet(1+n,constraints.map(cnt => Constraint.phase_I(cnt))) with FeasiblePoint {

      val x0 = sol.x  // may not be strictly feasible for the inequalities
      // recall inequality constraints have the form g_j(x) <= ub_j and are replaced with
      // g_j(x)-s <= ub_j. This can be satisfied with any s > g_j(x)-ub_j
      val y0:Double = self.constraints.map(cnt => cnt.valueAt(x0)-cnt.ub).max // max_j g_j(x0)
      def feasiblePoint = DenseVector.tabulate[Double](1+n)(j => if (j<n) x0(j) else 1+y0 )
      def pointWhereDefined = feasiblePoint
    }
  }

  /** Set of these constraints for basic phase I analysis when equalities are present,
    * [boyd], 11.4.1, p579.
    *
    * Recall: one new variable s and each constraint g_j(x) <= ub_j replaced with g_j(x)-s <= ub_j.
    * For these we can always find a point at which all these constraints are satisfied
    * but the feasible point should also satisfy the equalities Ax=b.
    * We try to find such a point by minimizing 0.5*||Ax-b||² subject to the inequality
    * constraints.
    */
  def phase_I_Constraints(
    eqs:Option[EqualityConstraint], pars:SolverParams, logger:Logger, debugLevel:Int
  ):ConstraintSet with FeasiblePoint = {

    if(eqs.isEmpty) phase_I_Constraints_noEqs
    else phase_I_Constraints_withEqs(eqs.get,pars,logger,debugLevel)
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
    * g_j(x)-s_j<=ub.
    *
    * For these we can always find a point at which all these constraints are satisfied.
    * The member function _samplePoint_ returns such a point.
    */
  def phase_I_SOI_Constraints:ConstraintSet with FeasiblePoint =  {

    val n = dim
    val p = numConstraints
    val x = pointWhereDefined
    val newDim = n+p

    new ConstraintSet(newDim,Constraint.phase_I_SOI_Constraints(n,this.constraints)) with FeasiblePoint {

      val cnts = self.constraints
      // new variable u = (x,s) = (x_1,...,x_n,s_1,...,s_p), where n=dim
      // each constraint g_j(x) <= ub_j is replaced with g_j(x)-s_j <= ub_j
      // and this can be satisfied with any s_j > g_j(x)-ub_j.
      // Recall also that s_j >= 0 is required
      val feasiblePoint = DenseVector.tabulate[Double](newDim)(
        j => if (j<n) x(j) else scala.math.max(0.5,1+cnts(j-n).valueAt(x)-cnts(j-n).ub)
      )
      def pointWhereDefined = feasiblePoint
    }
  }





  /**--------------------- Solvers for the feasibility problem -------------------------**/


  /** Barrier solver for phase I feasibility analysis according to basic algorithm,
    *  [boyd], 11.4.1, p579.
    *
    * @param eqs optional equality constraint(s) in the form Ax=b.
    * @param pars solver parameters, see [SolverParams].
    */
  private def phase_I_Solver_withEqs(
    eqs: EqualityConstraint, pars:SolverParams, logger:Logger, debugLevel:Int
  ):BarrierSolver = {

    val feasObjF = phase_I_ObjectiveFunction
    val feasCnts = phase_I_Constraints(Some(eqs),pars,logger,debugLevel)
    // map the equalities to the dimension of the phase_I analysis:
    val phase_I_eqs = eqs.phase_I_EqualityConstraint
    BarrierSolver(feasObjF,feasCnts,Some(phase_I_eqs),pars,logger)
  }

  /** Barrier solver for phase I feasibility analysis according to basic algorithm,
    *  [boyd], 11.4.1, p579.
    *
    * @param pars solver parameters, see [SolverParams].
    */
  private def phase_I_Solver_withoutEqs(
    pars:SolverParams, logger:Logger, debugLevel:Int
  ):BarrierSolver = {

    val feasObjF = phase_I_ObjectiveFunction
    val feasCnts = phase_I_Constraints(None,pars,logger,debugLevel)
    BarrierSolver(feasObjF,feasCnts,None,pars,logger)
  }


  /** Minimizes ||Ax-b||^^2 subject to the inequality constraints, then
    * investigates if the solution x0 satisfies Ax0=b.
    *
    * @param eqs optional equality constraint(s) in the form Ax=b.
    * @param pars solver parameters, see [SolverParams].
    */
  def phase_I_Analysis_withEqs(
    eqs: EqualityConstraint, pars:SolverParams, logger:Logger, debugLevel:Int
  ): FeasibilityReport = {

    if(debugLevel>0){

      val msg = "ConstraintSet: doing phase_I_Analysis with equalities:"
      println(msg); Console.flush()
      logger.println(msg)
    }
    Console.flush()
    val feasBS = phase_I_Solver_withEqs(eqs,pars,logger,debugLevel)
    val sol = feasBS.solve(debugLevel)

    val w_feas = sol.x                  // w = c(x,s)
    val x_feas = w_feas(0 until dim)    // unclear how many constraints that satisfies, so we check
    val s_feas = w_feas(dim)            // if s_feas < 0, all constraints are strictly satisfied.

    // for the equality constraints use the stricter pars.tol not pars.tolEqSolve which is for
    // the ill conditioned KKT systems
    val eqError = eqs.errorAt(x_feas)
    val isStrictlySatisfied = s_feas < 0.0 && eqError < pars.tol
    val violatedCnts = constraints.filter(!_.isSatisfiedStrictly(x_feas))

    val s = DenseVector[Double](1)
    s(0)=s_feas
    val report = FeasibilityReport(x_feas,s,isStrictlySatisfied,this,Some(eqError))

    if(debugLevel>1){
      val msg = report.toString(pars.tol)
      logger.println(msg)
      println(msg); Console.flush()
    }
    report
  }


  /** Phase I feasibility analysis according to basic algorithm, [boyd], 11.4.1, p579
    *  using a BarrierSolver on the feasibility problem.
    *
    * @param pars solver parameters, see [SolverParams].
    */
  def phase_I_Analysis_withoutEqs(
    pars:SolverParams, logger:Logger, debugLevel:Int
  ): FeasibilityReport = {

    if(debugLevel>0){

      val msg = "ConstraintSet: doing phase_I_Analysis without equalities:"
      println(msg); Console.flush()
      logger.println(msg)
    }
    Console.flush()
    val feasBS = phase_I_Solver_withoutEqs(pars,logger,debugLevel)

    // terminate if objective function has been pushed below zero (then we are at a strictly
    // feasible point already, or if the duality gap is small enough
    val terminationCriterion = (os:OptimizationState) =>
      (os.objectiveFunctionValue < 0 || os.dualityGap < pars.tol) && os.equalityGap < pars.tol
    val sol = feasBS.solveSpecial(terminationCriterion,debugLevel)

    val w_feas = sol.x                  // w = c(x,s)
    val x_feas = w_feas(0 until dim)    // unclear how many constraints that satisfies, so we check
    val s_feas = w_feas(dim)            // if s_feas < 0, all constraints are strictly satisfied.

    val isStrictlySatisfied = s_feas < 0.0
    val violatedCnts = constraints.filter(!_.isSatisfiedStrictly(x_feas))

    val s = DenseVector[Double](1)
    s(0)=s_feas
    val eqError = 0.0
    val report = FeasibilityReport(x_feas,s,isStrictlySatisfied,this,Some(eqError))

    if(debugLevel>1){
      val msg = report.toString(pars.tol)
      logger.println(msg)
      println(msg); Console.flush()
    }
    report
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

    val logFilePath = "logs/ConstraintSet_phase_I_log.txt"
    val logger = Logger(logFilePath)

    if(eqs.isEmpty) phase_I_Analysis_withoutEqs(pars,logger,debugLevel)
    else phase_I_Analysis_withEqs(eqs.get,pars,logger,debugLevel)
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
    eqs:EqualityConstraint, pars:SolverParams, debugLevel:Int
  ): FeasibilityReport = {

    val logFilePath = "logs/ConstraintSet_phase_I_by_dimension_reduction_log.txt"
    val logger = Logger(logFilePath)

    if(debugLevel>0) {

      val msg = "\nConstraintSet: doing phase_I_Analysis by reduction."
      println(msg); Console.flush()
      logger.println(msg)
    }

    val n = dim
    val solEqs = eqs.solutionSpace
    val F:DenseMatrix[Double] = solEqs.F
    val z0 = solEqs.z0
    assert(n==F.rows,"Dimension mismatch F.rows="+F.rows+" not equal to n=dim(problem)="+n)

    val k=F.cols     // dimension of variable u in x=z0+Fu

    val solverNoEqs = phase_I_Solver_withoutEqs(pars,logger,debugLevel)
    // change variables x = z0+Fu
    val solver = solverNoEqs.reduced(solEqs)
    val sol = solver.solve(debugLevel)

    val w_feas = sol.x             // w = c(u,s), dim(u)=m, s scalar
    val u_feas = w_feas(0 until k)
    val x_feas = z0+F*u_feas       // unclear how many constraints that satisfies, so we check
    val s_feas = w_feas(k)         // if s_feas < 0, all constraints are strictly satisfied.
    val violatedCnts = constraints.filter(!_.isSatisfiedStrictly(x_feas))

    // here equality constraints are always present, so no Option
    // for the equality constraints use the stricter pars.tol not pars.tolEqSolve which is for
    // the ill conditioned KKT systems
    val eqError = eqs.errorAt(x_feas)
    val isStrictlySatisfied = (s_feas < 0.0) && (eqError < pars.tol)

    val s = DenseVector[Double](1)
    s(0)=s_feas
    val report = FeasibilityReport(x_feas,s,isStrictlySatisfied,this,Some(eqError))

    if(debugLevel>1){
      val logFilePath = "logs/ConstraintSet_phase_I_log.txt"
      val logger = Logger(logFilePath)
      val msg = "\nConstraintSet.phase_I_Analysis, result:\n"+report.toString(pars.tol)
      logger.println(msg)
      println(msg); Console.flush()
    }
    report
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

    if(debugLevel>0) println("ConstraintSet: doing phase_I_SOI_Analysis:")
    Console.flush()

    val p = numConstraints
    val n = dim
    val phase_I_SOI_eqs:Option[EqualityConstraint] = eqs.map(eq => eq.phase_I_SOI_EqualityConstraint(p))
    val solver = phase_I_Solver_SOI(phase_I_SOI_eqs,pars)
    val sol = solver.solve(debugLevel)

    val w_feas = sol.x                     // w = c(x,s),  dim(x)=n, s_j=g_j(x), j<p
    val x_feas = w_feas(0 until n)         // unclear how many constraints that satisfies, so we check
    val s_feas = w_feas(n until n+p)       // if s_feas(j) < 0, for all j<p, then all constraints are strictly satisfied.
    val violatedCnts = constraints.filter(!_.isSatisfiedStrictly(x_feas))

    // for the equality constraints use the stricter pars.tol not pars.tolEqSolve which is for
    // the ill conditioned KKT systems
    val eqError = eqs.map(eqs => eqs.errorAt(x_feas))
    val isStrictlySatisfied = (0 until n).forall(j => s_feas(j) < 0) && (eqError.getOrElse(0.0) < pars.tol)

    val report = FeasibilityReport(x_feas,s_feas,isStrictlySatisfied,this,eqError)

    if(debugLevel>1){
      val logFilePath = "logs/ConstraintSet_phase_I_log.txt"
      val logger = Logger(logFilePath)
      val msg = report.toString(pars.tol)
      logger.println(msg)
      println(msg); Console.flush()
    }
    report
  }

  /** Returns a version of itself with a feasible point added.
    * If it already has one, returns itself, otherwise performs a simple feasibility
    * analysis and adds a feasible point if possible, otherwise throws
    * InfeasibleProblemException.
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
      val violatedConstraints = feasibilityReport.violatedConstraints(tol)
      val eqResidual = eqs.map(eqCnt => norm(eqCnt.A*x0-eqCnt.b))

      if (!feasibilityReport.isFeasible(tol))
        throw new InfeasibleProblemException(feasibilityReport,tol)
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
    new ConstraintSet(dim,theConstraints){ def pointWhereDefined = x0 }


}