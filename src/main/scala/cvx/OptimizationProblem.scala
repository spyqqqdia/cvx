package cvx

import breeze.linalg.{DenseVector, _}



/**
  * Created by oar on 12/2/16.
  *
  * Constrained or unconstrained optimization problem.
  *
  * @param solver Solver for the problem
  */
class OptimizationProblem(
  val id:String, val objectiveFunction: ObjectiveFunction, val solver:Solver, val logger:Logger
){
  self =>

  def solve(debugLevel:Int=0):Solution = solver.solve(debugLevel)

  /** Add the known (unique) solution to the minimization problem.
    * For testing purposes.
    */
  def addSolution(optSol:KnownMinimizer): OptimizationProblem with KnownMinimizer  =
    new OptimizationProblem(id,objectiveFunction,solver,logger) with KnownMinimizer {

      override def theMinimizer: DenseVector[Double] = optSol.theMinimizer
      def isMinimizer(x:DenseVector[Double],tol:Double):Boolean = optSol.isMinimizer(x,tol)
      def minimumValue:Double = optSol.minimumValue
    }

  /** Add the known solutions to the minimization problem. List must contain
    * all solutions. For testing purposes.
    */
  def addSolutions(sols:List[DenseVector[Double]]): OptimizationProblem with KnownMinimizer  =
    new OptimizationProblem(id,objectiveFunction,solver,logger) with KnownMinimizer {

      assert(sols.nonEmpty,"\naddSolutions: no solutions provided.\n")
      override def theMinimizer: DenseVector[Double] = sols.head
      def isMinimizer(x:DenseVector[Double],tol:Double):Boolean = min(sols.map(sol => norm(x-sol))) < tol
      def minimumValue:Double = min(sols.map(sol => objectiveFunction.valueAt(sol)))
    }
  /**
    * @param sol the computed solution
    * @param tol tolerated deviation from known solution (l2-norm)
    */
  def report(sol:Solution,tol:Double):Unit = {

    val x = sol.x                       // minimizer, solution found
    val y = objectiveFunction.valueAt(x)
    val msg = sol+"\n"+"Objective function value at computed solution: "+y+".\n"

    print(msg)
    Console.flush()
    logger.println(msg)
    logger.close()
  }

  /** The optimization problem under change of variables x = z0 + Fu, where x is
    * the original independent variable.
    * WARNING: solution will be reported in the variable u not the original variable x.
    *
    * @param u0 a vector satisfying this.solver.startingPoint = z0+F*u0.
    *
    * @return the problem of minimizing k(u)=f(z+Fu) under constraints
    *         h_j(u)=g_j(z+Fu) <= ub_j,
    * where the original problem (_this_) was the minimization of f(x) under constraints
    * g_j(x)<=ub_j. If the original problem had equality constraints of the form Ax=b
    * they will also be transformed accordingly.
    */
  def affineTransformed(
    z0:DenseVector[Double],F:DenseMatrix[Double], u0:DenseVector[Double]
  ):OptimizationProblem = {

    val x0 = solver.startingPoint
    assert(
      norm(x0-(z0+F*u0))<solver.pars.tolEqSolve,
      "\nu0 does not map to the starting point x0.\n"
    )
    val transformedObjectiveFunction = objectiveFunction.affineTransformed(z0,F)
    val transformedSolver = solver.affineTransformed(z0,F,u0)
    val id = self.id+"_affineTransformed"
    new OptimizationProblem(id,transformedObjectiveFunction,transformedSolver,logger)
  }
}

/** Factory functions to allocate problems and select the solver to use.
  */
object OptimizationProblem {


  /** Allocates an optimization problem constrained only by $x\in C$, where C is an
    * open convex set known to contain a minimizer (typically the full Euclidean space).
    *
    * @param id ID for problem
    * @param objF objective function
    * @param pars solver parameters, see [SolverParams].
    * @return problem minimizing objective function under the constraint $x\in C$ applying the parameters in pars
    * and starting the iteration at C.samplePoint.
    */
  def apply(
             id:String,
             objF:ObjectiveFunction,
             startingPoint:DenseVector[Double],
             C: ConvexSet,
             pars:SolverParams,
             logger:Logger
           ): OptimizationProblem = {

    assert(objF.dim == C.dim,
      "Dimension mismatch objF.dim = "+objF.dim+", C.dim = "+C.dim+"\n"
    )
    val solver = UnconstrainedSolver(objF,C,startingPoint,pars,logger)
    new OptimizationProblem(id,objF,solver,logger)
  }


  /** Allocates an optimization problem with inequality constraints and optional equality constraints
    * using a barrier solver for both the actual solution and the feasibility analysis.
    *
    * @param id name for problem
    * @param setWhereDefined: The convex set where the objective function
    *   and all the constraints are defined, usually ConvexSets.wholeSpace.
    * @param objF objective function
    * @param ineqs inequality constraints
    * @param eqs optional equality constraint(s) in the form Ax=b
    * @param solverType: "BR" (barrier solver), "PD0" (primal dual with one slack variable),
    *   "PD1" (primal dual with one slack variable for each inequality constraint), see docs/primaldual.pdf.
    * @param pars solver parameters, see [SolverParams].
    * @return problem minimizing objective function under constraints applying the parameters in pars
    * and starting the iteration at ineqs.feasiblePoint.
    */
  def apply(
             id:String,
             setWhereDefined:ConvexSet,
             objF:ObjectiveFunction,
             ineqs: ConstraintSet,
             eqs:Option[EqualityConstraint],
             solverType:String,
             pars:SolverParams,
             logger:Logger,
             debugLevel:Int
  ): OptimizationProblem = {

    assert(objF.dim==ineqs.dim,"\n\nobjF.dim = "+objF.dim+", ineqs.dim = "+ineqs.dim+"\n")
    eqs.map(eqCnt => assert(eqCnt.dim==objF.dim,"\n\neqCnt.dim = "+eqCnt.dim+", objF.dim = "+objF.dim+"\n"))
    require(solverType=="BR" || solverType == "PD0" || solverType == "PD1",
      "\nUnknown solverType = "+solverType+", should be one of BR, PD0 or PD1.\n"
    )

    val solver:Solver =

      if(solverType == "BR")
        BarrierSolver(objF,ineqs.withFeasiblePoint(eqs,pars,debugLevel),eqs,pars,logger)

      else if (solverType == "PD0"){

        require(pars.K.nonEmpty,
          "\nConstant K>0 for the primal dual solver is missing from pars.\n")
        val K:Double = pars.K.get
        require(pars.K.get>0,"\npars.K must be positive but is = "+K+"\n")
        PrimalDualSolver(setWhereDefined,objF,ineqs,eqs,pars,logger,K)

      } else {

        require(pars.vec_K.nonEmpty,
          "\nVector K>0 for the primal dual solver is missing from pars.\n")
        val K:Vector[Double] = pars.vec_K.get
        require(pars.vec_K.get.forall(_>0),"\npars.vec_K must be positive but is = "+K+"\n")
        require(K.length==ineqs.numConstraints,
          "Dimension mismatch: dim(K) = "+K.length+
            " not equal to the number of inequality constraints = "+ineqs.numConstraints+".\n"
        )
        PrimalDualSolver(setWhereDefined,objF,ineqs,eqs,pars,logger,K)
      }
    new OptimizationProblem(id,objF,solver,logger)
  }


}
