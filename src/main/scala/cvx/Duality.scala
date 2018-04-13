package cvx

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * Created by oar on 26.03.18.
  *
  * Dual objective function $L_*(z) = \inf_xL(x,z)$, where L(x,z) is the Lagrangian of
  * the convex problem and z denotes the dual variable, either
  *
  *    z = lambda        (if there are no equality constraints), or
  *    z = (lambda,nu)   (if there are equality constraints)
  *
  * Here lambda consists of the dual variables attached to the inequality
  * constraints and nu the dual variables corresponding to the equality constraints.
  * The dual problem then is the problem
  *
  *   max L_*(z) subject to lambda >= 0.
  *
  * For this to be useful we also need a function primalOptimum(z) which computes
  * the primal optimum $x^^*$ from the dual optimum $z^^*$. In practice this function
  * is computed from  the equation
  *
  *   $\grad_xL(x,z)=0$
  *
  * by solving for x as a function of z. For each (fixed) z the solution x=x(z)
  * satisfies $L(x(z),z)=\inf_xL(x,z)$ by convexity of the Lagrangian L(x,z) as
  * a function of the primal variable x.
  *
  * Note: you will implement a problem as an OptimizationProblem with Duality
  * only if you want to be able to attack the problem both directly and via duality.
  *
  * In practice this makes little sense: if the dual problem is simpler you simply
  * allocate the dual problem and solve that otherwise solve the primal problem.
  * We use this feature only to check the direct solution against the solution of the
  * primal problem via the dual problem.
  */
trait Duality {

  /** Id of primal problem.*/
  def id:String
  /** Dimension of dual problem = number of inequality constraints plus
    * the number of equality constraints.*/
  def dualDim:Int
  /** Number of inequality constraints of the primal problem.*/
  def numInequalities:Int
  /** The concave function $L_*(z) = \inf_xL(x,z)$, where L(x,z) is the
    * Lagrangian of the primal problem with primal variables x and dual variables z.
    */
  def dualObjFAt(z:DenseVector[Double]):Double
  /** Gradient of the dual objective function $L_*(z)$.
    */
  def gradientDualObjFAt(z:DenseVector[Double]):DenseVector[Double]

  /** Hessian of the dual objective function $L_*(z)$.
    */
  def hessianDualObjFAt(z:DenseVector[Double]):DenseMatrix[Double]

  /** The primal optimum x^^* as a function of the dual optimum z^^*.
    * Typically this is computed from the equation $\grad L(x,z)=0$
    * by solving for x as a function of the dual variables z.
    * Here L(x,z) denotes the Lagrangian of the primal problem, x the primal
    * variables and z the dual variables.
    */
  def primalOptimum(z:DenseVector[Double]):DenseVector[Double]

  /** Number of equalities in the EqualityConstraint (0 if no
    * EqualityConstraint is present)
    */
  def numEqualities:Int = dualDim - numInequalities
  /** Objective function $-L_*(x,z)$ for the convex formulation of the
    * dual problem.
    */
  private def objF:ObjectiveFunction = new ObjectiveFunction(dualDim) {

    override def valueAt(z:DenseVector[Double]):Double = -dualObjFAt(z)
    override def gradientAt(z:DenseVector[Double]):DenseVector[Double] = -gradientDualObjFAt(z)
    override def hessianAt(z:DenseVector[Double]):DenseMatrix[Double] = - hessianDualObjFAt(z)
  }
  /** The constraints lambda >= 0 for the dual problem.*/
  private def dualConstraintSet:ConstraintSet = {

    val lambdaPositive:List[Constraint] =
      Constraints.firstCoordinatesPositive(dualDim,numInequalities)

    val C = ConvexSets.wholeSpace(dualDim)
    val x0 = DenseVector.zeros[Double](dualDim)

    ConstraintSet(dualDim,lambdaPositive,C,x0)
  }

  /** Dual problem max L_*(z) subject to lambda = z(0:numInequalities)>=0.
    * Since $L_*(z)$ is concave this is equivalent to the convex problem
    * min -L_*(z) subject to lambda>=0.
    *
    * @param solverType: Solver used for dual problem, must be
    *                    "BR" (BarrierSolver) or "PD" (PrimalDualSolver).
    */
  def dualProblem(
    solverType:String,pars:SolverParams,logger:Logger,debugLevel:Int=0
  ): OptimizationProblem = {

    val theId = id+" dual problem"
    val setWhereDefined = ConvexSets.wholeSpace(dualDim)
    val theObjF = objF
    val cnts = dualConstraintSet
    val dualFeasiblePoint = DenseVector.fill[Double](dualDim)(0.001)
    val ineqs = cnts.addFeasiblePoint(dualFeasiblePoint)

    OptimizationProblem(theId,setWhereDefined,theObjF,ineqs,None,solverType,pars,logger,debugLevel)

  }

  /** Solve the primal problem via duality.
    *
    * @param solverType must be "BR" (BarrierSolver) or "PD" (PrimalDualSolver).
    * @return Solution of primal problem.
    */
  def solveDual(
    solverType:String,pars:SolverParams,logger:Logger,debugLevel:Int=0
  ): Solution = {

    val dP = dualProblem(solverType,pars,logger,debugLevel)
    val solD:Solution = dP.solve(debugLevel)
    val z:DenseVector[Double] = solD.x      // optimal dual variables
    val primalOpt:DenseVector[Double] = primalOptimum(z)
    // fix up the dual solution to become the primal one
    solD.copy(
      x = primalOpt,
      lambda = Some(z(0 until numInequalities)),
      nu = if (numEqualities==0) None else Some(z(numInequalities until dualDim))
    )
  }

}
