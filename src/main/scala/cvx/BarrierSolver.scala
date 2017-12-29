package cvx

import breeze.linalg.{DenseMatrix, DenseVector, _}



// WARNING: 
// in the barrier method handle the multiplications with the dimension reducing
// matrix x = x0+Fu _outside_ the sum in the barrier function (bilinear!) or else we will matrix
// multiply ourselves to death.
// This is the reason why we do not put this operation into the constraints themselves.

/** Solver for constrained convex optimization using the barrier method.
  * C.samplePoint will be used as the starting point of the optimization.
  *
  * @param C domain of definition of the barrier function.
  * @param objF: objective function of the optimization problem (needed to
  *            monitor the optimization state)
  * @param eqs Optional equality constraint(s) of the form Ax=b
  * @param pars see [SolverParams]
  */
abstract class BarrierSolver(
  val C:ConvexSet, val startingPoint:DenseVector[Double], val objF:ObjectiveFunction,
  val eqs:Option[EqualityConstraint], val pars:SolverParams, val logger:Logger
)
extends Solver {

  //  check if the pieces fit together
  assert(C.dim==startingPoint.length,
    "\nDimension mismatch C.dim="+C.dim+", startingPoint.length="+startingPoint.length+"\n"
  )
  assert(objF.dim==startingPoint.length,
    "\nDimension mismatch objF.dim="+objF.dim+", startingPoint.length="+startingPoint.length+"\n"
  )
  assert(C.isInSet(startingPoint),"Starting point x not in set C, x:\n"+startingPoint+"\n")
  eqs.map(eqCnt => {

    val A:DenseMatrix[Double] = eqCnt.A
    assert(C.dim==A.cols,  "\n\nDimension mismatch: C.dim = "+C.dim+", A.cols = "+A.cols+"\n")
  })


  override val dim:Int = C.dim

  def barrierFunction(t:Double,x:DenseVector[Double]):Double
  def gradientBarrierFunction(t:Double,x:DenseVector[Double]):DenseVector[Double]
  def hessianBarrierFunction(t:Double,x:DenseVector[Double]):DenseMatrix[Double]
  /** Number m of inequality constraints. */
  def numConstraints:Int

  def checkDim(x:DenseVector[Double]):Unit =
    assert(x.length==dim,"Dimension mismatch x.length="+x.length+" unequal to dim="+dim)

  private def logStep(t:Double):Unit = {

    val border = "\n****************************************************************\n"
    val content =    "**           BarrierSolver: step t = "+t+"                  **"
    val msg = "\n"+border+content+border+"\n"
    println(msg); Console.flush()
    logger.println(msg)
  }

  /** Find the location $x$ of the minimum of f=objF over C by the newton method
    * starting from the starting point x0 with no equality constraints.
    *
    * @param terminationCriterion: depends on context (e.g. in a phase I analysis we terminate
    *                            as soon as the objective function is pushed below zero).
    * @return Solution object: minimizer with additional info.
    */
  def solveWithoutEQs(terminationCriterion:(OptimizationState)=>Boolean, debugLevel:Int=0):Solution = {

    val tol=pars.tol // tolerance for duality gap
    val mu = 10.0    // factor to increase parameter t in barrier method.
    var t = 1.0
    var x = startingPoint       // iterates x=x_k
    var obfFcnValue = Double.MaxValue
    var normGrad = Double.MaxValue
    var dualityGap = Double.MaxValue
    var newtonDecrement = Double.MaxValue
    val equalityGap = 0.0
    var optimizationState = OptimizationState(normGrad,newtonDecrement,dualityGap,equalityGap,obfFcnValue)
    var sol:Solution = null   // solutions at parameter t

    // insurance against nonterminating loop, normally terminates long before that
    val maxIter = 300/mu
    var iter = 0
    while(!terminationCriterion(optimizationState) && iter<maxIter){

      if(debugLevel>2) logStep(t)
      // solver for barrier function at fixed parameter t
      val objF_t = new ObjectiveFunction(dim){

        def valueAt(x:DenseVector[Double]):Double = { checkDim(x); barrierFunction(t,x) }
        def gradientAt(x:DenseVector[Double]):DenseVector[Double] = { checkDim(x); gradientBarrierFunction(t,x) }
        def hessianAt(x:DenseVector[Double]):DenseMatrix[Double] = { checkDim(x); hessianBarrierFunction(t,x) }
      }
      val solver = new UnconstrainedSolver(objF_t,C,x,pars,logger)
      sol = solver.solve(debugLevel)

      x = sol.x
      obfFcnValue = objF.valueAt(x)
      normGrad = sol.normGrad
      newtonDecrement = sol.newtonDecrement
      dualityGap = numConstraints/t
      optimizationState = OptimizationState(normGrad,newtonDecrement,dualityGap,equalityGap,obfFcnValue)

      if(debugLevel>3){
        print("\nOptimization state:"+optimizationState)
        Console.flush()
      }
      t = mu*t
      iter+=1
    }
    sol
  }
  /** Find the location $x$ of the minimum of f=objF over C with equality constraints Ax=b
    * by iteratively solving the KKT system with backtracking line search starting from the
    * starting point x0.
    *
    * @return Solution object: minimizer with additional info.
    */
  def solveWithEQs(
                    terminationCriterion:(OptimizationState)=>Boolean,
                    A:DenseMatrix[Double],b:DenseVector[Double],debugLevel:Int=0
                  ):Solution = {

    val tol=pars.tol // tolerance for duality gap
    val mu = 10.0    // factor to increase parameter t in barrier method.
    var t = 1.0
    var x = startingPoint     // iterates x=x_k
    var obfFcnValue = Double.MaxValue
    var normGrad = Double.MaxValue
    var dualityGap = Double.MaxValue
    var newtonDecrement = Double.MaxValue
    var equalityGap = Double.MaxValue
    var optimizationState = OptimizationState(normGrad,newtonDecrement,dualityGap,equalityGap,obfFcnValue)
    var sol:Solution = null   // solutions at parameter t


    // insurance against nonterminating loop, normally terminates long before that
    val maxIter = 300/mu
    var iter = 0
    while(!terminationCriterion(optimizationState) && iter<maxIter){

      if(debugLevel>2) logStep(t)
      // solver for barrier function at fixed parameter t
      val objF_t = new ObjectiveFunction(dim){

        def valueAt(x:DenseVector[Double]):Double = { checkDim(x); barrierFunction(t,x) }
        def gradientAt(x:DenseVector[Double]):DenseVector[Double] = { checkDim(x); gradientBarrierFunction(t,x) }
        def hessianAt(x:DenseVector[Double]):DenseMatrix[Double] = { checkDim(x); hessianBarrierFunction(t,x) }
      }
      val solver = EqualityConstrainedSolver(objF_t,C,x,A,b,pars,logger)
      sol = solver.solve(debugLevel)

      x = sol.x
      obfFcnValue = objF.valueAt(x)
      normGrad = sol.normGrad
      newtonDecrement = sol.newtonDecrement
      dualityGap = numConstraints/t
      equalityGap = sol.equalityGap
      optimizationState = OptimizationState(normGrad,newtonDecrement,dualityGap,equalityGap,obfFcnValue)

      if(debugLevel>3) {
        print("\nOptimization state:" + optimizationState)
        Console.flush()
      }
      t = mu*t
      iter+=1
    }
    sol
  }
  def solveSpecial(terminationCriterion:(OptimizationState)=>Boolean, debugLevel:Int=0):Solution =
    if(eqs.isDefined) solveWithEQs(terminationCriterion,eqs.get.A,eqs.get.b,debugLevel)
    else solveWithoutEQs(terminationCriterion,debugLevel)

  /** Solution based on standard termination criterion: dualityGap < tol
    */
  def solve(debugLevel:Int=0):Solution = {

    val terminationCriterion = (os:OptimizationState) => os.dualityGap < pars.tol
    solveSpecial(terminationCriterion,debugLevel)
  }
}


/** Some factory functions.*/
object BarrierSolver {

  /** BarrierSolver for minimization with or without equality constraints Ax=b.
    *
    * @param cnts set of inequality constraints.
    * @param pars see [SolverParams].
    */
  def apply(
             objF: ObjectiveFunction, cnts: ConstraintSet with FeasiblePoint,
             eqs: Option[EqualityConstraint], pars: SolverParams, logger:Logger
           ): BarrierSolver = {

    val Feas = cnts.strictlyFeasibleSet
    val C = ConvexSet.addSamplePoint(Feas, cnts.feasiblePoint)
    new BarrierSolver(C, cnts.feasiblePoint, objF, eqs, pars, logger) {

      def numConstraints:Int = cnts.constraints.length

      def barrierFunction(t: Double, x: DenseVector[Double]): Double =
        cnts.constraints.foldLeft(t * objF.valueAt(x))((sum: Double, cnt: Constraint) => {

          val d = cnt.ub - cnt.valueAt(x)
          if (d <= 0)
            throw new IllegalArgumentException("x not strictly feasible, d = " + d + ", x:\n" + x)
          sum - Math.log(d)
        })

      def gradientBarrierFunction(t: Double, x: DenseVector[Double]): DenseVector[Double] =
        cnts.constraints.foldLeft(objF.gradientAt(x) * t)((sum: DenseVector[Double], cnt: Constraint) => {

          val d = cnt.ub - cnt.valueAt(x)
          val G = cnt.gradientAt(x)
          if (d <= 0)
            throw new IllegalArgumentException("x not strictly feasible, d = " + d + ", x:\n" + x)
          sum + G / d
        })

      def hessianBarrierFunction(t: Double, x: DenseVector[Double]): DenseMatrix[Double] =
        cnts.constraints.foldLeft(objF.hessianAt(x) * t)((sum: DenseMatrix[Double], cnt: Constraint) => {

          val d = cnt.ub - cnt.valueAt(x)
          val G = cnt.gradientAt(x)
          val H = cnt.hessianAt(x)
          if (d <= 0)
            throw new IllegalArgumentException("x not strictly feasible, d = " + d + ", x:\n" + x)

          val GGt = G * G.t
          sum + GGt / (d * d) + H / d
        })
    }
  }
  /** Version of solver bs which operates on the dimension reduced variable u related to
    * the original variable as x = z0+Fu.
    * This solves the minimization problem of bs under the additional constraint that
    * x is of the form z0+Fu and operates on the variable u. Results are reported using the variable x.
    *
    * The intended application is to problems with equality constraints Ax=b, where the solution
    * spoace of the equality constraints is parametrized as x=z0+Fu, u unconstrained.
    *
    * @param sol solution space of Ax=b (then z0=sol.z0, F=sol.F).
    */
  def reducedSolver(bs:BarrierSolver, sol:SolutionSpace, logger:Logger): BarrierSolver = {

    // pull the domain bs.C back to the u variable
    val C = bs.C
    val z0 = sol.z0
    val F = sol.F
    val dim_u = F.cols
    val x0 = bs.startingPoint
    val u0 = sol.parameter(x0)      // u0 with x0 = z0+F*u0

    val D = new ConvexSet(dim_u) {

      def isInSet(u:DenseVector[Double]):Boolean = C.isInSet(z0+F*u)
      def samplePoint = Some(u0)
    }

    new BarrierSolver(D,u0,bs.objF,None,bs.pars,logger){

      def numConstraints:Int = bs.numConstraints
      def barrierFunction(t:Double,u:DenseVector[Double]):Double = bs.barrierFunction(t,z0+F*u)
      def gradientBarrierFunction(t:Double,u:DenseVector[Double]):DenseVector[Double] =
        F.t*bs.gradientBarrierFunction(t,z0+F*u)
      def hessianBarrierFunction(t:Double,u:DenseVector[Double]):DenseMatrix[Double] =
        (F.t*bs.hessianBarrierFunction(t,z0+F*u))*F

      override def solveSpecial(terminationCriterion:(OptimizationState)=>Boolean,debugLevel:Int):Solution = {

        // 'super': with new X { ... } we automatically extend X
        val sol = super.solveSpecial(terminationCriterion,debugLevel)
        Solution(z0+F*sol.x, sol.newtonDecrement, sol.dualityGap, 0, sol.normGrad, sol.iter, sol.maxedOut)
      }
    }
  }
}