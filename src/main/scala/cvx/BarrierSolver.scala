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
extends Solver { self =>

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
  /** Version of _this_ solver which operates on the variable u related to
    * the original variable as x = z0+Fu.
    * This solves the minimization problem under the additional constraint that
    * x is of the form z0+Fu and operates on the variable u. Results are reported using
    * the variable u not x.
    *
    * The intended application is to problems with equality constraints Ax=b, where the solution
    * space of the equality constraints is parametrized as x=z0+Fu, u unconstrained.
    *
    * REMARK: affine transformation can induce catastrophic overhead via large numbers of
    * big matrix multiplications if not handled correctly. In our approach this does not
    * happen, since we transform the completed barrier function instead of transforming
    * all summands in the barrier function and then summing up the transformed parts.
    * Note for example that we do not need a method affineTransformed for the class
    * ConstraintSet.
    *
    * @param u0 a vector satisfying x0 = z0+F*u0, where x0 is the startingPoint of _this_
    *           solver.
    *
    */
  def affineTransformed(
    z0:DenseVector[Double], F:DenseMatrix[Double], u0:DenseVector[Double]
  ): BarrierSolver = {

    // pull the domain bs.C back to the u variable
    val dim_u = F.cols
    val x0 = startingPoint
    assert(
      norm(x0-(z0+F*u0))<pars.tolEqSolve,
      "\nu0 does not map to x0 under the variable transform.\n"
    )
    val D = C.affineTransformed(z0,F,u0)
    val transformedObjF = objF.affineTransformed(z0,F)
    val transformedEqs = eqs.map(_.affineTransformed(z0,F))

    new BarrierSolver(D,u0,transformedObjF,transformedEqs,pars,logger){

      def numConstraints:Int = self.numConstraints
      def barrierFunction(t:Double,u:DenseVector[Double]):Double = self.barrierFunction(t,z0+F*u)
      def gradientBarrierFunction(t:Double,u:DenseVector[Double]):DenseVector[Double] =
        F.t*self.gradientBarrierFunction(t,z0+F*u)
      def hessianBarrierFunction(t:Double,u:DenseVector[Double]):DenseMatrix[Double] =
        (F.t*self.hessianBarrierFunction(t,z0+F*u))*F
    }
  }
  /** As [[affineTransformed(z0:DenseVector[Double],F:DenseMatrix[Double], u0:DenseVector[Double])]]
    * with u0 computed as the solution of Fu=x0-z0 using the SVD of F.
    */
  def affineTransformed(z0:DenseVector[Double],F:DenseMatrix[Double]): BarrierSolver = {

    val x0 = startingPoint
    val debugLevel = 0
    val u0 = MatrixUtils.svdSolve(F,x0-z0,logger,pars.tolEqSolve,debugLevel)
    affineTransformed(z0,F,u0)
  }

  /** As As [[affineTransformed(z0:DenseVector[Double],F:DenseMatrix[Double], u0:DenseVector[Double])]]
    * with z0,F,u0 computed by the solution space sol. This usually implies a dimension reduction
    * in the independent variable ( x -> u ).
    */
  def reduced(sol:SolutionSpace): BarrierSolver = {

    val z0 = sol.z0
    val F  = sol.F
    val x0 = startingPoint
    val u0 = sol.parameter(x0)     // more efficient than MatrixUtils.svdSolve above.
    affineTransformed(z0,F,u0)
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
}