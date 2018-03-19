package cvx

import breeze.linalg.{DenseMatrix, DenseVector, NotConvergedException, _}
import breeze.numerics.sqrt

/**
  * Created by oar on 12/1/16.
  *
  * Solver for min f(x), where x is constrained as $x\in C$, C an _open_ convex set and it is known that
  * $f(x)\to+\infty$, as x approaches the boundary of C. This can be treated as an unconstrained convex
  * minimization if the line search always backtracks into the interior of C.
  *
  * The starting point is set to C.samplePoint by default, method needs to be overridden if something else
  * is desired.
  *
  * Intended application: the convex sub-problems of the barrier method without equality constraints.
  * Then C is the set of _strictly_ feasible points.
  *
  * @param C open convex set over which the objective function is minimized
  * @param pars see [SolverParams].
  */
class UnconstrainedSolver(
                           val objF:ObjectiveFunction, val C:ConvexSet, val startingPoint:DenseVector[Double],
                           val pars:SolverParams, val logger:Logger
                         ) extends Solver {

  /** Find the location $x$ of the minimum of f=objF over C by the newton method
    * starting from the starting point x0.
    *
  is smaller than sqrt(delta), the regularization will be applied.
    *
    * @return Solution object: minimizer with additional info
    */
  def solve(debugLevel:Int):Solution = {

    val maxIter = pars.maxIter; val alpha=pars.alpha; val beta=pars.beta
    val tol=pars.tol; val tolEqSolve=pars.tolEqSolve; val delta=pars.delta

    val breakDown = NotConvergedException.Breakdown
    var iter = 0
    var newtonDecrement = tol + 1
    var x = startingPoint
    var y = objF.gradientAt(x)
    var normGrad = norm(y)
    var trustRadius = Double.NaN

    while (iter < maxIter && newtonDecrement > tol && normGrad > tol) {

      val f = objF.valueAt(x)
      val H = objF.hessianAt(x)
      val n = H.rows

      // newton  step
      val d = try {
        MatrixUtils.choleskySolve(H, -y, logger, tolEqSolve, debugLevel)
      } catch {

        case e: Exception => try {

          val M = H + DenseMatrix.eye[Double](H.rows)*1e-9
          MatrixUtils.choleskySolve(M, -y, logger, tolEqSolve, debugLevel)

        } catch {

          case e: Exception => MatrixUtils.symSolve(H, -y, logger, tolEqSolve, debugLevel)
        }
      }
      val q = d dot y
      newtonDecrement = -q/2
      if(q>0){     // loop will terminate on newtonDecrement < tol

        if(debugLevel>2){

          var msg = "\n\nUnconstrainedSolver, Iteration "+iter+":\n"
          msg += "Newton step d is not a descent direction: (d dot gradF(x)) = "+q
          msg+="\nx:\n"+x+"\ngradF(x):\n"+y+"\nhessF(x):\n"+H+"\nNewton step d:\n"+d+"\n\n"
          logger.print(msg)
        }
      }
      //continue only if newtonDecrement > eps
      if(newtonDecrement > tol){

        // trust region with fallback on backtracking line search,
        // in particular backtrack into the set C
        val hNorm_d = sqrt(-q)                  // |d|_H = sqrt(d'Hd)
        if(iter==0) trustRadius = hNorm_d
        // step vector
        val s = if(iter==0 || hNorm_d <= trustRadius) d else d * (trustRadius/hNorm_d)

        // from x+s move back through x+t*s towards x into the set C
        var it = 0 // safeguard against bugs
        var t = 1.0
        while (!C.isInSet(x + s*t) && (it < 200)) { t *= beta; it += 1 }
        if (it == 100) throw new NotConvergedException(
          breakDown, "Line search: backtracking into the set C failed."
        )
        // adjust the trust radius according to decrease in function value
        // if that step is too big, reduce the trust radius
        val rho = 1+1/4
        if(!C.isInSet(x + s)){ trustRadius /= rho }
        else {
          val f_new = objF.valueAt(x + s*t)
          if (f_new > f + alpha * t * q) trustRadius /= rho
          if ((f_new < f + ((1+alpha)/2) * t * q) && trustRadius <= hNorm_d) trustRadius *= rho
        }
        // move back further to ensure sufficient value decrease
        while ((objF.valueAt(x + s*t) > f + alpha * t * q) && (it < 200)) { t *= beta; it += 1 }
        // cannot happen unless gradient is messed up
        if (it == 100) throw new NotConvergedException(
          breakDown, "Line search: sufficient decrease not reached after 200 iterations"
        )
        // step to next iterate
        x = x + s*t
        y = objF.gradientAt(x)
        normGrad = norm(y)
      }
      iter+=1
    }
    // None: duality gap, equality gap, slack variables, dual variables
    Solution(
      x,None,None,
      Some(newtonDecrement),None,None,Some(normGrad),None,
      iter,iter>=maxIter
    )
  }

  /** Same as [solve(Int)], the parameter terminationCriterion is ignored. We need that only in the
    * outer loop of the barrier solver.
    *
    * @return Solution object: minimizer with additional info, tuple
    *         (x,newtonDecrement,||Ax-b||,||grad f(x)||,iter,iter==maxIter).
    */
  def solveSpecial(terminationCriterion:(OptimizationState)=>Boolean, debugLevel:Int):Solution = solve(debugLevel)

  /** The solver operating on the variable u related to the original variable x via the
    * affine transform x = z0+Fu. This means that the underlying problem has been similarly
    * transformed (objective function and constraints).
    *
    * This can be viewed as introducing and additional constraint that the solution x0 must
    * be of the form x0 = z0+F*u0.
    * Solution will be reported in the variable u not in x.
    *
    * @param u0 a vector satisfying this.startingPoint = z0 + F*u0.
    */
  override def affineTransformed(
    z0: DenseVector[Double], F: DenseMatrix[Double], u0: DenseVector[Double]
  ): UnconstrainedSolver = {

    val x0 = startingPoint
    assert(
      norm(x0-(z0+F*u0))<pars.tolEqSolve,
      "\nu0 does not map to starting point x0 under transform.\n"
    )
    val transformedObjF = objF.affineTransformed(z0,F)
    val D = C.affineTransformed(z0,F,u0)
    new UnconstrainedSolver(transformedObjF,D,u0,pars,logger)
  }
}






object UnconstrainedSolver {

  def apply(
             objF:ObjectiveFunction,
             C:ConvexSet,
             startingPoint:DenseVector[Double],
             pars:SolverParams,
             logger:Logger
           ): UnconstrainedSolver = new UnconstrainedSolver(objF,C,startingPoint,pars,logger)

  /** Unconstrained solver in the variable u with C being the whole space in dimension dim(u), where
    * the solutions to the equality constraints Ax=b are parametrized as x=z0+Fu, see [SolutionSpace].
    * The solver reports the solution x not u.
    *
    * @param pars solver parameters, see [SolverParams].
    * @return
    */
  def apply(
             objF:ObjectiveFunction,
             A:DenseMatrix[Double],
             b:DenseVector[Double],
             pars:SolverParams,
             logger:Logger
           ): UnconstrainedSolver = {

    val solSpace = SolutionSpace(A,b)
    val z0 = solSpace.z0
    val F = solSpace.F
    val dim_u = F.cols
    val C = ConvexSet.fullSpace(dim_u)
    // override solve to report x instead of u
    new UnconstrainedSolver(objF,C,z0,pars,logger) {

      override def solve(debugLevel:Int=0) = {

        val sol = super.solve(debugLevel)
        val u = sol.x; val x = z0+F*u
        Solution(
          x,None,None,
          sol.newtonDecrement,None,None,sol.normGrad,None,
          sol.iter, sol.maxedOut)
      }
    }
  }

}