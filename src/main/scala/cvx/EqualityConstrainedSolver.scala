/**
  * Created by oar on 09.10.17.
  */
package cvx

import breeze.linalg.{DenseMatrix, DenseVector, _}

/** Solver for minimizing the objective function objF on the convex set C subject to
  * equality constraints Ax=b. It is assumed that the set [Ax=b] intersects the interior
  * of C.
  *
  * @param C domain of definition of the objective function objF.
  * @param startingPoint: a point in interior of C, does not need to satisfy Ax=b.
  * @param A matrix A in equality constraints Ax=b.
  * @param b vector b in equality constraints Ax=b.
  * @param pars parameters to control solver behaviour.
  */
class EqualityConstrainedSolver(
   val objF:ObjectiveFunction, val C:ConvexSet, val startingPoint:DenseVector[Double],
   val A:DenseMatrix[Double], val b:DenseVector[Double], val pars:SolverParams, val logger:Logger
)
extends Solver {

  assert(objF.dim==C.dim, "Dimension mismatch: objF.dim = "+objF.dim+", C.dim = "+C.dim+"\n")
  assert(C.dim==A.cols,  "Dimension mismatch: C.dim = "+C.dim+", A.cols = "+A.cols+"\n")
  assert(A.rows==b.length, "Dimension mismatch: A.rows = "+A.rows+", b.length = "+b.length+"\n")
  assert(C.isInSet(startingPoint),"Starting point x not in set C, x:\n"+startingPoint+"\n")


  /** Find the location $x$ of the minimum of f=objF over C subject to the equality constraints
    * Ax=b by iteratively solving the KKT system with backtracking line search starting from the
    * starting point x0.
    *
    * @return Solution object: minimizer with additional info, tuple
    *         (x,newtonDecrement,||Ax-b||,||grad f(x)||,iter,iter==maxIter).
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
    var eqDiff = b-A*x
    while (iter < maxIter && ((newtonDecrement > tol && normGrad > tol) || norm(eqDiff)>tol)) {


      val f = objF.valueAt(x)
      val H = objF.hessianAt(x)
      val n = H.rows

      // FIX ME: the solution step
      val KKTS = KKTSystem(H,A,y,eqDiff)
      val d:DenseVector[Double] = KKTS.solve(logger,tolEqSolve,debugLevel)._1

      val q = d dot y
      newtonDecrement = -q/2
      if(q>0){

        var msg = "\n\nEqualityConstrainedSolver, Iteration "+iter+":\n"
        msg += "Newton step d is not a descent direction: (d dot gradF(x)) = "+q
        if(debugLevel>0) logger.print(msg)
        if(debugLevel>1){
          logger.print("\n\ngradF(x):\n")
          MatrixUtils.print(y,logger,3)
          logger.print("\n\nhessF(x):\n")
          MatrixUtils.print(H,logger,3)
          logger.println("\n")
        }
        iter = maxIter      // break off
      }
      //continue only if newtonDecrement > tol
      if(newtonDecrement > tol){

        // backtracking line search
        var it = 0 // safeguard against bugs
        var t = 1.0

        // from x+d move back towards x into the set C
        while (!C.isInSet(x + d*t) && (it < 100)) { t *= beta; it += 1 }
        if (it == 100) throw new NotConvergedException(
          breakDown, "\n\nLine search: backtracking into the set C failed.\n\n"
        )
        // move back further to ensure sufficient value decrease
        while ((objF.valueAt(x + d*t) > f + alpha * t * q) && (it < 100)) { t *= beta; it += 1 }
        // cannot happen unless gradient is messed up
        if (it == 100) throw new NotConvergedException(
          breakDown, "\n\nLine search: sufficient decrease not reached after 100 iterations.\n\n"
        )
        // step to next iterate
        x = x + d*t
        y = objF.gradientAt(x)
        normGrad = norm(y)
        eqDiff = b-A*x
      }
      iter+=1
    }
    Solution(x,newtonDecrement,norm(eqDiff),normGrad,iter,iter==maxIter)
  }
}
object EqualityConstrainedSolver{

  def apply(
    objF:ObjectiveFunction,
    C:ConvexSet,
    startingPoint:DenseVector[Double],
    A:DenseMatrix[Double],
    b:DenseVector[Double],
    pars:SolverParams,
    logger:Logger
  ):EqualityConstrainedSolver = new EqualityConstrainedSolver(objF,C,startingPoint,A,b,pars,logger)


}
