package cvx

import breeze.linalg.{DenseMatrix, DenseVector, NotConvergedException, _}

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
  */
class UnconstrainedSolver(val objF:ObjectiveFunction, val C:ConvexSet with SamplePoint)
extends Solver {

    /** @return C.samplePoint, override as needed.*/
    def startingPoint: DenseVector[Double] = C.samplePoint

    /** Find the location $x$ of the minimum of f=objF over C by the newton method
      * starting from the starting point x0.
      *
      * @param maxIter : maximal number of Newton steps computed.
      * @param alpha   : line search descent factor
      * @param beta    : line search backtrack factor
      * @param tol     : termination as soon as both the norm of the gradient is less than tol and
      *                the Newton decrement satisfies $l=\lambda(x)$ satisfies $l^2/2 < tol$.
      *                Recall that $l^2/2$ indicates the distance of f(x) from the optimal value.
      * @param delta   : if hessF(x) is close to singular, then the regularization hessF(x)+delta*I
      *                is used to compute the Newton step
      *                (this can be interpreted as restricting the step to a trust region,
      *                See docs/cvx_notes.tex, section Regularization).
      *
      *                Distance from singularity of the Hessian will be determined from the size of the smallest
      *                diagonal element of the Cholesky factor hessF(x)=LL'.
      *                If this is smaller than sqrt(delta), the regularization will be applied.
      *
      * @return Solution object: minimizer with additional info
      */
    def solve(maxIter:Int,alpha:Double,beta:Double,tol:Double,delta:Double):Solution = {

        val breakDown = NotConvergedException.Breakdown
        var iter = 0
        var newtonDecrement = tol + 1
        var x = startingPoint
        var y = objF.gradientAt(x)
        var normGrad = norm(y)
        while (iter < maxIter && newtonDecrement > tol && normGrad > tol) {

            val f = objF.valueAt(x)
            val H = objF.hessianAt(x)
            val n = H.rows

            val d = MatrixUtils.solveWithPreconditioning(H,-y,delta)        // newton  step

            val q = d dot y
            newtonDecrement = -q/2
            if(q>0){

                var msg = "Newton step d is not a descent direction: (d dot gradF(x)) = "+q
                msg+="\nx:\n"+x+"\ngradF(x):\n"+y+"\nhessF(x):\n"+H+"\nNewton step d:\n"+d
                throw new NotConvergedException(breakDown,msg)
            }
            //continue only if newtonDecrement > eps
            if(newtonDecrement > tol){

                // backtracking line search
                var it = 0 // safeguard against bugs in gradient
                var t = 1.0
                var dx = d
                while (!C.isInSet(x + dx)) {
                    t *= beta; dx *= beta; it += 1
                }
                while ((objF.valueAt(x + dx) > f + alpha * t * q) && (it < 100)) {
                    t *= beta; dx *= beta; it += 1
                }

                // cannot happen unless gradient is messed up
                if (it == 100) throw new NotConvergedException(
                    breakDown, "Line search: sufficient decrease not reached after 100 iterations, gradient messed up??"
                )

                // step to next iterate
                x = x + dx
                y = objF.gradientAt(x)
                normGrad = norm(y)
            }
            iter+=1
        }
        Solution(x,newtonDecrement,normGrad,iter,iter==maxIter)
    }
}
