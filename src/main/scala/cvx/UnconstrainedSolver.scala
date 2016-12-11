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
  * @param pars see [SolverParams].
  */
class UnconstrainedSolver(
     val objF:ObjectiveFunction, val C:ConvexSet with SamplePoint, val pars:SolverParams
) extends Solver {

    /** @return C.samplePoint, override as needed.*/
    def startingPoint: DenseVector[Double] = C.samplePoint

    /** Find the location $x$ of the minimum of f=objF over C by the newton method
      * starting from the starting point x0.
      *
      is smaller than sqrt(delta), the regularization will be applied.
      *
      * @return Solution object: minimizer with additional info
      */
    def solve():Solution = {

        val maxIter = pars.maxIter; val alpha=pars.alpha; val beta=pars.beta
        val tol=pars.tol; val delta=pars.delta

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
                    breakDown, "Line search: sufficient decrease not reached after 100 iterations"
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
object UnconstrainedSolver {

    def apply(objF:ObjectiveFunction, C:ConvexSet with SamplePoint, pars:SolverParams): UnconstrainedSolver =
        new UnconstrainedSolver(objF,C,pars)

    /** Unconstrained solver in the variable u with C being the whole space in dimension dim(u), where
      * the solutions to the equality constraints Ax=b are parameterized as x=z0+Fu, see [EqualityConstraints].
      * The solver reports the solution x not u.
      *
      * @param pars solver parameters, see [SolverParams].
      * @return
      */
    def apply(objF:ObjectiveFunction, eqs:EqualityConstraints,pars:SolverParams): UnconstrainedSolver = {

        val z0 = eqs.z0
        val F = eqs.F
        val dim_u = F.cols
        val C = ConvexSet.fullSpace(dim_u)
        // ovverride solve to report x instead of u
        new UnconstrainedSolver(objF,C,pars) {

            override def solve = {

                val sol = super.solve
                val u = sol.x; val x = z0+F*u
                Solution(x, sol.gap, sol.normGrad, sol.iter, sol.maxedOut)
            }
        }
    }

}
