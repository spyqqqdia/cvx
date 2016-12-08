package cvx

import breeze.linalg.{DenseMatrix, DenseVector, _}


// ToDo: UnconstrainedSolver: x0 must become an option, must get method ::startingPoint.
//
// WARNING: 
// in the barrier method handle the multiplications with the dimension reducing
// matrix x = x0+Fu _outside_ the sum in the barrier function (bilinear!) or else we will matrix
// multiply ourselves to death.
// This is the reason why we do not put this operation into the constraints themselves.

/** Solver for constrained convex optimization using the barrier method.
 *
 * @param C open convex set known to contain the minimizer
 */
abstract class BarrierSolver(val C:ConvexSet)
extends Solver {
 
    val dim=C.dim

	def startingPoint:DenseVector[Double]
	def barrierFunction(t:Double,x:DenseVector[Double]):Double
    def gradientBarrierFunction(t:Double,x:DenseVector[Double]):DenseVector[Double]
	def hessianBarrierFunction(t:Double,x:DenseVector[Double]):DenseMatrix[Double]
	/** Number m of inequality constraints. */
	def numConstraints:Int

	def checkDim(x:DenseVector[Double]):Unit =
		assert(x.length==dim,"Dimension mismatch x.length="+x.length+" unequal to dim="+dim)

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
	  * @return Solution object: minimizer with additional info.
	  */
	def solve(maxIter:Int,alpha:Double,beta:Double,tol:Double,delta:Double):Solution = {
	
	    val mu = 10.0    // factor to increase parameter t in barrier method.
		var t = 1.0
		var x = startingPoint       // iterates x=x_k
		var sol:Solution = null   // solutions at parameter t
	    while(numConstraints/t>=tol){
		
		    // solver for barrier function at fixed parameter t
			val objF_t = new ObjectiveFunction(dim){

				def valueAt(x:DenseVector[Double]) = { checkDim(x); barrierFunction(t,x) }
				def gradientAt(x:DenseVector[Double]) = { checkDim(x); gradientBarrierFunction(t,x) }
				def hessianAt(x:DenseVector[Double]) = { checkDim(x); hessianBarrierFunction(t,x) }
			}
			val solver = new UnconstrainedSolver(objF_t,C){
			
			    override def startingPoint = x
			}
		    sol = solver.solve(maxIter,alpha,beta,tol,delta)
			x = sol.x
			t = mu*t
		}
	    sol
	}
}


/** Some factory functions.*/
object BarrierSolver {


	/** BarrierSolver when there are no equality constraints.
	  * @param constraints list of inequality constraints
      */
	def solverNoEqualities(constraints: List[Constraint]):BarrierSolver = {

		if(constraints.isEmpty) throw new IllegalArgumentException("Use an UnconstrainedSolver")
		// check if all constraints have the same dimension
		val dim = constraints(0).dim
		assert(constraints.forall(cnt => cnt.dim==dim))

		val C = StrictlyFeasibleSet(dim,constraints)

		/*new BarrierSolver(C) {

            // FIX ME

		}*/
        null

	}

}