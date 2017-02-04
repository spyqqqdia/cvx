package cvx

import breeze.linalg.{DenseVector, norm}

/** Solution to an optimization problem, both location and value of the minimum.
  * Since the location of the minimum may not be unique it is formulated as a boolean check
  * (indicator function of the solution set).
  */
trait KnownMinimizer {

    /** Minimum value of the objective function: sum_j\phi_j(0).*/
    def isMinimizer(x:DenseVector[Double],tol:Double):Boolean
    /** Minimum value of the objective function: sum_j\phi_j(0).*/
    def minimumValue:Double
}
object KnownMinimizer {

    /** Uniquely determined solution at x=x0 with value objF(x0)=y0.*/
    def apply(x0:DenseVector[Double],y0:Double) = new KnownMinimizer {

        def isMinimizer(x:DenseVector[Double],tol:Double) = norm(x-x0) < tol
        def minimumValue = y0
    }

}