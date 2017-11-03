package cvx

import breeze.linalg.{DenseVector, norm}

/** Solution to an optimization problem, both location and value of the minimum.
  * Since the location of the minimum may not be unique it is formulated as a boolean check
  * (indicator function of the solution set).
  */
trait KnownMinimizer {

  /** Minimum value of the objective function.*/
  def isMinimizer(x:DenseVector[Double],tol:Double):Boolean
  /** Minimum value of the objective function.*/
  def minimumValue:Double
}
object KnownMinimizer {

  /** x0 is a minimizer for objF on domain specified in context.*/
  def apply(x0:DenseVector[Double],objF:ObjectiveFunction) = new KnownMinimizer {

    val y0:Double = objF.valueAt(x0)

    /**
      * @return true if f(x) < min(f)+tol, false otherwise
      */
    def isMinimizer(x:DenseVector[Double],tol:Double):Boolean = {

      val y = objF.valueAt(x)
      norm(y-y0) < tol
    }
    def minimumValue:Double = y0
    /** With x0 being the specified minimizer
      * @return true if ||x-x0||<tol, false otherwise.
      *
      * This is useful if the minimizer is uniquely determined. In this
      * case it makes sense to check if the solution x is close to the known
      * unique minimizer in addition to checking that f(x) < min(f)+tol.
      */
    def isEqualToSpecifiedMinimizer(x:DenseVector[Double],tol:Double):Boolean = norm(x-x0)<tol
  }

}