package cvx

import breeze.linalg.{DenseVector, norm}

/** Solution to an optimization problem, both location and value of the minimum.
  * Since the location of the minimum may not be unique it is formulated as a boolean check
  * (indicator function of the solution set).
  */
trait KnownMinimizer {

  def theMinimizer:DenseVector[Double]
  /** Does x minimize the objective function.*/
  def isMinimizer(x:DenseVector[Double],tol:Double):Boolean
  /** Minimum value of the objective function.*/
  def minimumValue:Double

  /**
    * @param sol the computed solution
    * @param objFcnValue value of objective function at computed solution
    * @param tol tolerated deviation from known solution (l2-norm)
    * @param logger
    */
  def reportWithKnownSolution(sol:Solution,objFcnValue:Double,tol:Double,logger:Logger):Unit = {

    val x = sol.x                       // minimizer, solution found
    val y_opt = minimumValue

    val newtonDecrement = sol.dualityGap      // Newton decrement at solution
    val normGrad = sol.normGrad        // norm of gradient at solution
    val iter = sol.iter
    val maxedOut = sol.maxedOut
    val isSolution = isMinimizer(x,tol)
    val knownSolution = theMinimizer

    val msg = s"${sol}\n"+
      s"value at computed solution y=f(x):  ${MathUtils.round(objFcnValue,9)}\n"+
      s"value of global min:  ${MathUtils.round(y_opt,9)}\n"+
      s"known minimizer:\n${MathUtils.roundVec(knownSolution,d=9)}\n"+
      s"Computed solution x:\n${MathUtils.roundVec(x,d=9)}\n"+
      s"Is global solution at tolerance ${tol}: ${isSolution}\n"

    print(msg)
    Console.flush()
    logger.println(msg)
    logger.close()
  }
}
object KnownMinimizer {

  /** x0 is a minimizer for objF on domain specified in context.*/
  def apply(x0:DenseVector[Double],objF:ObjectiveFunction) = new KnownMinimizer {

    val theMinimizer:DenseVector[Double] = x0
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