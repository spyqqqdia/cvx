package cvx

import breeze.linalg.{DenseMatrix, DenseVector, _}

/**
  * Created by oar on 12/2/16.
  *
  * Convex function f(x) which knows its global minimum (both value and location).
  * Since the location need not be unique it will be represented as a boolean check
  * (indicator function of the solution set).
  *
  * @param dim: number of independent variables x_j.
  */
abstract class TestFunction(override val dim:Int) extends ObjectiveFunction(dim) {

    def id:String


    /** Minimum value of the objective function: sum_j\phi_j(0).*/
    def isMinimizer(x:DenseVector[Double],tol:Double):Boolean
    /** Minimum value of the objective function: sum_j\phi_j(0).*/
    def globalMin:Double

    /** Unconstrained solver for minimizing the the test function over the convex set C.
      * Note: at least one global solution should be in C, so that unconstrained minimization
      * is reasonable.
      * The idea is to mostly use the full Euclidean space.
      */
    def solver(C:ConvexSet with SamplePoint):UnconstrainedSolver = new UnconstrainedSolver(this,C)
}


object TestFunction {

    /** f(x) = (1/2)*(x dot x).*/
    def normSquared(dim:Int):TestFunction = new TestFunction(dim){

            def id = "f(x) = 0.5*||x||^2  in dimension "+dim
            def valueAt(x:DenseVector[Double]) = 0.5*(x dot x)
            def gradientAt(x:DenseVector[Double]) = x
            def hessianAt(x:DenseVector[Double]) = DenseMatrix.eye[Double](dim)

            def isMinimizer(x:DenseVector[Double],tol:Double) = norm(x)<tol
            def globalMin = 0.0
        }
}
