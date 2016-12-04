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
abstract class TestFunction(val dim:Integer) {

    def id:String
    /** Function phi_j as in docs/cvx_notes, section Hessian, example 1.*/
    def _objF(x:DenseVector[Double]):Double
    /** Derivative of function phi_j.*/
    def _gradF(x:DenseVector[Double]):DenseVector[Double]
    /** Second derivative of function phi_j.*/
    def _hessF(x:DenseVector[Double]):DenseMatrix[Double]

    /** Minimum value of the objective function: sum_j\phi_j(0).*/
    def isMinimizer(x:DenseVector[Double],tol:Double):Boolean
    /** Minimum value of the objective function: sum_j\phi_j(0).*/
    def globalMin:Double

    /** Unconstrained solver for minimizing the the test function over the convex set C.
      * Note: at least one global solution should be in C, so that unconstrained minimization
      * is reasonable.
      * The idea is to mostly use $C=R^n$.
      */
    def solver(x0:DenseVector[Double], C:ConvexSet):UnconstrainedSolver =
        new UnconstrainedSolver(x0:DenseVector[Double], C:ConvexSet){

            def objF(x:DenseVector[Double]) =_objF(x)
            def gradF(x:DenseVector[Double]) = _gradF(x)
            def hessF(x:DenseVector[Double]) = _hessF(x)
        }
}


object TestFunction {

    /** f(x) = (1/2)*(x dot x).*/
    def normSquared(dim:Int):TestFunction = new TestFunction(dim){

            def id = "f(x) = 0.5*||x||^2  in dimension "+dim
            def _objF(x:DenseVector[Double]) = 0.5*(x dot x)
            def _gradF(x:DenseVector[Double]) = x
            def _hessF(x:DenseVector[Double]) = DenseMatrix.eye[Double](dim)

            def isMinimizer(x:DenseVector[Double],tol:Double) = norm(x)<tol
            def globalMin = 0.0
        }
}
