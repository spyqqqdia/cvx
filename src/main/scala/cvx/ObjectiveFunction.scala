package cvx

import breeze.linalg.{DenseMatrix, DenseVector, _}

/** Objective function (should be convex, twice continuously differentiable).
 *  Provides value, gradient and hessian of self.
 */
abstract class ObjectiveFunction(val dim:Int){

    def valueAt(x:DenseVector[Double]):Double
	def gradientAt(x:DenseVector[Double]):DenseVector[Double]
    def hessianAt(x:DenseVector[Double]):DenseMatrix[Double]
}



/** Examples of objective functions (test cases).*/
object ObjectiveFunctions {

    def normSquared(dim:Int) = new ObjectiveFunction(dim) {

        def valueAt(x:DenseVector[Double]) = 0.5*(x dot x)
        def gradientAt(x:DenseVector[Double]) = x
        def hessianAt(x:DenseVector[Double]) = DenseMatrix.eye[Double](dim)
    }



}