package cvx

import breeze.linalg.{DenseMatrix, DenseVector}

/** Examples of objective functions (test cases).*/
object ObjectiveFunctions {

    def normSquared(dim:Int) = new ObjectiveFunction(dim) {

        def valueAt(x:DenseVector[Double]) = 0.5*(x dot x)
        def gradientAt(x:DenseVector[Double]) = x
        def hessianAt(x:DenseVector[Double]) = DenseMatrix.eye[Double](dim)
    }

}