package cvx

import breeze.linalg.{DenseMatrix, DenseVector}
import MatrixUtils._

/**
  * Created by oar on 2/4/17.
  * Quadratic function f(x) of the form f(x) = r + x'a + (1/2)*x'Px,
  * where P is a square symmetric matrix
  */
class QuadraticObjectiveFunction(

        override val dim:Int,
        val r:Double,
        val a:DenseVector[Double],
        val P:DenseMatrix[Double]
)
extends ObjectiveFunction(dim) {

    if(a.length!=dim){
        val msg = "Vector a must be of dimension "+dim+" but length(a) "+a.length
        throw new IllegalArgumentException(msg)
    }
    if(!(P.rows==dim & P.cols==dim)) {

        val msg = "Matrix P must be square of dimension "+dim+" but is "+P.rows+"x"+P.cols
        throw new IllegalArgumentException(msg)
    }
    checkSymmetric(P,1e-13)

    def valueAt(x:DenseVector[Double]) = { checkDim(x); r + (a dot x) + (x dot (P*x))/2 }
    def gradientAt(x:DenseVector[Double]) = { checkDim(x); a+P*x }
    def hessianAt(x:DenseVector[Double]) = { checkDim(x); P }

}
