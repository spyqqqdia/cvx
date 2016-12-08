package cvx

import breeze.linalg.{DenseVector, _}

/**
  * Created by oar on 12/1/16.
  *
  * Abstraction of a convex set. Needless to say we cannot check that the membership
  * condition specifies a convex set. So the class name points to the intended application only.
  */
abstract class ConvexSet(val dim:Int) {

    def isInSet(x:DenseVector[Double]):Boolean
    def samplePoint:DenseVector[Double]

}
class FullSpace(override val dim:Int) extends ConvexSet(dim) {

    def isInSet(x:DenseVector[Double]):Boolean = true
    /**Random point x with x_j = j*(10.0/dim)*u_j, where u_j is uniform in (-1,1).*/
    def samplePoint = DenseVector.tabulate[Double](dim){j => 10*j*(2*rand()-1)/dim}


}

