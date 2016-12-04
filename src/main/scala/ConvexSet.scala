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

}
class FullSpace(override val dim:Int) extends ConvexSet(dim) {

    def isInSet(x:DenseVector[Double]):Boolean = true

}

