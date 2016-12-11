package cvx

import breeze.linalg.{DenseVector, _}



/** A point in a convex set.*/
trait SamplePoint {

    def samplePoint:DenseVector[Double]
}


/**
  * Created by oar on 12/1/16.
  *
  * Abstraction of a convex set. Needless to say we cannot check that the membership
  * condition specifies a convex set. So the class name points to the intended application only.
  */
abstract class ConvexSet(val dim:Int) {

    def isInSet(x:DenseVector[Double]):Boolean
}

object ConvexSet {

    def addSamplePoint(C:ConvexSet,x:DenseVector[Double]): ConvexSet with SamplePoint = {

        assert(C.dim==x.length)
        assert(C.isInSet(x))
        new ConvexSet(C.dim) with SamplePoint {

            def isInSet(u:DenseVector[Double]) = C.isInSet(u)
            def samplePoint = x
        }
    }
    def fullSpace(dim:Int) = new ConvexSet(dim) with SamplePoint {

        def isInSet(x:DenseVector[Double]):Boolean = true
        /**Random point x with x_j = j*(10.0/dim)*u_j, where u_j is uniform in (-1,1).*/
        def samplePoint = DenseVector.tabulate[Double](dim){j => 10*j*(2*rand()-1)/dim}
    }
}


/** Set of points which strictly satisfy all constraints in cnts.
  *
  * @param dim common dimension of all constraints in cnts
  * @param cnts a set of constraints all in the same dimension
  */
class StrictlyFeasibleSet(override val dim:Int,val cnts:ConstraintSet) extends ConvexSet(dim) {

    // check that all constraints have the same dimension
    assert(cnts.constraints.forall(cnt => cnt.dim==dim))
    // this is called often in line search, may have to do this more efficiently
    def isInSet(x:DenseVector[Double]):Boolean =
        cnts.constraints.forall(cnt => cnt.isSatisfiedStrictly(x))
}
object StrictlyFeasibleSet {

    // factory method
    def apply(dim:Int, cnts:ConstraintSet):StrictlyFeasibleSet = new StrictlyFeasibleSet(dim,cnts)
}