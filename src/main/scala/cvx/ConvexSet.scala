package cvx

import breeze.linalg.{DenseVector, _}



/**
  * Created by oar on 12/1/16.
  *
  * Abstraction of a convex set. Needless to say we cannot check that the membership
  * condition specifies a convex set. So the class name points to the intended application only.
  */
abstract class ConvexSet(val dim:Int) {

  self:ConvexSet =>

  for(x <- samplePoint) {
    assert(x.length == dim,
      "ConvexSet C: dimension mismatch: C.dim = " + dim + ", samplePoint.dim = " + x.length + "\n")
    assert(isInSet(x),
      "ConvexSet C: samplePoint x is not in set, x:\n"+x+"\n")
  }

  def isInSet(x:DenseVector[Double]):Boolean
  def samplePoint: Option[DenseVector[Double]]

  /** Cartesian product of this set with R (in this order).
    * The sample point sets the new (last) coordinate to Double.MaxValue which is useful
    * for phase_I analysis.
    * */
  def cross_R = new ConvexSet(self.dim+1){

    override def isInSet(x:DenseVector[Double]):Boolean = self.isInSet(x(0 until self.dim))
    override def samplePoint:Option[DenseVector[Double]] = self.samplePoint match {

      case Some(x) => Some( DenseVector.tabulate[Double](1+self.dim){i => if(i<self.dim) x(i) else Double.MaxValue } )
      case _ => None
    }

  }
}


object ConvexSet {

  def addSamplePoint(C:ConvexSet,x:DenseVector[Double]): ConvexSet = {

    assert(C.dim==x.length)
    assert(C.isInSet(x))
    new ConvexSet(C.dim) {

      def isInSet(u:DenseVector[Double]):Boolean = C.isInSet(u)
      def samplePoint = Some(x)
    }
  }
  def fullSpace(dim:Int) = new ConvexSet(dim) {

    def isInSet(x:DenseVector[Double]):Boolean = true
    /**Random point x with x_j = j*(10.0/dim)*u_j, where u_j is uniform in (-1,1).*/
    def samplePoint = Some(DenseVector.tabulate[Double](dim){j => 10*j*(2*rand()-1)/dim})
  }
}


/** Set of points which strictly satisfy all constraints in cnts.
  *
  * @param cnts a set of constraints all in the same dimension
  */
class StrictlyFeasibleSet(val cnts:ConstraintSet) extends ConvexSet(cnts.dim) {

  // check that all constraints have the same dimension
  assert(cnts.constraints.forall(cnt => cnt.dim==dim))
  // this is called often in line search, may have to do this more efficiently
  def isInSet(x:DenseVector[Double]):Boolean =
  cnts.constraints.forall(cnt => cnt.isSatisfiedStrictly(x))

  def samplePoint:Option[DenseVector[Double]] = None
}
object StrictlyFeasibleSet {

  // factory method
  def apply(cnts:ConstraintSet):StrictlyFeasibleSet = new StrictlyFeasibleSet(cnts)
  def apply(cnts:ConstraintSet,feasiblePoint:DenseVector[Double]):StrictlyFeasibleSet =
    new StrictlyFeasibleSet(cnts){

      assert(
        cnts.isSatisfiedStrictlyBy(feasiblePoint),
        "\nStrictlyFeasibleSet: feasible point does not satisfy all constraints strictly.\n"
      )
      override def samplePoint = Some(feasiblePoint)
    }
}