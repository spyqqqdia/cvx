package cvx

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * Created by vagrant on 12.03.18.
  */
object ConvexSets {

  def wholeSpace(dim:Int):ConvexSet = new ConvexSet(dim){

    override def isInSet(x:DenseVector[Double]):Boolean = true
    override def samplePoint:Option[DenseVector[Double]] = Some(DenseVector.zeros[Double](dim))
  }
  /** The region where all coordinates are strictly positive
    */
  def firstQuadrant(dim:Int):ConvexSet = new ConvexSet(dim){

    override def isInSet(x:DenseVector[Double]):Boolean = x.forall(_>0)
    override def samplePoint:Option[DenseVector[Double]] =
      Some(DenseVector.tabulate[Double](dim)(j => 1.0/dim))
  }

  /** Set of points which strictly satisfy all constraints in cnts.
    *
    * @param cnts a set of constraints all in the same dimension
    */
  def strictlyFeasibleSet(cnts:ConstraintSet):ConvexSet = new ConvexSet(cnts.dim) {

    // check that all constraints have the same dimension
    assert(cnts.constraints.forall(cnt => cnt.dim==dim))
    // this is called often in line search, may have to do this more efficiently
    override def isInSet(x:DenseVector[Double]):Boolean =
      cnts.constraints.forall(cnt => cnt.isSatisfiedStrictly(x))

    override def samplePoint:Option[DenseVector[Double]] = None
  }
  /** Set of points which strictly satisfy all constraints in cnts.
    *
    * @param cnts a set of constraints all in the same dimension
    */
  def strictlyFeasibleSet(
    cnts:ConstraintSet,feasiblePoint:DenseVector[Double]
  ):ConvexSet = new ConvexSet(cnts.dim) {

    // check that all constraints have the same dimension
    assert(cnts.constraints.forall(cnt => cnt.dim==dim))
    // this is called often in line search, may have to do this more efficiently
    override def isInSet(x:DenseVector[Double]):Boolean =
      cnts.constraints.forall(cnt => cnt.isSatisfiedStrictly(x))

    override def samplePoint:Option[DenseVector[Double]] = Some(feasiblePoint)
  }

  /** Cartesian product CxD of the convex sets C,D.
    */
  def cartesianProduct(C:ConvexSet,D:ConvexSet):ConvexSet = {

    val n = C.dim
    val m = D.dim
    val opt_c = C.samplePoint
    val opt_d = D.samplePoint

    if(opt_c.isEmpty || opt_d.isEmpty){

      new ConvexSet(C.dim+D.dim){

        override val samplePoint = None
        override def isInSet(x:DenseVector[Double]):Boolean =
          C.isInSet(x(0 until n)) && D.isInSet(x(n until n+m))
      }
    } else {

      // sample points for C and D
      val c = opt_c.get
      val d = opt_d.get
      val x0 = DenseVector.vertcat(c,d)

      new ConvexSet(C.dim+D.dim){

        override def samplePoint = Some(x0)
        override def isInSet(x:DenseVector[Double]):Boolean =
          C.isInSet(x(0 until n)) && D.isInSet(x(n until n+m))
      }
    }
  }
  /** Preimage of C under the affine transformation u -> x = z+Fu
    */
  def affinePreimage(C:ConvexSet,z:DenseVector[Double],F:DenseMatrix[Double]):ConvexSet = {

    val logger = Logger("ConvexSet.affinePreimage.txt")
    val tol = 1e-8
    val debugLevel = 0
    val newDim = F.cols // variable is u in x = z+Fu
    // z+Fu0 = x0
    val x0:Option[DenseVector[Double]] = C.samplePoint
    val u0 = if(x0.isEmpty)
               None
             else
               Some(MatrixUtils.svdSolve(F,x0.get-z,logger,tol,debugLevel))

    new ConvexSet(F.cols){

      val samplePoint:Option[DenseVector[Double]] = u0
      override def isInSet(u:DenseVector[Double]):Boolean = C.isInSet(z+F*u)
    }
  }

}
