package cvx

import breeze.linalg.{DenseMatrix, DenseVector, max, min, norm}
import breeze.numerics.abs

/**
  * Created by oar on 15.11.17.
  *
  * Holds data H,A,g,r of the KKT system
  *
  *    Hx + A'nu = -g
  *    Ax        = r
  *
  * with KKT matrix
  *
  * | H, A'|
  * | A, 0 |
  *
  * and (primal and dual) variables x,nu. This is a KKTsystem
  * without inequality constraints as it occurs in solutions of
  * convex problems with the barrier method. Recall that the barrier
  * method absorbs the inequality constraints into the objective function.
  *
  * This class has a method to eliminate zero equations as they can occur
  * in the KKTSystems of phase I analysis.
  *
  * @param nullIndices vector of indices j such that the j-th row and column of H,
  *                    the j-th column of A and g_j are all zero, see method
  *                    [reduced] for explanation
  *
  */
class KKTData(
  val H:DenseMatrix[Double], val A:DenseMatrix[Double],
  val g:DenseVector[Double], val r:DenseVector[Double],
  val nullIndices: Option[Vector[Int]]
) {

  private val n = H.cols
  assert(H.rows==n,"Matrix H not square, n=H.cols="+n+", H.rows="+H.rows)
  assert(A.cols==n,"A.cols="+A.cols+" not equal to n=M.rows="+H.rows)

  /** In principle we can set up a convex problem where neither the
    * objective function f(x) nor any of the inequality constraints g_i(x)<=0
    * or of the equality constraints Ax=r depend on a certain variable
    * x_j.
    *
    * This happens e.g. in phase I analysis whenever the variable x_j is
    * unconstrained (by either inequalities or equalities).
    *
    * In that case the j-th equation in the KKT system is 0'x+0'nu=0
    * and can be eliminated. Moreover the system does not depend on the variable
    * x_j which can therefore be eliminated also (i.e. the j-th column in the KKT
    * matrix is zero and can be eliminated).
    *
    * However we need to keep track of which variables x_j are eliminated since
    * (arbitrary) values for these need to be reinserted into the solution.
    *
    * If the j-th row of H and A' are zero and g_j=0, then the j-th row and column
    * are deleted from H, the j-th row is deleted from A' (i.e. the j-th column
    * deleted from A) and the j-th coordinate deleted from g and the index j is stored
    * in a vector in increasing order.
    *
    * If the j-th row of H and A' are zero and g_j is not zero an UnsolvableSystemException
    * is thrown.
    *
    * This function does this for all j=0,...H.rows-1 and returns the reduced system.
    */
  def reduced:KKTData = {

    var theNullIndices = Vector[Int]()
    var I = Vector[Int]()                  // indices of nonnull rows
    for(j <- 0 until n){

      val t = norm(H(::,j))+norm(A(::,j))
      if(t>0) I = I:+j
      else
        if(abs(g(j))<1e-15) theNullIndices = theNullIndices:+j
      else {

        val msg = "\nUnsolvable KKT system, row "+j+" is zero with nonzero right hand side."
        throw UnsolvableSystemException(msg)
      }
    }
    if(theNullIndices.isEmpty)
      KKTData(H,A,g,r,None)
    else {

      val red_H = H(I,I).toDenseMatrix
      val red_A = A(::,I).toDenseMatrix
      val red_g = g(I).toDenseVector
      KKTData(red_H,red_A,red_g,r,Some(theNullIndices))
    }
  }
}



object KKTData {

  def apply(
     H: DenseMatrix[Double], A: DenseMatrix[Double],
     g: DenseVector[Double], r: DenseVector[Double],
     nullIndices: Option[Vector[Int]]
           ): KKTData = new KKTData(H, A, g, r, nullIndices)

  /** x is interpreted to be part of a larger vector z where all the
    * coordinates z_j with j in nullIndices have been deleted (removed).
    *
    * @param nullIndices vector of indices of deleted coordinates, must be sorted
    *                    in increasing order.
    * @return vector z with zeros in the deleted coordinates j in nullIndices.
    */
  def paddVector(x:DenseVector[Double],nullIndices:Vector[Int]): DenseVector[Double] = {

    val n = nullIndices.length
    assert(max(nullIndices)<x.length+n)
    val z = DenseVector.zeros[Double](x.length+n)
    var i=0    // count up the x-coordinates
    var k=0    // count up the nullIndices
    var j=0    // count up the z-coordinates
    while(j<z.length){

      if(j==nullIndices(min(k,n-1))) k+=1; else{ z(j)=x(i); i+=1; }
      j+=1
    }
    z
  }

}






