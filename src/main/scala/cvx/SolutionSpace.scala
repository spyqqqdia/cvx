package cvx

import breeze.linalg.{DenseMatrix, DenseVector, _}


/**
  * Created by oar on 12/1/16.
  *
  * Class computes the affine space of all solutions x to a linear equation Ax=b, where A
  * is an mxn matrix with m<n of full rank (hence the system Ax=b is underdetermined).
  *
  * The rank condition will not be checked.
  *
  * The solutions x will be represented in the form x=z0+Fu, where z0 is the minimum norm
  * solution of Ax=b and the matrix F maps onto ker(A), more precisely, the columns of F are an
  * orthonormal basis of ker(A).
  *
  * In particular thus one has AF=0 (equivalently: Im(F) is contained in ker(A)).
  */
class SolutionSpace(val A:DenseMatrix[Double], val b:DenseVector[Double]){

    assert(A.rows==b.length)
    assert(A.rows < A.cols)
    val sol = MatrixUtils.solveUnderdetermined(A,b)
    val z0 = sol._1
    val F = sol._2

    /** If Ax0=b then x0 = z0 + Fu0 and so, since F'F=I (orthonormal columns) we have
      * F'(x0-z0)=F'Fu0=u0.
      * @return if Ax0=b returns u0 such that x0 = z0 + Fu0.
      */
    def parameter(x0:DenseVector[Double]) = F.t*(x0-z0)
}

object SolutionSpace {

    def apply(A:DenseMatrix[Double], b:DenseVector[Double]) = new SolutionSpace(A,b)
}
