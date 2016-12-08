package cvx

import breeze.linalg.{DenseMatrix, DenseVector, _}


/**
  * Created by oar on 12/1/16.
  *
  * Class computes the affine space of all solutions x to a linear equation Ax=b, where A
  * is an mxn matrix with m<n of full rank (hence the system Ax=b is underdetermined).
  *
  * The solutions x will be represented in the form x=x0+Fu, where x0 is a particular
  * solution and the matrix F maps onto ker(A), more precisely, the columns of F are an
  * orthonormal basis of ker(A).
  * In particular thus one has AF=0 (equivalently: Im(F) is contained in ker(A)).
  *
  * The solution space will be represented by the pair (F,x0).
  */

class SolutionSpace(val F:DenseMatrix[Double] , val x0:DenseVector[Double])

object SolutionSpace {

    /** Solves Ax=b as x=x0+Fu, where Im(F)=ker(A) and x0 is a special solution.
      * A is assumed to have full rank (not checked!). Thus a solution always exists.
      *
      * @param A matrix of full rank with m=A.rows < n=A.cols
      * @param b vector of length
      * @return SolutionSpace(F,x0)
      */
    def apply(A:DenseMatrix[Double], b:DenseVector[Double]){

        val m = A.rows; val n = A.cols
        assert(m<n, "System Ax=b is not underdetermined. Other solution methods apply.")
        assert(m==b.length, "length(b) must be = rows(A)")
        // compute the complete QR factorization (Q a full ON basis of the whole space, R with zero rows on bottom)
        // recall (a until b)=[a,b)
        val qrAt = qr(A.t); val Qc = qrAt.q; val R = qrAt.r(0 until m,::)

        val F = Qc(::,m until n)
        val Q = Qc(::,0 until m)
        val y = MatrixUtils.forwardSolve(R.t,b)
        val x0 = Q*y

        new SolutionSpace(F,x0)
    }
}
