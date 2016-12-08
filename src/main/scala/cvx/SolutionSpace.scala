package cvx

import breeze.linalg.{DenseMatrix, DenseVector, _}


/**
  * Created by oar on 12/1/16.
  *
  * Class computes the affine space of all solutions x to a linear equation Ax=b, where A
  * is an mxn matrix with m<n of full rank (hence the system Ax=b is underdetermined).
  *
  * The solutions x will be represented in the form x=z0+Fu, where z0 is the minimum norm
  * solution of Ax=b and the matrix F maps onto ker(A), more precisely, the columns of F are an
  * orthonormal basis of ker(A).
  * In particular thus one has AF=0 (equivalently: Im(F) is contained in ker(A)).
  *
  * The solution space will be represented by the pair (F,z0).
  */

class SolutionSpace(val z0:DenseVector[Double], val F:DenseMatrix[Double]){

    assert(F.rows==z0.length)
    // check that F has orthonormal columns
    val H = F.t*F
    val I = DenseMatrix.eye[Double](F.cols)
    assert(Math.sqrt(sum(H-I))<F.rows*1e-14)

    /** If Ax0=b then x0 = z0 + Fu0 and so, since F'F=I (orthonormal columns) we have
      * F'(x0-z0)=F'Fu0=u0.
      * @return if Ax0=b returns u0 such that x0 = z0 + Fu0.
      */
    def parameter(x0:DenseVector[Double]) = F.t*(x0-z0)
}

object SolutionSpace {

    def apply(z0:DenseVector[Double],F:DenseMatrix[Double]) = new SolutionSpace(z0,F)

    /** Solves Ax=b as x=x0+Fu, where Im(F)=ker(A) and x0 is a special solution.
      * A is assumed to have full rank (not checked!). Thus a solution always exists.
      *
      * @param A matrix of full rank with m=A.rows < n=A.cols
      * @param b vector of length
      * @return SolutionSpace(F,x0)
      */
    def solve(A:DenseMatrix[Double], b:DenseVector[Double]):SolutionSpace = {

        val m = A.rows; val n = A.cols
        assert(m<n, "System Ax=b is not underdetermined. Other solution methods apply.")
        assert(m==b.length, "length(b) must be = rows(A)")
        // compute the complete QR factorization (Q a full ON basis of the whole space, R with zero rows on bottom)
        // recall (a until b)=[a,b)
        val qrAt = qr(A.t); val Qc = qrAt.q; val R = qrAt.r(0 until m,::)

        val F = Qc(::,m until n)
        val Q = Qc(::,0 until m)
        val y = MatrixUtils.forwardSolve(R.t,b)
        val z0 = Q*y

        new SolutionSpace(z0,F)
    }
}
