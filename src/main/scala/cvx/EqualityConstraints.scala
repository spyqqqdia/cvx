package cvx

import breeze.linalg.{DenseMatrix, DenseVector, _}


/** Affine equality constraints of the form Ax=b, where A is mxn with m < n
 *  and full rank m. The condition rank(A)=m will not be checked.
 
 *  Parameterizes solutions as x = z0+Fu, where x=z0 is the minimum norm solution
 *  and $Fu\perp z0$, for all u in dimension n-m.
 *  Used for change of variables x --> u to reduce dimension and get rid of explicit 
 *  equality constraints.
 */
class EqualityConstraints(val A:DenseMatrix[Double], val b:DenseVector[Double]){

    assert(A.rows < A.cols, "m=A.rows="+A.rows+" is not less than n=A.cols="+A.cols)
	val dim:Int = A.cols - A.rows
	val solutionSpace = SolutionSpace(A,b)
	val F:DenseMatrix[Double] = solutionSpace.F
	val z0:DenseVector[Double] = solutionSpace.z0

	def isSatisfiedBy(x:DenseVector[Double]):Boolean = norm(A*x-b) < 1e-14*A.rows
}