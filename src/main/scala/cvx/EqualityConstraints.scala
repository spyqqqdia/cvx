package cvx

import breeze.linalg.{DenseMatrix, DenseVector, _}


/** Affine equality constraints of the form Ax=b, where A is mxn with m < n
 *  and full rank m. The condition rank(A)=m will not be checked.
 
 *  Parameterizes solutions as x = z0+Fu, where x=z0 is the minimum norm solution
 *  and $Fu\perp z0$, for all $u\in R^{n-m}$.
 *  Used for change of variables x --> u to reduce dimension and get rid of explicit 
 *  equality constraints.
 */
class EqualityConstraints(val A:DenseMatrix[Double], val b:DenseVector[Double]){

    assert(A.rows < A.cols, "m=A.rows="+A.rows+" is not less than n=A.cols="+A.cols)
	
	private val solutions = MatrixUtils.solveUnderdetermined(A,b)
	val z0:DenseVector[Double] = solutions._1
	val F:DenseMatrix[Double] = solutions._2
	/** Dimension n-m = A.cols - A.rows of solution space.*/
	val dim = A.cols - A.rows
}