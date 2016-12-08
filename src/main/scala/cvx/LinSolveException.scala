package cvx

import breeze.linalg.{DenseMatrix, DenseVector, _}

/** Exceptional condition encountered when solvin a linear system Ax=b.
 * 
 *@param A left hand side of Ax=b
 *@param b right hand side of Ax=b
 *@param L Cholesky factor of A (if used and available)
 */
case class LinSolveException(
	A: DenseMatrix[Double],        
	b: DenseVector[Double],        
	L: DenseMatrix[Double],
	message: String = ""
) 
extends Exception(message)