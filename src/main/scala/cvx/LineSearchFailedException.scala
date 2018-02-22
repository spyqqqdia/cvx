package cvx

import breeze.linalg.{DenseMatrix, DenseVector, _}

/** Backtracking line search did not result in a point satisfying
  * the termination criterion.
  */
case class LineSearchFailedException(message: String = "") extends Exception(message)