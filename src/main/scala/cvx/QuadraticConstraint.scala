package cvx

import breeze.linalg.{DenseMatrix, DenseVector}
import cvx.MatrixUtils._

/** Quadratic constraint r + a'x + (1/2)*x'Px <= ub, where P is a symmetric matrix.
  */
class QuadraticConstraint(
  override val id : String,
  override val dim: Int,
  override val ub : Double,
  val r           : Double,
  val a           : DenseVector[Double],
  val P           : DenseMatrix[Double]
) extends Constraint(id, dim, ub) {

  if (a.length != dim) {
    val msg = "Vector a must be of dimension " + dim + " but length(a) " + a.length
    throw new IllegalArgumentException(msg)
  }
  if (!(P.rows == dim & P.cols == dim)) {

    val msg = "Matrix P must be square of dimension " + dim + " but is " + P.rows + "x" + P.cols
    throw new IllegalArgumentException(msg)
  }
  checkSymmetric(P, 1e-13)

  def isDefinedAt(x:DenseVector[Double]):Boolean = true

  def valueAt(x: DenseVector[Double]):Double = {
    checkDim(x); r + (a dot x) + (x dot (P * x)) / 2
  }

  def gradientAt(x: DenseVector[Double]):DenseVector[Double] = {
    checkDim(x); a + P * x
  }

  def hessianAt(x: DenseVector[Double]):DenseMatrix[Double] = {
    checkDim(x); P
  }

  /** This constraint restricted to values of the original variable x of the form x=z+Fu
    * now viewed as a constraint on the variable u in dimension dim-p, where p is the rank
    * of F.
    * F is assumed to be of full rank and this condition is not checked.
    * The intended application is the case where the x=z+Fu are the solutions of
    * equality constraints Ax=b.
    *
    * The result is another quadratic constraint, see docs/cvx_notes.pdf, p4, equation (5).
    *
    * @param z a vector of dimension dim-p (intended: special solution of Ax=b)
    * @param F a nxp matrix (intended: p = number of equality constraints)
    */
  override def affineTransformed(z: DenseVector[Double], F: DenseMatrix[Double]):QuadraticConstraint = {

    val rID = id + "_reduced"
    val rDim = dim - F.cols
    val s = valueAt(z)
    val b = F.t * (a + P * z)
    val Q = (F.t * P) * F
    QuadraticConstraint(rID, rDim, ub, s, b, Q)
  }
}

object QuadraticConstraint {

  /** Constraint r + (a dot x) + x'Qx <= ub. */
  def apply(
    id: String, dim: Int, ub: Double, r: Double, a: DenseVector[Double], Q: DenseMatrix[Double]
  ):QuadraticConstraint =
    new QuadraticConstraint(id, dim, ub, r, a, Q)
}
