package cvx

import breeze.linalg.{DenseMatrix, DenseVector}
import cvx.MatrixUtils.{
  ruizEquilibrate,checkSymmetric,choleskySolve,symSolve,svdSolve
}

/**
  * Created by vagrant on 12.02.18.
  *
  * A system of equations Hx=r where the matrix H is symmetric.
  * This is a wrapper for the various solution algos in MatrixUtils.
  * Works also if H is not symmetric but maybe not as well.
  */
class SymmetricLinearSystem(
  val H:DenseMatrix[Double], val r:DenseVector[Double], val logger:Logger
) {

  // Ruiz equilibration does not change H
  private val eqH = ruizEquilibrate(H)
  private val d = eqH._1
  val Q = eqH._2 // diag(d)*H*diag(d)
  private val s = d :* r

  /** Solve the equilibrated system Qu=s, where Q=diag(d)*H*diag(d),
    * s = diag(d)*r and x = diag(d)*u.
    */
  private def solveEquilibrated(tol: Double, debugLevel: Int): DenseVector[Double] =
    if (!checkSymmetric(Q, 1e-13)) svdSolve(Q, s, logger, tol, debugLevel)
    else try {
      choleskySolve(Q, s, logger, tol, debugLevel)
    } catch {

      case e:Exception => symSolve(Q, s, logger, tol, debugLevel)
    }

  /** First we precondition the matrix H by replacing H --> Q=DHD,
    * where D is a suitable diagonal matrix (algorithm of Ruiz).
    * In particular this brings the matrix to unit size.
    * Then we attempt a Cholesky factorization (in case H is positive
    * definite). If this fails we try a regularization Q -> Q+eps*I
    * with eps = 1e-10 in case H is almost positive definite.
    *
    * If this fails also we solve the system via symmetric eigenvalue
    * decomposition.
    *
    * If the matrix Q fails the check for symmetry we solve the system
    * via SVD. This will still work but the regularization assumed that
    * H was symmetric and thus may not have worked very well.
    */
  def solve(tol:Double, debugLevel:Int): DenseVector[Double] = {

    val u = solveEquilibrated(tol,debugLevel)
    d:*u
  }
}

object SymmetricLinearSystem {

  def apply(
    H:DenseMatrix[Double], r:DenseVector[Double], logger:Logger
  ): SymmetricLinearSystem =
    new SymmetricLinearSystem(H,r,logger)

}
