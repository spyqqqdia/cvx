package cvx

import breeze.linalg.{DenseMatrix, DenseVector, NotConvergedException, _}
import breeze.numerics._

/**
  * Created by oar on 1/21/17.
  * KKT system for minimization with only equality constraints (such as in
  * the barrier method):
  *
  *     Hx+A'w = -q   and   Ax = b,
  *
  * see docs/KKT.pdf. The standard approach to solution is to first
  * equilibrate the matrix H -> Q=DHD with a diagonal matrix D, then solve the
  * resulting system using a Cholesky factorization Q=LL', see docs/KKT.pdf.
  * This approach is taken if the parameter kktType is set to zero (the default).
  *
  * However we also provide a way to take advantage of special structure in the matrix H.
  * In this case the user does the factorization H=LL' and submits the matrix L instead
  * H and the solution proceeds directly using this factor L. In this case the
  * parameter kktType must be set to 1.
  *
  * @param H the matrix H as described above.
  */
class KKTSystem(
  val H:DenseMatrix[Double], val A:DenseMatrix[Double],
  val q:DenseVector[Double], val b:DenseVector[Double]
) {

  val n=H.cols
  assert(H.rows==n,"Matrix M not square, n=M.cols="+n+", M.rows="+H.rows)
  assert(A.cols==n,"A.cols="+A.cols+" not equal to n=M.rows="+H.rows)
  val p=A.rows   // number of equalities


  /** Solution in case of kktType=0, i.e. M=H.
    *
    * @param debugLevel if set to a value > 2 the condition number of the matrix H
    *              before and after equilibration is printed.
    * @param delta: regularization pos semidef H -> H+delta*||H||*I
    * @return pair of solutions (x,w)
    */
  def solve(
    delta:Double,logger:Logger,tol:Double,debugLevel:Int
  ):(DenseVector[Double],DenseVector[Double]) = {

    // move to non singular system (positive definite K) immediately
    val K = H + A.t * A
    val z = q - A.t * b
    try {

      KKTSystem.solvePD(K,A,z,b,logger,tol,debugLevel)

    } catch {

      case e: Exception => try {

        val M = K + DenseMatrix.eye[Double](K.rows)*delta
        KKTSystem.solvePD(M,A,z,b,logger,tol,debugLevel)

      } catch {

        case e: Exception => KKTSystem.kktSolveSVD(H,A,q,b,logger,tol,debugLevel)
      }
    }
  }
}
object KKTSystem {

  /** System of equations
    *                 Hx+A'w = -q   and   Ax = b,
    * see docs/KKT.pdf.
    *
    * @param H matrix H as above.
    *
    * */
  def apply(
             H: DenseMatrix[Double], A: DenseMatrix[Double],
             q: DenseVector[Double], b: DenseVector[Double]
           ): KKTSystem = new KKTSystem(H,A,q,b)

  /** Solution of
    * Hx+A'w = -q  and  Ax = b
    * where H=LL' is positive definite, L is lower triangular and we have
    * the Cholesky factor L.
    * Block elimination is used, see docs/KKT.pdf. No equilibration of H
    * is performed.
    *
    * We do not check that L is lower triangular with nonzero diagonal
    * elements.
    *
    * @param debugLevel if set to >2 checks for the symmetry error which may result
    *                   from bad openblas.
    * @param tol: tries to get the accuracy of the solution below tol*dim,
    *             throws LinsolveException if that fails.
    *
    * @return pair of solutions (x,w)
    */
  def solveWithCholFactor(
    L: DenseMatrix[Double], A: DenseMatrix[Double],
    q: DenseVector[Double], b: DenseVector[Double],
    logger:Logger, tol:Double, debugLevel:Int
  ): (DenseVector[Double], DenseVector[Double]) = {

    val n = L.cols
    assert(L.rows == n, "Matrix M not square, n=M.cols=" + n + ", M.rows=" + L.rows)
    assert(A.cols == n, "A.cols=" + A.cols + " not equal to n=M.rows=" + L.rows)

    val p = A.rows // number of equalities

    val B = DenseMatrix.zeros[Double](n, p + 1)
    B(::, 0 until p) := A.t
    B(::, p) := q

    // solving HX=LL'X=B as L'X=Y, LY=B
    val Y = MatrixUtils.triangularSolve(L, "L", B)
    val X = MatrixUtils.triangularSolve(L.t, "U", Y)
    val Hinv_At: DenseMatrix[Double] = X(::, 0 until p) // inv(H)A'
    val Hinv_q: DenseVector[Double] = X(::, p) // inv(H)q

    val R = A * Hinv_At // A*inv(H)*A', pxp matrix
    if (debugLevel > 2) {
      val symmErr = Math.sqrt(sum((R - R.t) :* (R - R.t)))
      if (symmErr > 1e-14*R.rows){

        val src = "\nKKTSystem.solveWithCholFactor:\n"
        val problem = "unexpectedly large deviation from symmetry in R = A*inv(H)*A':\n"
        val size = "size per row: ||R-R'||/rows = "+symmErr/R.rows+"\n"
        val fix = "Could be a problem with openblas, recompile from source!\n"
        val msg = src+problem+size+fix
        logger.println(msg)
        println(msg); Console.flush()
      }
    }
    val S = (R + R.t) * 0.5 // make exactly symmetric
    val K = cholesky(S) // S = KK'
    val z = -(b + A * Hinv_q) // z = -(b+inv(H)q)
    // solve KK'w = Sw = z
    val u = MatrixUtils.forwardSolve(K, z) // Ku = z
    val w = MatrixUtils.backSolve(K.t, u) // K'w = u
    val x = -(Hinv_q + Hinv_At * w)

    // check for accuracy, recall Hx=LL'x
    val f = Math.sqrt(L.rows+L.cols)
    val Ltx = L.t*x; val Hx = L*Ltx
    val err1 = norm(Hx+A.t*w+q)/f

    val g = Math.sqrt(A.rows+A.cols)
    val err2 = norm(A*x-b)/g
    if(err1 > tol || err2 > tol){

      if(debugLevel>0){
        val src = "\nKKTSystem.solveWithCholFactor: "
        val problem = "Error in solution exceeds tolerance:\n"+
                      "With f = sqrt(H.rows+H.cols) = "+f+" and g = sqrt(A.rows+A.cols) = "+g+"\n"
        val size = "we have\t\t ||Hx+A.t*w+q||/f = "+err1+", and\t\t||Ax-b||/g = "+err2+".\n"
        val fix = "Throwing exception to hand off to remediation attempts.\n"
        val msg = src+problem+size+fix
        logger.println(msg)
        println(msg); Console.flush()
      }
      throw LinSolveException(null, null, null,"Error in solution exceeds tolerance.")
    }
    (x, w)
  }

  /** Solution of
    * Hx+A'w = -q  and  Ax = b
    * assuming that H is positive definite using block elimination
    * and no equilibration of H.
    *
    * @param debugLevel if > 2, the matrix H is written to log file
    *                   if the KKT system is singular.
    * @return pair of solutions (x,w)
    */
  def blockSolve(
                  H: DenseMatrix[Double], A: DenseMatrix[Double],
                  q: DenseVector[Double], b: DenseVector[Double],
                  logger: Logger, tol:Double, debugLevel: Int
  ): (DenseVector[Double], DenseVector[Double]) = {

    val n = H.cols
    assert(H.rows == n, "Matrix M not square, n=M.cols=" + n + ", M.rows=" + H.rows)
    assert(A.cols == n, "A.cols=" + A.cols + " not equal to n=M.rows=" + H.rows)

    try {
      val L = cholesky(H) // M=H
      solveWithCholFactor(L, A, q, b, logger, tol, debugLevel)

    } catch {

      case e: NotConvergedException => {

        val msg = "\nKKTSystem.blockSolve: problem with Cholesky factor. Will switch to SVD."
        if(debugLevel > 0){ print(msg); Console.flush() }
        if (debugLevel > 2) {

          logger.print("\n\nKKTSystem::blocksolve: singular KKT system, matrix H:\n")
          MatrixUtils.print(H, logger, 3)
          logger.println("\n")
        }
        throw LinSolveException(H, null, null, msg)
      }
    }
  }

  /** Solution of
    * Hx+A'w = -q  and  Ax = b
    * assuming that H is positive definite with equilibration of H.
    *
    * @param debugLevel if > 2, the condition number of the matrix H
    *                   before and after equilibration is printed.
    * @return pair of solutions (x,w)
    */
  private def solvePD(
                       H: DenseMatrix[Double], A: DenseMatrix[Double],
                       q: DenseVector[Double], b: DenseVector[Double],
                       logger: Logger, tol:Double, debugLevel: Int
  ): (DenseVector[Double], DenseVector[Double]) = {

    val n = H.cols
    assert(H.rows == n, "Matrix M not square, n=M.cols=" + n + ", M.rows=" + H.rows)
    assert(A.cols == n, "A.cols=" + A.cols + " not equal to n=M.rows=" + H.rows)

    val eq = MatrixUtils.ruizEquilibrate(H)
    val Q = eq._2
    val d = eq._1 // Q=DMD, D=diag(d)

    if (debugLevel > 2) {
      val eig_H = eig(H)
      val src = "\nKKTSystem.solvePD:\n"
      val msg_EH = "Eigenvalues of H:\n" + eig_H.eigenvalues + "\n"
      val eig_Q = eig(Q)
      val msg_EQ = "Eigenvalues of Q:\n" + eig_Q.eigenvalues + "\n"
      val condH = MatrixUtils.conditionNumber(H)
      val condQ = MatrixUtils.conditionNumber(Q)
      val msg_H = "KKTSystem::solvePD: condition number of H: " + MathUtils.round(condH, 0) + "\n"
      val msg_Q = "KKTSystem::solvePD: condition number of Q=equilibrated(H): " + MathUtils.round(condQ, 0) + "\n\n"
      val msg = src+msg_EH + msg_EQ + msg_H + msg_Q
      print(msg)
      Console.flush()
      logger.print(msg)

      if (debugLevel > 3 && condH>1e10) {

        logger.println("Matrix H:\n"); MatrixUtils.print(H,logger,digits=3)
        println("Matrix H:\n"); MatrixUtils.print(H,digits=3); Console.flush()
      }
    }


    // B = AD = A*diag(d), multiply col_j(A) with d_j
    val B = DenseMatrix.tabulate(A.rows, A.cols)((i, j) => d(j) * A(i, j))
    // D*q
    val Dq = DenseVector.tabulate(n)(i => d(i) * q(i))

    val (y, w) = blockSolve(Q, B, Dq, b, logger, tol, debugLevel)
    // x = Dy = diag(d)y
    val x = DenseVector.tabulate(n)(i => d(i) * y(i))
    (x, w)
  }

  /** Returns the KKT  matrix composed of the Hessian H and left hand side
    * A of the equation constraint Ax=b:
    * | H, A'|
    * | A, 0 |
    */
  def kktMatrix(H: DenseMatrix[Double], A: DenseMatrix[Double]) = {

    assert(H.rows == H.cols, "\nkktMatrix: H is not square, H.rows = " + H.rows + ", H.cols = " + H.cols)
    assert(H.cols == A.cols, "\nkktMatrix: dimension mismatch, H.cols = " + H.cols + ", A.cols = " + A.cols)

    val Z = DenseMatrix.zeros[Double](A.rows, A.rows)
    DenseMatrix.vertcat(DenseMatrix.horzcat(H, A.t), DenseMatrix.horzcat(A, Z))
  }



  /** Solve the KKT system via SVD decomposition of the KKT matrix
    * | H, A'|
    * | A, 0 |
    * This is expensive but can lead to a solution even if the KKT matrix
    * is singular (depending on the right hand side).
    * We will use this only if the solution by block elimination is impossible
    * because the KKT matrix is singular.
    * Here the KKT system has the form
    * Hu+A'nu = -g
    * Au=r
    * This will compute a candidate solution even if there are zero singular values
    * (by sharp cutoff of the singular values below 1e-14). It then checks if the candidate
    * satisfies the KKT system to within the desired tolerance and throws a LinSolveException
    * if it does not.
    *
    * @param tol tolerated size of ||Mw-q|| where Mw=q is the KKT system.
    * @return pair (u,nu). Here the interpretation of u=dx is the Newton step
    *         and nu the lagrange multiplier associated with the equality constraints.
    */
  def kktSolveSVD(
                   H: DenseMatrix[Double], A: DenseMatrix[Double],
                   g: DenseVector[Double], r: DenseVector[Double],
                   logger: Logger, tol: Double, debugLevel:Int
                 ) = {

    assert(H.rows == H.cols, "\nkktMatrix: H is not square, H.rows = " + H.rows + ", H.cols = " + H.cols)
    assert(H.cols == A.cols, "\nkktMatrix: dimension mismatch, H.cols = " + H.cols + ", A.cols = " + A.cols)

    val q = DenseVector.vertcat(-g, r) // the right hand side
    val M = kktMatrix(H, A)
    try {

      val n = H.rows
      val w = MatrixUtils.svdSolve(M, q, logger, tol, debugLevel)
      (w(0 until n), w(n until n + A.rows))

    } catch {

      case e:LinSolveException => if (debugLevel > 2) {

        logger.println(e.message + "\nKKT matrix:\n")
        val digits = 3
        MatrixUtils.print(e.A, logger, digits)
      }
      throw e
    }
  }

}