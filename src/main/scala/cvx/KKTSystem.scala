package cvx

import breeze.linalg.{DenseMatrix, DenseVector, NotConvergedException, _}

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
  * @param M if kktType=0, M is interpreted to be the matrix H and solution proceeds
  *          as described above. If kktType=1, M is interpreted to be the factor M=L
  *          in the Cholesky factorization H=LL' and solution uses ths factor L.
  * @param kktType flag to indicate how the matrix M is to be interpreted.
  *
  * The idea is that you will not call the constructor directly but rather
  * use the apply factory functions in the companion object.
  */
class KKTSystem(
     val M:DenseMatrix[Double], val A:DenseMatrix[Double],
     val q:DenseVector[Double], val b:DenseVector[Double],
     val kktType:Int=0
) {

    val n=M.cols
    assert(M.rows==n,"Matrix M not square, n=M.cols="+n+", M.rows="+M.rows)
    assert(A.cols==n,"A.cols="+A.cols+" not equal to n=M.rows="+M.rows)
    val p=A.rows   // number of equalities


    /** Solution in case of kktType=0, i.e. M=H.
      * @return pair of solutions (x,w)
      */
    private def solveType0:(DenseVector[Double],DenseVector[Double]) =  try {

        KKTSystem.solvePD(M,A,q,b)

    } catch {

        case e:LinSolveException => {

            val K = M+A*A.t
            val z = q-A.t*b
            KKTSystem.solvePD(K,A,z,b)
        }
    }
    /** Solution in case of kktType=1, i.e. M=L, where H=LL' is positive definite
      * and L is lower triangular. No equilibration of H performed.
      * @return pair of solutions (x,w)
      */
    private def solveType1:(DenseVector[Double],DenseVector[Double]) =  KKTSystem.solveWithCholFactor(M,A,q,b)

    /** Solution of the system in the following form:
      * @return pair of solutions (x,w)
      */
    def solve:(DenseVector[Double],DenseVector[Double]) = if(kktType==1) solveType1 else solveType0
}
object KKTSystem {

    /** System of equations
      *            Hx+A'w = -q   and   Ax = b,
      *  see docs/KKT.pdf.
      *
      * @param M if kktType=0, M is interpreted to be the matrix H and solution proceeds
      *          by Cholesky factorization M=H=LL', if H is positive definite, or H+AA'=LL'
      *          if M=H is only positive semi definite.
      *          If kktType=1, M is interpreted to be the factor M=L
      *          in the Cholesky factorization H=LL' and solution uses this factor L.
      * @param kktType flag to indicate how the matrix M is to be interpreted.
      *
      **/
    def apply(
        M:DenseMatrix[Double], A:DenseMatrix[Double],
        q:DenseVector[Double], b:DenseVector[Double], kktType:Int=0
    ):KKTSystem = new KKTSystem(M,A,q,b,kktType)

    /** Solution of
      *     Hx+A'w = -q  and  Ax = b
      * where H=LL' is positive definite, L is lower trinagular and we have
      * the Cholesky factor L.
      * Block elimination is used, see docs/KKT.pdf. No equilibration of H
      * is performed.
      *
      * We do not check that L is lower triangular with nonzero diagonal
      * elements.
      *
      * @return pair of solutions (x,w)
      */
    def solveWithCholFactor(
        L:DenseMatrix[Double], A:DenseMatrix[Double],
        q:DenseVector[Double], b:DenseVector[Double]
    ):(DenseVector[Double],DenseVector[Double]) =  {

        val n=L.cols
        assert(L.rows==n,"Matrix M not square, n=M.cols="+n+", M.rows="+L.rows)
        assert(A.cols==n,"A.cols="+A.cols+" not equal to n=M.rows="+L.rows)

        val p=A.rows   // number of equalities

        val B = DenseMatrix.zeros[Double](n,p+1)
        B(::,0 until p) := A.t; B(::,p) := q

        // solving HX=LL'X=B as L'X=Y, LY=B
        val Y = MatrixUtils.triangularSolve(L,"L",B)
        val X = MatrixUtils.triangularSolve(L.t,"U",Y)
        val Hinv_At:DenseMatrix[Double] = X(::,0 until p)     // inv(H)A'
        val Hinv_q:DenseVector[Double] = X(::,p)              // inv(H)q

        val R = A*Hinv_At                                     // A*inv(H)*A', pxp matrix
        val symmErr = Math.sqrt(sum((R-R.t):*(R-R.t)))
        if(symmErr>1e-11) println("\nsolveWithCholFactor: symmErr(R) = "+symmErr+"\n")

        val S = (R+R.t)*0.5                                   // make exactly symmetric
        val K = cholesky(S)                                   // S = KK'
        val z = -(b+A*Hinv_q)                                 // z = -(b+inv(H)q)
        // solve KK'w = Sw = z
        val u = MatrixUtils.forwardSolve(K,z)                 // Ku = z
        val w = MatrixUtils.backSolve(K.t,u)                  // K'w = u
        val x = -(Hinv_q+Hinv_At*w)

        (x,w)
    }

    /** Solution of
      *     Hx+A'w = -q  and  Ax = b
      * assuming that H is positive definite using block elimination
      * and no equilibration of H.
      * @return pair of solutions (x,w)
      */
    def blockSolve(
        H:DenseMatrix[Double], A:DenseMatrix[Double],
        q:DenseVector[Double], b:DenseVector[Double]
    ):(DenseVector[Double],DenseVector[Double]) = {

        val n=H.cols
        assert(H.rows==n,"Matrix M not square, n=M.cols="+n+", M.rows="+H.rows)
        assert(A.cols==n,"A.cols="+A.cols+" not equal to n=M.rows="+H.rows)
        val p=A.rows   // number of equalities

        try{
            val L = cholesky(H)                                   // M=H
            solveWithCholFactor(L,A,q,b)

        } catch {

            case e:NotConvergedException => {

                val msg = "Singular KKT system."
                print(msg)
                throw new LinSolveException(H,null,null,msg)
            }
        }
    }
    /** Solution of
      *     Hx+A'w = -q  and  Ax = b
      * assuming that H is positive definite with equilibration of H.
      * @return pair of solutions (x,w)
      */
    private def solvePD(
        H:DenseMatrix[Double], A:DenseMatrix[Double],
        q:DenseVector[Double], b:DenseVector[Double]
    ):(DenseVector[Double],DenseVector[Double]) = {

        val n=H.cols
        assert(H.rows==n,"Matrix M not square, n=M.cols="+n+", M.rows="+H.rows)
        assert(A.cols==n,"A.cols="+A.cols+" not equal to n=M.rows="+H.rows)

        val eq = MatrixUtils.ruizEquilibrate(H)         // Q=DMD
        val Q = eq._2; val d = eq._1                    // D=diag(d)

        // B = AD = A*diag(d), multiply row_i(A) with d_i
        val B = DenseMatrix.tabulate(A.rows,A.cols)((i,j) => d(i)*A(i,j))
        // D*q
        val Dq = DenseVector.tabulate(n)(i => d(i)*q(i))

        val (y,w) = blockSolve(Q,B,Dq,b)
        // x = Dy = diag(d)y
        val x = DenseVector.tabulate(n)(i => d(i)*y(i))
        (x,w)
    }


}