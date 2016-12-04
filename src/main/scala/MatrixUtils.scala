package cvx

import breeze.linalg.{DenseMatrix, DenseVector,qr, _}
import breeze.numerics.{abs, _}
import org.netlib.util.intW
import com.github.fommil.netlib.LAPACK.{getInstance => lapack}


/**
  * Created by oar on 12/1/16.
  */
object MatrixUtils {

    /** Solves the equation Lx=b where L is a lower, upper or diagonal matrix
      * by calling lapack.DTRTRS.
      *
      * Triangular solve in lapack is the routine DTRTRS, error bounds are
      * computed with DTRRFS. It has the following structure
      *
      * Netlib-java triangular solve:
      * Solves AX=B or A'X=B where A is triangular or diagonal.
      * public void dtrtrs(
      *     java.lang.String,        // UPLO
      *     java.lang.String,        // TRANS
      *     java.lang.String,        // DIAG
      *     int,                     // n,   (breeze: A.rows=A.cols)
      *     int,                     // nrhs (number of right hand sides, breeze: B.cols
      *     double[],                // A    (left hand side coefficients, breeze: A.data
      *     int,                     // LDA  (leading dimension of A, breeze: max(1,n))
      *     double[],                // B    (right hand side, breeze: B.data)
      *     int,                     // LDB  (leading dimension of B, breeze: max(1,n))
      *     org.netlib.util.intW     // number coded info, breeze: new intW(0)
      * );
      *
      * @param L  lower triangular matrix
      * @param Ltype  "L", "U", "D" (lower, upper triangular, diagonal)
      * @param B  right hand side of equation
      * @return
      */
    def triangularSolve(L:DenseMatrix[Double],Ltype:String, B:DenseMatrix[Double]): DenseMatrix[Double] = {

        val Lc = lowerTriangular(L)       // copy the lower triangular part
        val Y = B.copy                    // result will be written to Y

        val n = Lc.rows
        val info = new intW(0)

        // solve Lc*X=Y  with result X written to Y
        if(Ltype=="L" || Ltype=="U")
            lapack.dtrtrs(Ltype,"N","N",n,B.cols,Lc.data,n,Y.data,n,info)
        else
            lapack.dtrtrs("L","N","U",n,B.cols,Lc.data,n,Y.data,n,info)
        Y
    }
    /** Solves Lx=b. where L is viewed as a lower triangular matrix. Only the part
      * on and below the diagonal is used.
      *
      * This version (with only one right hand side) should not be so bad and in any case
      * less critical and no lapack version is implemented.
      */
    def forwardSolve(L:DenseMatrix[Double], b:DenseVector[Double]): DenseVector[Double] = {

        val m = b.length
        val n = L.rows
        assert(m==n,"forwardSolve: length(b) = "+m+" is not equal to rows(L) = "+n)
        assert( // abs(L_ii)>0, for all i
            (0 until n).forall(i => Math.abs(L(i,i))>0),
            "Singular lower trianguar matrix L: zero on the diagonal"
        )
        val x = DenseVector.zeros[Double](n)
        var i=0
        while(i<n){

            var sum = 0.0
            var j=0
            while(j<i){ sum += L(i,j)*x(j); j+=1 }
            x(i) = (b(i)-sum)/L(i,i)
            i+=1
        }
        x
    }
    /** Solves Ux=b. where U is viewed as an upper triangular matrix. Only the part
      * on and above the diagonal is used.
      *
      * This version (with only one right hand side) should not be so bad and in any case
      * less critical and no lapack version is implemented.
      */
    def backSolve(U:DenseMatrix[Double], b:DenseVector[Double]): DenseVector[Double] = {

        val m = b.length
        val n = U.rows
        assert(m==n,"backSolve: length(b) = "+m+" is not equal to rows(U) = "+n)
        assert( // abs(L_ii)>0, for all i
            (0 until n).forall(i => Math.abs(U(i,i))>0),
            "Singular upper trianguar matrix U: zero on the diagonal"
        )
        val x = DenseVector.zeros[Double](n)
        var i=n-1
        while(i>=0){

            var sum = 0.0
            var j=n-1
            while(j>i){ sum += U(i,j)*x(j); j-=1 }
            x(i) = (b(i)-sum)/U(i,i)
            i-=1
        }
        x
    }

    /** Equilibrate (a form of preconditioning) the square matrix H with the ruiz algorithm via
      * one round of l_oo-norm equilibration followed by 5 rounds of l_2-norm equilibration.
      *
      * @param H square matrix without zero row.
      * @return tuple (d,Q) where Q=DHD is the equilibrated version of H and D is a diagonal matrix
      *         with diagonal d
      */
    def ruizEquilibrate(H:DenseMatrix[Double]):(DenseVector[Double],DenseMatrix[Double]) = {

        val n = H.rows
        assert(n==H.cols,"Matrix H not square: H.rows="+n+", H.cols="+H.cols)

        var Q = H.copy
        var d=DenseVector.zeros[Double](n)     // diagonal of equilibration matrix D

        // one round of l_oo-norm equilibration
        var i=0
        while(i<n){

            val f_i = Math.sqrt(max(abs(Q(i,::).t)))
            if(f_i==0) throw new IllegalArgumentException("row_"+i+"(H) is zero.")
            d(i)=1.0/f_i;
            i+=1
        }
        Q = (d*d.t):*H                         // outer(d,d) eltwise-* Q

        // five rounds of l_2-norm equilibration
        var k=0
        while(k<5){

            i=0
            while(i<n){ d(i)/=norm(Q(i,::).t); i+=1 }
            Q = (d*d.t):*H
            k+=1
        }
        (d,Q)
    }

    /** Solve Hx=b, where H is positive semi-definite without zero row.
      *
      * First we precondition the matrix H by replacing H --> Q=DHD, where D is a suitable diagonal matrix
      * (algorithm of Ruiz), then using Cholesky factorization on the preconditioned matrix Q.
      *
      * If Q is found to be singular, it is regularized via Q --> Q + delta*I.
      * If the Cholesky factor L has a diagonal element with |L_ii|<sqrt(delta), the same
      * regularization will be carried out.
      *
      * For a rationale see docs/cvx_notes.pdf, section Regularization.
      *
      * @return solution vector x
      */
    def solveWithPreconditioning(H:DenseMatrix[Double], b:DenseVector[Double], delta:Double): DenseVector[Double] = {

        val m = b.length
        val n = H.rows
        assert(n==H.cols,"Matrix H not square: H.rows="+n+", H.cols="+H.cols)
        assert(m==n,"length(b) = "+m+" is not equal to H.rows="+H.rows)

        val eqH = ruizEquilibrate(H); val d = eqH._1; val Q = eqH._2  // diag(d)*H*diag(d)

        //trying the Cholesky factorization of Q=LL'
        val breakDown = NotConvergedException.Breakdown
        var L: DenseMatrix[Double] = null
        try {

            L = cholesky(Q)
            // check if |L_ii|>=eps, if not throw exception to be caught below
            (0 until L.rows).map(i =>
                if(Math.abs(L(i,i))<Math.sqrt(delta)) throw new NotConvergedException(breakDown)
            )

        } catch {

            // try Cholesky factorization of H+delta*I
            case e: NotConvergedException => try {

                val I = DenseMatrix.eye[Double](Q.rows)
                L = cholesky(Q + I * delta)

            } catch {

                case e: NotConvergedException => throw new NotConvergedException(
                    breakDown, "Matrix H not positive semidefinite, equilibrated matrix Q=Q(H):\n"+Q
                )
            }
        }
        // Set D=diag(d), rewrite Hx=b as HDu=b with x=Du, then multiply with D to obtain Qu=DHDu=Db,
        // i.e. LL'u=Db, solve as Lw=Db, w=L'u. Finally get x=Du
        val w = MatrixUtils.forwardSolve(L,d:*b)
        val u = MatrixUtils.backSolve(L.t,w)
        d:*u
    }
}
