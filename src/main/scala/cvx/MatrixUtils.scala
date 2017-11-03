package cvx

import breeze.linalg.{DenseMatrix, DenseVector,qr, _}
import breeze.numerics.{abs}
import org.netlib.util.intW
import com.github.fommil.netlib.LAPACK.{getInstance => lapack}


/**
  * Created by oar on 12/1/16.
  */
object MatrixUtils {

  /** HS-norm sqrt(sum(A_ij*A_ij)).*/
  def normHS(A:DenseMatrix[Double]):Double = Math.sqrt(sum(A:*A))

  /** @return norm(Q-Q-t) < tol.*/
  def checkSymmetric(Q:DenseMatrix[Double],tol:Double):Boolean = {

    val diff = Q-Q.t
    Math.sqrt(sum(diff:*diff)) < tol
  }

  /** The condition number: quotient of largest singular value divided by smallest one
    * computed using the SVD factorization of H.
    * Needless to say the accuracy of this computation is also negatively influenced by
    * ill conditioning.
    */
  def conditionNumber(H:DenseMatrix[Double]):Double = {

    val svdH = svd(H)
    val sigma = svdH.singularValues
    max(sigma)/min(sigma)
  }

  /** Solves the equation Ax=b where A is a lower, upper or diagonal matrix
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
    * @param A  triangular matrix
    * @param Ltype  "L", "U", "D" (lower, upper triangular, diagonal)
    * @param B  right hand side of equation
    * @return
    */
  def triangularSolve(A:DenseMatrix[Double],Ltype:String, B:DenseMatrix[Double]): DenseMatrix[Double] = {

    assert(Ltype=="L" || Ltype=="U","Ltype must be L or U but is "+Ltype)
    // copy the triangular part:
    val Q = if(Ltype=="L") lowerTriangular(A) else upperTriangular(A)
    val Y = B.copy                    // result will be written to Y

    val n = Q.rows
    val info = new intW(0)

    // solve Lc*X=Y  with result X written to Y
    if(Ltype=="L" || Ltype=="U")
      lapack.dtrtrs(Ltype,"N","N",n,B.cols,Q.data,n,Y.data,n,info)
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
      d(i)=1.0/f_i
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

  /** Solve Hx=b, where H is positive semi-definite without zero row using Cholesky factorization.
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
    * WARNING: we have to deal with the case that the matrix H has zero rows.
    * This can happen for example in a KKT system for phase one analysis if the constraints
    * do not depend on some variables. The right hand side of the KKT system is then also zero
    * so that a solution exists.
    * We cannot deal with this problem in a naive fashion, such as replacing H with H+delta*I,
    * since then the solution will not be a descent direction for the minimization problem.
    *
    * @return solution vector x
    */
  def solveWithPreconditioning(H:DenseMatrix[Double], b:DenseVector[Double], delta:Double): DenseVector[Double] = {

    val m = b.length
    val n = H.rows
    assert(n==H.cols,"Matrix H not square: H.rows="+n+", H.cols="+H.cols)
    assert(m==n,"length(b) = "+m+" is not equal to H.rows="+H.rows)

    //-- FIX ME: If H has zero rows we make the diagonal entry of each zero row equal to one
    // this might still lead to a singular matrix and is a lazy hack
    val M = H.copy
    var i=0
    while(i<n){ if(max(abs(M(i,::).t))<1e-14) M(i,i)=1.0; i=i+1; }

    val eqM = ruizEquilibrate(M); val d = eqM._1; val Q = eqM._2  // diag(d)*M*diag(d)

    //trying the Cholesky factorization of Q=LL'
    var L: DenseMatrix[Double] = null
    try {

      L = cholesky(Q)
      // check if |L_ii|>=eps, if not throw exception to be caught below
      (0 until L.rows).map(i =>
        if(Math.abs(L(i,i))<Math.sqrt(delta)) throw LinSolveException(H,b,L)
      )

    } catch {

      // try Cholesky factorization of H+delta*I
      case e: Exception => try {

        val I = DenseMatrix.eye[Double](Q.rows)
        L = cholesky(Q + I * delta)

      } catch {

        case e: NotConvergedException => throw LinSolveException(H,b,L,
          "Matrix H not positive semidefinite, equilibrated matrix Q=Q(H):\n"+Q
        )
      }
    }
    // Set D=diag(d), rewrite Hx=b as HDu=b with x=Du, then multiply with D to obtain Qu=DHDu=Db,
    // i.e. LL'u=Db, solve as Lw=Db, w=L'u. Finally get x=Du
    val w = MatrixUtils.forwardSolve(L,d:*b)
    val u = MatrixUtils.backSolve(L.t,w)
    d:*u
  }

  /** Solves underdetermined systen Ax=b where A is an mxn matrix with m < n and full rank m,
    * using the QR factorization of the adjoint A', see docs/nullspace.pdf
    *
    * The condition rank(A)=m is not checked and implies that dim(ker(A))=n-m.
    * The solutions x are parametrized as x=z0+Fu, where F is an nx(n-m) matrix with
    * orthonormal columns forming a basis of ker(A) (in particular then Im(F)=ker(A))
    * and z0 is the minimum norm solution of Ax=b.
    *
    * Note that the matrix F then satisfies AF=0 (i.e. $Im(F)\subseteq ker(A)$), and conversely
    * this condition combined with the fact that rank(F)=n-m=dim(ker(A)) implies that Im(F)=ker(A).
    * If then x0 is any solution of Ax=b it follows that x=x0+Fu yields all solutions of this system.
    *
    * Intended application:
    * getting rid of equality constraints Ax=b by change of variables x --> u via x = z0+Fu.
    * This is why we assume that A has full rank as this is expected in all applications.
    *
    * @return ordered pair (z0,F)
    */
  def solveUnderdetermined(A:DenseMatrix[Double],b:DenseVector[Double]): (DenseVector[Double],DenseMatrix[Double])= {

    val qrA=qr(A.t); val Q=qrA.q; val R=qrA.r  // A'=QR

    val m = A.rows; val n = A.cols
    val F = Q(::,m until n)
    // special solution: rewrite Ax=b as R'Q'x=b, set y=Q'x, solve R'y=b for y, then set x=Qy.
    val y = forwardSolve(R.t,b)
    val z0 = Q(::,0 until m)*y

    // check that F has orthonormal columns (F.t*F=I)
    val I = DenseMatrix.eye[Double](n)

    (z0,F)
  }

  /** Print the vector with a Logger (to the log file of the Logger).
    */
  def print(vec: DenseVector[Double],logger:Logger,digits:Int):Unit = {

    val n = vec.length
    var i=0
    while(i<n){

        val suffix:String = if(i<n-1) ", " else "\n"
        logger.print(MathUtils.round(vec(i),digits).toString+suffix)
        i+=1
    }
  }

  /** Print the matrix with a Logger (to the log file of the Logger).
    */
  def print(matrix: DenseMatrix[Double],logger:Logger,digits:Int):Unit = {

    val n = matrix.rows; val m = matrix.cols
    var i=0
    while(i<n){
      var j=0
      while(j<m){
        val suffix:String = if(j<m-1) ", " else "\n"
        logger.print(MathUtils.round(matrix(i,j),digits).toString+suffix)
        j+=1
      }
      i+=1
    }
  }


  /** Solve the system Mu=q via SVD decomposition of the matrix M.
    * This is expensive but can lead to a solution even if the matrix M
    * is singular (depending on the right hand side).
    *
    * In fact we will compute a candidate solution even if there are zero singular values
    * (by sharp cutoff of the singular values below 1e-14). We then checks if the candidate
    * satisfies the system to within the desired tolerance. If not a LinSolveException is
    * thrown.
    *
    * @param tol tolerated size of ||Mw-q|| where Mw=q is the KKT system.
    * @return pair (u,nu). Here the interpretation of u=dx is the Newton step
    *         and nu the lagrange multiplier associated with the equality constraints.
    */
  def svdSolve(
    M: DenseMatrix[Double], q: DenseVector[Double], tol:Double
  ):DenseVector[Double] = {

    val svd.SVD(u,s,v) = svd(M)
    // the solution in the form docs/svdSolve.pdf, eq(1)
    val n = M.rows
    var w = DenseVector.zeros[Double](n)
    var j=0
    while(j<n){

      if(abs(s(j))>1e-14) w += v(::,j)*(u(::,j) dot q)/s(j)
      j+=1
    }
    val error = norm(M*w-q)
    if(error>tol) throw LinSolveException(M,q,null,"\nUnsolvable KKT system, error (in norm): "+error)
    w
  }
}