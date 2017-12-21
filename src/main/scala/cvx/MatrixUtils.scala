package cvx

import breeze.linalg.eigSym.EigSym
import breeze.linalg.{DenseMatrix, DenseVector, qr, _}
import breeze.numerics.{abs, _}
import org.netlib.util.intW
import com.github.fommil.netlib.LAPACK.{getInstance => lapack}


/**
  * Created by oar on 12/1/16.
  */
object MatrixUtils {

  val rng = scala.util.Random

  /*************************************************************/
  /******************* Matrix generation ***********************/
  /*************************************************************/

  /** A vector in dimension n with uniformly random entries from the
    * intervall [a,b].
    */
  def randomVector(n:Int,a:Double,b:Double):DenseVector[Double] =
    DenseVector.tabulate[Double](n)(j => a+(b-a)*rng.nextDouble())

  /** An nxm matrix with uniformly random entries from the
    * intervall [a,b].
    * Preferably use the version of this function with control over
    * the condition number.
    */
  def randomMatrix(n:Int,m:Int,a:Double,b:Double):DenseMatrix[Double] =
    DenseMatrix.tabulate[Double](n,m)((i,j) => a+(b-a)*rng.nextDouble())

  /** Diagonal nxn matrix (interpretation of diagonal as singular values
    * of some other matrix) with exponentially declining diagonal elements
    * d_j such that d_1=1 and d_n=1/condNum.
    * If dimKernel>0, then the dimKernel smallest (last) diagonal elements
    * are set to zero.
    */
  def diagonalMatrix(n:Int,condNum:Double,dimKernel:Int):DenseMatrix[Double] = {

    val rho = log(condNum)/n
    val d = DenseVector.tabulate[Double](n)(j => exp(-j*rho))
    for(j <- ((n-dimKernel) until n)) d(j) = 0.0
    diag(d)
  }

  /** A random nxn orthogonal matrix based on the QR-decomposition of a
    * STN-random nxn matrix. This is uniform in the Haar measure on O(n).
    */
  def randomOrthogonalMatrix(n:Int):DenseMatrix[Double] = {

    val rng = scala.util.Random
    val A = DenseMatrix.tabulate[Double](n,n)((i,j)=>rng.nextGaussian())
    val qr.QR(q,r) = qr(A)     // complete QR factorization
    q
  }

  /** A random symmetric, positive definite nxn matrix A with given condition number.
    * Constructed as A=UDU', where U is a random orthogonal matrix and D a diagonal
    * matrix with exponentially declining values on the diagonal.
    */
  def randomMatrix(n:Int,condNumber:Double):DenseMatrix[Double] = {

    val U = randomOrthogonalMatrix(n)
    val D = diagonalMatrix(n,condNumber,dimKernel=0)
    (U*D)*U.t
  }


  /** A matrix with m columns containing all combinations of signs +1, -1.
    * Note: this matrix has pow(2,m) rows!!
    */
  def signCombinationMatrix(m:Int):DenseMatrix[Double] = {

    assert(m>=1,"\nm must be at least 1 but is m = "+m+"\n")
    if(m==1) DenseMatrix((1.0,-1.0)).t
    else {

      val sgnM = signCombinationMatrix(m-1)
      val n = sgnM.rows
      val n_ones = DenseVector.ones[Double](n)
      val col1 = DenseVector.vertcat(n_ones,-n_ones).toDenseMatrix.t
      val sgnM2 = DenseMatrix.vertcat(sgnM,sgnM)
      assert(sgnM2.rows==col1.rows)
      DenseMatrix.horzcat(col1,sgnM2)
    }
  }

  /** A matrix with n columns where columns j in [p,q) contain
    * all combinations of signs +1, -1, all other entries being zero.
    *
    * The intended application is the resolution of a constraint
    *   |x_p|+...+|x_{q-1}| <= ub into linear constraints
    *    a_px_p +...+ a_{q-1}x_{q-1} <= ub,
    * where the coefficient vector (a_p,...,a_{q-1}) runs through all
    * such combinations of signs.
    *
    * Note: this matrix has pow(2,q-p) rows!!
    *
    * @param n: n>=1
    * @param p: p>=0
    * @param q: p<=q<=n.
    */
  def signCombinationMatrix(n:Int,p:Int,q:Int):DenseMatrix[Double] = {

    assert(n>=1 && q<=n && p<=q && p>=0)
    val sgnM = signCombinationMatrix(q-p)
    val rows = sgnM.rows
    val res0 = if(p==0) sgnM else {  // tack on the block of zeros on the left

      val preBlock = DenseMatrix.zeros[Double](rows,p)
      DenseMatrix.horzcat(preBlock,sgnM)
    }
    // if q==n no zeros on right
    if(q==n) res0 else {  // tack on the block of zeros on the right

      val postBlock = DenseMatrix.zeros[Double](rows,n-q)
      DenseMatrix.horzcat(res0,postBlock)
    }
  }


  /****************************************************/
  /******************* Printing ***********************/
  /****************************************************/


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

  /** Print the vector with a Logger (to the log file of the Logger).
    */
  def print(vec: DenseVector[Double],digits:Int):Unit = {

    val n = vec.length
    var i=0
    while(i<n){

      val suffix:String = if(i<n-1) ", " else "\n"
      Console.print(MathUtils.round(vec(i),digits).toString+suffix)
      i+=1
    }
  }

  /** Print the matrix with a Logger (to the log file of the Logger).
    */
  def print(matrix: DenseMatrix[Double],digits:Int):Unit = {

    val n = matrix.rows; val m = matrix.cols
    var i=0
    while(i<n){
      var j=0
      while(j<m){
        val suffix:String = if(j<m-1) ", " else "\n"
        Console.print(MathUtils.round(matrix(i,j),digits).toString+suffix)
        j+=1
      }
      i+=1
    }
  }


  /********************************************************/
  /******************* Miscellaneous***********************/
  /********************************************************/

  /** HS-norm sqrt(sum(A_ij*A_ij)).*/
  def normHS(A:DenseMatrix[Double]) = Math.sqrt(sum(A:*A))

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


  /************************************************************/
  /***************** Matrix equilibration *********************/
  /************************************************************/



  /** Equilibrate (a form of preconditioning) the square matrix H with the ruiz algorithm
    * in the l_2-norm until convergence, see docs/ruiz.pdf, p4 algorithm 1.
    * Our algorithm assumes that H is symmetric.
    *
    * @param H a symmetric matrix.
    * @return tuple (d,Q) where Q=DHD is the equilibrated version of H and D is a diagonal matrix
    *         with diagonal d
    */
  def ruizEquilibrate(H:DenseMatrix[Double]):(DenseVector[Double],DenseMatrix[Double]) = {

    val n = H.rows
    assert(n==H.cols,"Matrix H not square: H.rows="+n+", H.cols="+H.cols)

    val d = DenseVector.fill(n)(1.0)     // diagonal of equilibration matrix D
    var rho = 1.0                        // convergence measure
    val itermax = 20
    var iter = 0

    while(iter<itermax && rho>1e-6){

      val Q = (d*d.t):*H
      var i = 0
      rho = 0.0                              // max_i|1-||row_i(Q)||_oo|
      while(i<n){

        //val u = sqrt(max(abs(Q(i,::))))
        val u = sqrt(norm(Q(i,::).t))
        val v = if(u>0) 1.0/u else 1.0
        d(i)*=v
        val r = abs(1-u)
        if(r>rho) rho=r
        i+=1
      }
      iter+=1
    }
    (d,(d*d.t):*H)
  }


  /** Equilibrate (a form of preconditioning) the square matrix H with the ruiz algorithm via
    * one round of l_oo-norm equilibration followed by 5 rounds of l_2-norm equilibration.
    *
    * @param H square matrix without zero row.
    * @return tuple (d,Q) where Q=DHD is the equilibrated version of H and D is a diagonal matrix
    *         with diagonal d
    */
  def ruizEquilibrate0(H:DenseMatrix[Double]):(DenseVector[Double],DenseMatrix[Double]) = {

    val n = H.rows
    assert(n==H.cols,"Matrix H not square: H.rows="+n+", H.cols="+H.cols)

    var Q = H.copy
    var d=DenseVector.zeros[Double](n)     // diagonal of equilibration matrix D

    // one round of l_oo-norm equilibration
    var i=0
    while(i<n){

      var f_i = Math.sqrt(max(abs(Q(i,::).t)))
      if(f_i==0) f_i=1.0
      d(i)=1.0/f_i
      i+=1
    }
    Q = (d*d.t):*H                         // outer(d,d) eltwise-* H

    // five rounds of l_2-norm equilibration
    var k=0
    while(k<5){

      i=0
      while(i<n){ d(i)/=sqrt(norm(Q(i,::).t)); i+=1 }
      Q = (d*d.t):*H
      k+=1
    }
    (d,Q)
  }





  /************************************************************/
  /******************* Equation solving ***********************/
  /************************************************************/


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


  /** Solve Hx=b, where H is positive semi-definite without zero row using Cholesky factorization.
    *
    * First we precondition the matrix H by replacing H --> Q=DHD, where D is a suitable diagonal matrix
    * (algorithm of Ruiz), then using Cholesky factorization on the preconditioned matrix Q.
    * Throws LinSolveException if the Cholesky factorization fails (i.e. if H is not positive definite).
    *
    * @return solution vector x
    */
  def solveWithPreconditioning(
      H:DenseMatrix[Double], b:DenseVector[Double], tol:Double, debugLevel:Int
  ): DenseVector[Double] = {

    val m = b.length
    val n = H.rows
    assert(n==H.cols,"Matrix H not square: H.rows="+n+", H.cols="+H.cols)
    assert(m==n,"length(b) = "+m+" is not equal to H.rows="+H.rows)

    // Ruiz equilibration changes H
    val M = H.copy
    val eqM = ruizEquilibrate(M);
    val d = eqM._1; val Q = eqM._2  // diag(d)*M*diag(d)

    try {

      val L = cholesky(Q)
      // Set D=diag(d), rewrite Hx=b as HDu=b with x=Du, then multiply with D to obtain Qu=DHDu=Db,
      // i.e. LL'u=Db, solve as Lw=Db, w=L'u. Finally get x=Du
      val w = MatrixUtils.forwardSolve(L, d :* b)
      val u = MatrixUtils.backSolve(L.t, w)
      val x = d :* u // the solution

      // throw LinSolveException if the solution is not accurate to tolerance,
      val err = norm(H * x - b)
      val f = Math.sqrt(H.rows + H.cols) // scaling the tolerated error
      if (err > tol * f) {

        val src = "\nMatrixUtils.solveWithPreconditioning: error exceeds tolerance:\n"
        val problem = "With f = sqrt(H.rows+H.cols) = " + f + " we have ||Hx-b||/f = " + err / f + ".\n"
        val msg = src + problem
        if (debugLevel > 1) {
          println(msg); Console.flush()
        }
        throw LinSolveException(H, b, L, msg)
      }
      x

    } catch {

        case e:Exception =>
          val msg = "MatrixUtils.solveWithPreconditioning: Cholesky factorization of H failed.\n"
          throw LinSolveException(H,b,null,msg)
      }
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



  /** Returns a bad right hand side b for a symmetric,
    * positive definite system Ax=b.
    *
    * Let A = UDV' denote the singular value decomposition of A,
    * with D=diag(s_j) the diagonal matrix with the singular values s_j of A
    * and  u_j = col_j(U).
    *
    * b will be constructed with large components
    *           u_j'b = rand_unif(10,100)
    * in directions of all the u_j corresponding to nonzero
    * singular values s_j and zero components in direction of u_j if s_j=0.
    * This guarantees that the system Ax=b has an exact solution but
    * the right hand side b makes the solution challenging if the system is
    * ill conditioned.
    *
    * @param s: vector of singular values of A.
    * @param U: vector of left singular vectors of A.
    */
    def nastyRHS(s:DenseVector[Double],U:DenseMatrix[Double]):DenseVector[Double] = {

      val n = s.length
      assert(U.rows == n && U.cols==n,"\nU.rows = "+U.rows+", U.cols = "+U.cols+" should both = "+n+"\n")
      val  r = scala.util.Random
      val w = DenseVector.tabulate[Double](U.rows)(j => if(s(j)>0) 1+2*r.nextDouble() else 0.0)
      U*w
    }


  /** Computes the vector x0 of minimal norm minimizing the distance ||Mx-q|| and throws an
    * UnsolvableSystemException if this distance exceeds the tolerance tol*f, where the factor
    * f = sqrt(M.rows+M.cols) scales the acceptable error with the matrix size.
    *
    * If the system Mx=q has a solution then x0 will be the solution of minimal norm.
    * Uses the SVD of A. First the norm ||Mx0-q|| is computed
    * (for this we do not need x0, see docs/svdSolve.pdf, eq(3)) and
    * we check if this is less than the tolerance tol.
    * If ||Mx0-q||>=tol an UnsolvableSystemException is thrown.
    * Otherwise we first try the straightforward
    * solution (docs/svdSolve.pdf, eq(2)) using all positive singular values.
    * If this fails to solve the system within tolerance tol, we proceed to regularization
    * along the lines of docs/cvx_notes.pdf, section ???. If this also fails, an
    * UnsolvableSystemException is thrown.
    *
    * @param tol tolerated size of ||Mx0-q|| is tol*f, where f=sqrt(M.rows+M.cols).
    * @return solution x0
    */
  def svdSolve(
    M: DenseMatrix[Double], q: DenseVector[Double],
    logger:Logger, tol:Double, debugLevel:Int
  ) = {

    val n = M.rows; val m = M.cols; val f = Math.sqrt(n+m)
    assert(q.length==n,
      "\nminimumNormSolution: incompatible dimensions in Mx=q, M.rows = "+n+", q.length = "+q.length+"\n"
    )

    val svd.SVD(u, s, v) = svd(M)
    // compute min_x||Mx-q||=||q-proj_q||, where proj_q is
    // the projection of q onto range of M, docs/svdSolve.pdf, eq(3)
    val a = DenseVector.tabulate[Double](n)(j => if (s(j) > 0) u(::, j) dot q  else 0.0)
    val proj_q = u*a
    val minDist = norm(q-proj_q)    // min_x||Mx-q||
    if(minDist>tol*f) {

      val msg = "\nsvdSolve: min_x||Mx-q||/f = "+minDist/f+" > tol = "+tol+
                ", where f = sqrt(A.rows+A.cols) = "+f+".\n"
      throw UnsolvableSystemException(msg)
    }
    // now minDist<=tol and the system is solvable to within tolerance at least theoretically
    // but we may have numeric problems
    // first the straightforward solution in the form docs/svdSolve.pdf, eq(1)
    val z = DenseVector.tabulate[Double](n)(j => if (s(j) > 0) ((u(::, j) dot q) / s(j)) else 0.0)
    val w = v*z // the solution
    var error = norm(M * w - q)
    var optSol = w
    var minError = error
    // if this error is too large we need to regularize the system
    if (error > tol * f) {

      if (debugLevel > 0) {

        val problem = "\nmatrixUtils.svdSolve: error ||Mw-q||/f = " + error / f +
                      " > tol, where f = sqrt(M.rows+M.cols) = "+f+".\n"
        val next = "Trying regularization.\n"
        val msg = problem + next
        println(msg)
        Console.flush()
        logger.println(msg)
      }
      // try a couple of regularizations similar to docs/cvx_notes.pdf ??? with p=1
      val deltas = Vector[Double](1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100, 1000)
      for (delta <- deltas) {

        val z = DenseVector.tabulate[Double](n)(j => {

          val sj=s(j); val sj_num = sj*sj*sj; val sj_denom = sj_num*sj
          if (abs(s(j)) > 0) sj_num*(u(::, j) dot q) / (sj_denom+delta*delta) else 0.0
        })
        val w = v*z // the solution

        error = norm(M * w - q)
        if (error < minError) {
          minError = error
          optSol = w
        }

        // log what's going on
        if (debugLevel > 1) {

          val msg = "delta = " + delta + ",\t\terror ||Mw-q||/f = " + MathUtils.round(error/f,2) +
                    ",\t\twhere f = sqrt(M.rows+M.cols) = "+MathUtils.round(f,2)+"."
          println(msg)
          Console.flush()
          logger.println(msg)
        }
      }
      if (minError > tol * M.rows) {

        val problem = "\nsolveSVD: system not solvable within tolerance tol = "+tol+":\n"
        val size = "error ||Mw-q||/f = " + minError / f + ", where f = sqrt(M.rows+M.cols) = "+f+".\n"
        throw UnsolvableSystemException(problem + size)
      }
    }
    optSol
  }
}