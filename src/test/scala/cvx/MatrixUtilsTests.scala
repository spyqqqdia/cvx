package cvx

import breeze.linalg.{DenseMatrix, DenseVector, _}


/**
  * Created by oar on 12/3/16.
  */
object MatrixUtilsTests {

  /** Testing if Ruiz equlibration performs as expected if the matrix
    * has zero rows
    */
  def testRuizEquilibration:Unit = {

    val A = DenseMatrix((0.5,0.0,0.5),(0.0,0.0,0.0),(-0.5,0.0,-0.5))
    val eqA = MatrixUtils.ruizEquilibrate(A)
    print("\nA:\n")
    MatrixUtils.print(A,digits=1)
    val d = eqA._1; val Q = eqA._2
    print("\nd: "+d)
    print("\nQ:\n")
    MatrixUtils.print(Q,digits=8)
  }




  /** Testing the solution of a system HX=B using lapack through MatrixUtils.triangularSolve.
    * Here we do H of the form H=LL' with given lower triangular L
    *
    * @return true if both forward and backward error are less than tol.
    */
  def testTriangularSolve(L:DenseMatrix[Double], X:DenseMatrix[Double], tol:Double):Boolean = {

    val K = lowerTriangular(L)     // make sure its lower triangular
    val H = K*K.t
    val B = H*X
    val condH = MatrixUtils.conditionNumber(H)
    println("Condition number of H: "+condH)

    // solving HX=LL'X=B as L'X=Y, LY=B
    val Y = MatrixUtils.triangularSolve(L,"L",B)
    val X1 = MatrixUtils.triangularSolve(L.t,"U",Y)

    val B1 = H*X1
    val diffB = B-B1
    val forwardErrorAbs = MatrixUtils.normHS(diffB)
    val forwardErrorRel = forwardErrorAbs/MatrixUtils.normHS(B)
    val diffX = X-X1
    val backwardErrorAbs = MatrixUtils.normHS(diffX)
    val backwardErrorRel = backwardErrorAbs/MatrixUtils.normHS(X)
    println(
      "Forward error (absolute): "+MathUtils.round(forwardErrorAbs,10)+
        ",\t\tforward error (relative): "+MathUtils.round(forwardErrorRel,10)
    )
    println(
      "Backward error (absolute): "+MathUtils.round(backwardErrorAbs,10)+
        ",\t\tbackward error (relative): "+MathUtils.round(backwardErrorRel,10)
    )
    forwardErrorRel<tol && backwardErrorRel<tol
  }

  /** Testing the solution of a system HX=B using lapack through MatrixUtils.triangularSolve.
    * Here we do H of the form H=LL' with random nxn lower triangular L and random nxp matrix X.
    *
    * @return true if both forward and backward error are less than tol.
    */
  def testTriangularSolve(n:Int,p:Int,tol:Double):Boolean = {

    // uniform in (-1,1)
    val Q:DenseMatrix[Double] = DenseMatrix.tabulate(n,n)((i,j) => -5+10*Math.random())
    val L = lowerTriangular(Q)
    (0 until n).map(i => L(i,i) = L(i,i)+20.0)    // improve conditioning

    val X:DenseMatrix[Double] = DenseMatrix.rand(n,p)
    testTriangularSolve(L,X,tol)
  }


  /** Testing the solution of m systems HX=B using lapack through MatrixUtils.triangularSolve.
    * Here we do H of the form H=LL' with random nxn lower triangular L and random nxp matrix X.
    *
    * @return true if all forward and backward errors are less than tol.
    */
  def testTriangularSolve(n:Int,p:Int,reps:Int,tol:Double):Boolean = {

    println("\n#---Testing triangularSolve on systems HX=B:")
    val results = (0 until reps).map(i => testTriangularSolve(n,p,tol))
    results.forall(p => p)
  }



  /** Testing MatrixUtils::backSolve on a system Ux=b with U being the upper triangular part of W+delta*I,
    * where W is a dim x dim matrix with random entries in (0,1).
    *
    * The right hand side b is chosen as b = Ux0 so we know the solution is x0.
    * The vector x0 will also be chosen randomly with entries in (0,1), so its norm will not be too small.
    * The parameter delta acts as a regularizer to keep the condition number down.
    *
    * Check out what happens if this is run with delta=0. The condition numbers _explode_ and the
    * solutions cannot be trusted at all.
    */
  def testBackSolve(dim:Int,delta:Double,testID:String):Unit = {

    val x0 = DenseVector.rand[Double](dim)
    val U = upperTriangular(DenseMatrix.rand[Double](dim,dim))+DenseMatrix.eye[Double](dim)*delta
    val b = U*x0

    // let's get the condition number of this:
    val svd.SVD(u,s,v) = svd(U)
    val condNum = max(s)/min(s)

    val x = MatrixUtils.backSolve(U,b)
    val errSol = norm(x0-x)/norm(x0)
    val errVal = norm(b-U*x)/norm(b)
    var msg = "\n"+testID+": error in solution = "+errSol+",  error in value Ux = "+errVal
    msg += "\nCondition number: "+MathUtils.round(condNum,2)
    print(msg)
  }

  /** Run reps tests testSolve(dim,"Test_j") */
  def testBackSolve(dim:Int,delta:Double,reps:Int):Unit = for(j <- 1 to reps) testBackSolve(dim,delta,"Test_"+j)


  /** Testing MatrixUtils::forwardSolve on a system Lx=b with L being the lower triangular part of W+delta*I,
    * where W is a dim x dim matrix with random entries in (0,1).
    *
    * The right hand side b is chosen as b = Lx0 so we know the solution is x0.
    * The vector x0 will also be chosen randomly with entries in (0,1), so its norm will not be too small.
    * The parameter delta acts as a regularizer to keep the condition number down.
    *
    * Check out what happens if this is run with delta=0. The condition numbers _explode_ and the solutions
    * cannot be trusted at all.
    */
  def testForwardSolve(dim:Int,delta:Double,testID:String):Unit = {

    val x0 = DenseVector.rand[Double](dim)
    val L = lowerTriangular(DenseMatrix.rand[Double](dim,dim))+DenseMatrix.eye[Double](dim)*delta
    val b = L*x0

    // let's get the condition number of this:
    val svd.SVD(u,s,v) = svd(L)
    val condNum = max(s)/min(s)

    val x = MatrixUtils.forwardSolve(L,b)
    val errSol = norm(x0-x)/norm(x0)
    val errVal = norm(b-L*x)/norm(b)
    var msg = "\n"+testID+": error in solution = "+errSol+",  error in value Ux = "+errVal
    msg += "\nCondition number: "+MathUtils.round(condNum,2)
    print(msg)
  }

  /** Run reps tests testSolve(dim,"Test_j") */
  def testForwardSolve(dim:Int,delta:Double,reps:Int):Unit = for(j <- 1 to reps) testForwardSolve(dim,delta,"Test_"+j)


  /** Testing MatrixUtils::solveWithPreconditioning on a matrix H = A'A with random nxn matrix A
    * and right hand side b = Hx so we know the solution is x. The vector x will also be chosen
    * randomly. All random entries in (0,1).
    */
  def testSolveWithPreconditioning(H:DenseMatrix[Double], x:DenseVector[Double], testID:String):Unit = {

    val dim = H.rows
    val b = H*x

    val delta = 1e-15
    val debugLevel = 3
    val tolEqSolve = 1e-8
    val u = MatrixUtils.solveWithPreconditioning(H,b,tolEqSolve,debugLevel)

    val errSol = norm(u-x)/norm(x)
    val errVal = norm(b-H*u)/norm(b)
    val msg = "\n"+testID+": error in solution = "+errSol+",  error in value Hx = "+errVal
    print(msg)
  }


  /** Testing MatrixUtils::solveWithPreconditioning on a matrix H = A'A with random nxn matrix A
    * and right hand side b = Hx so we know the solution is x. The vector x will also be chosen
    * randomly. All random entries in (0,1).
    */
  def testSolveWithPreconditioning(dim:Int, testID:String):Unit = {

    val x = DenseVector.rand[Double](dim)
    val A = DenseMatrix.rand[Double](dim,dim)
    val S = A.t*A
    val H = S+S.t         // make symmetric (numerical issues)
    testSolveWithPreconditioning(H,x,testID)
  }

  /** Run reps tests testSolve(dim,"Test_j") */
  def testSolveWithPreconditioning(dim:Int, reps:Int):Unit =
    for(j <- 1 to reps) testSolveWithPreconditioning(dim,"Test_"+j)



  /** Test solution of underdetermind system Ax=b where A is mxn with m < n full rank m.
    * Solution is x=x0+Fu, all u, where F has n-m orthonormal columns.
    * This it suffices to check AF=0 and Ax0=b, see MatrixUtils::solveUnderdetermined.
    */
  def testSolveUnderdetermined(A:DenseMatrix[Double], b:DenseVector[Double]):Unit = {

    assert(A.rows == b.length,"Dimension mismatch in Ax=b: A.rows="+A.rows+", b.length="+b.length)
    val sol = MatrixUtils.solveUnderdetermined(A,b)
    val z0 = sol._1
    val F = sol._2
    val AF = A*F
    val errF = Math.sqrt(sum(AF:*AF))
    val errY = norm(A*z0-b)

    print("||AF||="+errF+",  ||Ax0-b||="+errY+"\n")
  }

  /** Runs the preceeding test on reps systems Ax=b with m=n/2 and A,b having random entries in (0,1)
    * and 1.0 added to the diagonal of A to keep the condition number reasonable.
    */
  def testSolveUnderdetermined(n:Int,reps:Int):Unit = {

    print("\n\n#---Testing solution of random underdetermined systems Ax=b:\n")
    for(rep <- 0 until reps){

      val m=n/2
      val b = DenseVector.rand[Double](m)
      val A = DenseMatrix.rand[Double](m,n)
      for(i <- 0 until m) A(i,i)+=1.0

      testSolveUnderdetermined(A,b)
    }
  }


  /** We construct randomly some ill conditioned systems symmetric,
    * positive definite systems Ax=b and solve them via SVD.
    *
    * Let A = UDV' denote the singular value decomposition of A,
    * with D=diag(s_j) the diagonal matrix with the singular values s_j of A.
    * The singular values of A are declining exponentially from 1.0 to 1/condNum.
    * If dimKernel>0, the last (smallest) dimKernel singular values are set to zero.
    *
    * Let u_j = col_j(U). Then a right hand side b with large components
    *           u_j'b = rand_unif(10,100)
    * in directions of the u_j corresponding to small (but nonzero)
    * singular values s_j is constructed which has zero components in
    * direction u_j if s_j=0.
    * This guarantees that the system Ax=b has an exact solution but is
    * very ill conditioned with right hand side for which this ill conditioning
    * is a problem.
    *
    * @param condNum: condition number of A (if dimKernel=0).
    * @param dimKernel: dim(ker(A)).
    * @param debugLevel: no debug output when set to zero,
    *                  maximum output as soon as > 2.
    * @param tolEqSolve: solution of Ax=b will be classified as failed
    *                  if ||Ax-b|| > tolEqSolve*A.cols.
    * In that case we go into the regularization loop to try to get
    * within accepted tolerance.
    */
  def testSvdSolve(
    nTests:Int, dim:Int, condNum:Double, dimKernel:Int, tolEqSolve:Double, debugLevel:Int
  ):Unit = {

    val  startMsg = "\n\n\nDoing MatrixUtilsTests.testSvdSolve in dimension "+dim+":\n"
    println(startMsg); Console.flush()
    val logger = Logger("logs/testSVDSolve.txt")
    logger.println(startMsg)
    var OK = true
    for(nT <- 1 to nTests){

      logger.println("\n\nTest "+nT+":\n")
      val U = MatrixUtils.randomOrthogonalMatrix(dim)
      val D = MatrixUtils.diagonalMatrix(dim,condNum,dimKernel)
      val d = diag(D)
      val A = (U.t*D)*U          // SVD of symmetric, positive definite matrix A
      val b = MatrixUtils.nastyRHS(d,U)

      try{
        val  x = MatrixUtils.svdSolve(A,b,logger,tolEqSolve,debugLevel)
      } catch {

        case e:Exception =>
          logger.println("\n"+e.getMessage+"\n")
          OK = false
      }
    }
    if(OK) println("Finished, all tests passed within tolerance.")
    else println("Finished, some tests failed, see logs/logs/testSVDSolve.txt")
    Console.flush()
  }


  /** We print several sign combination matrices, one without additional columns
    * and one with additional columns of zeros.
    */
  def testSignCombinationMatrices:Unit = {

    print("\n\nsignCombinationMatrix("+4+"):\n\n")
    val sgnM = MatrixUtils.signCombinationMatrix(4)
    MatrixUtils.print(sgnM,digits=1)

    print("\n\nsignCombinationMatrix("+4+","+2+","+3+"):\n\n")
    val sgnM1 = MatrixUtils.signCombinationMatrix(4,2,3)
    MatrixUtils.print(sgnM1,digits=1)

    print("\n\nsignCombinationMatrix("+4+","+0+","+2+"):\n\n")
    val sgnM2 = MatrixUtils.signCombinationMatrix(4,0,2)
    MatrixUtils.print(sgnM2,digits=1)

    print("\n\nsignCombinationMatrix("+4+","+3+","+4+"):\n\n")
    val sgnM3 = MatrixUtils.signCombinationMatrix(4,3,4)
    MatrixUtils.print(sgnM3,digits=1)
  }

  /** Run all tests in dimension dim with reps repetitions of each test.
    */
  def runAll(dim:Int,reps:Int,tol:Double):Unit = {

    print("\n\n Solving Ix=-b:\n")
    val I = DenseMatrix.eye[Double](dim)
    val x = DenseVector.rand[Double](dim)
    testSolveWithPreconditioning(I,-x,"System Ix=-b")

    // run all tests for the following values of the regularizer delta:
    for(delta <- List(0.0,0.5,1.0)){

      var msg = "\n\n\n###---MatrixUtilsTests: all errors relative and in L_2-norm:---###"
      msg += "\nRegularizer delta = "+delta
      if(Math.abs(delta)<0.05)
        msg += "\nWARNING: note the astronomical condition numbers!!\n"
      print(msg)

      print("\n\n#----- Tests of forwardSolve Lx=b:\n")
      testForwardSolve(dim,delta,reps)
      print("\n\n#----- Tests of backSolve Ux=b:\n")
      testBackSolve(dim,delta,reps)
      print("\n\n#----- Tests of solveWithPreconditioning Hx=b:\n")
      testSolveWithPreconditioning(dim,reps)
    }
    testSolveUnderdetermined(dim,reps)
    testTriangularSolve(dim,10,reps,tol)
    val nTests = 3; val condNum = 1e12; val debugLevel = 3; val tolEqSolve=1e-4
    val dimKernel =  0
    testSvdSolve(nTests,dim,condNum,dimKernel,tolEqSolve,debugLevel)
  }

  /** We allocate N nxn random matrices A with condition number condNum.
    * Then compute the condition number after Ruiz equilibration and write
    * the ratios condNum(equilibrated(A))/condNum to the file logs/condNumRatio.txt
    *
    * The purpose is to verify that Ruiz equilibration improves the condition
    * nunmbers of ill conditioned matrices.
    */
  def testRandomMatrixCondNum(N:Int,n:Int,condNum:Double):Unit = {

    val condNumRatio = DenseVector.zeros[Double](N)
    println("Computing condition numbers: ")
    val d = if(N>100) N/20 else 5
    for(k <- 0 until N ){

      if(k%d==0) print("*")
      val M = MatrixUtils.randomMatrix(n,condNum)
      val A = M+M.t
      val rA = MatrixUtils.ruizEquilibrate(A)    // pair (d,Q) with Q=DAD with D=diag(d)
      val B = rA._2
      condNumRatio(k) = MatrixUtils.conditionNumber(B)/condNum
    }
    val logger = Logger("logs/condNumRatio.txt")
    logger.println("Ratios condNum(equilibrated_A)/condNum(A):\n")
    MatrixUtils.print(condNumRatio,logger,digits=1)
    logger.close()

    println("\nFinished, results in logs/condNumRatio.txt.\n")
  }
}

