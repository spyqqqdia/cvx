package cvx

import breeze.linalg.{DenseMatrix, DenseVector, _}


/**
  * Created by oar on 12/3/16.
  */
object MatrixUtilsTests {

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
    def testSolve(H:DenseMatrix[Double], x:DenseVector[Double], testID:String):Unit = {

        val dim = H.rows
        val b = H*x

        val delta = 1e-15
        val u = MatrixUtils.solveWithPreconditioning(H,b,delta)

        val errSol = norm(u-x)/norm(x)
        val errVal = norm(b-H*u)/norm(b)
        var msg = "\n"+testID+": error in solution = "+errSol+",  error in value Hx = "+errVal
        print(msg)
    }


    /** Testing MatrixUtils::solveWithPreconditioning on a matrix H = A'A with random nxn matrix A
      * and right hand side b = Hx so we know the solution is x. The vector x will also be chosen
      * randomly. All random entries in (0,1).
      */
    def testSolve(dim:Int,testID:String):Unit = {

        val x = DenseVector.rand[Double](dim)
        val A = DenseMatrix.rand[Double](dim,dim)
        val S = A.t*A
        val H = S+S.t         // make symmetric (numerical issues)
        testSolve(H,x,testID)
    }

    /** Run reps tests testSolve(dim,"Test_j") */
    def testSolve(dim:Int,reps:Int):Unit = for(j <- 1 to reps) testSolve(dim,"Test_"+j)

    /** Run all tests in dimension dim with reps repetitions of each test.
      */
    def runAll(dim:Int,reps:Int):Unit = {

        print("\n\n Solving Ix=-b:\n")
        val I = DenseMatrix.eye[Double](dim)
        val x = DenseVector.rand[Double](dim)
        testSolve(I,-x,"System Ix=-b")

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
            testSolve(dim,reps)
        }
    }
}
