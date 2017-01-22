package cvx

import breeze.linalg.{DenseMatrix, DenseVector, norm, sum, lowerTriangular}

/**
  * Created by oar on 1/21/17.
  *
  * Test for solving KKT systems.
  */
object KktTest {


    //------------- Tests of type 1 solution ----------------//

    /** Test the solution of system
      *      Hx + A'w = -q
      *      Ax = b
      * where H is nxn and A is pxn where the Cholesky factor H=LL' is given.
      *
      * @param tol tolerance (L2-norm) in forward and backward error
      * @return true if both forward and backward error (L2-norm) are less than tol, else false.
      */
    def testSolutionWithCholFactor(
        L:DenseMatrix[Double], A:DenseMatrix[Double],
        x:DenseVector[Double], w:DenseVector[Double], tol:Double
    ):Boolean = {

        println("\nSolving KKT system:")
        val H = L*L.t
        val q = -(H*x+A.t*w)
        val b = A*x
        val condH = MatrixUtils.conditionNumber(H)
        println("Condition number of H: "+condH)

        val kktType = 0
        val (x1,w1) = KKTSystem.solveWithCholFactor(L,A,q,b)

        val q1 = -(H*x1+A.t*w1)
        val b1 = A*x1

        val forwardErrorAbs = norm(q1-q) + norm(b1-b)
        val backwardErrorAbs = norm(x1-x) + norm(w1-w)
        val forwardErrorRel = norm(q1-q)/norm(q) + norm(b1-b)/norm(b)
        val backwardErrorRel = norm(x1-x)/norm(x) + norm(w1-w)/norm(w)
        println(
            "Forward error (absolute): "+MathUtils.round(forwardErrorAbs,2)+
                ",\t\tforward error (relative): "+MathUtils.round(forwardErrorRel,4)
        )
        println(
            "Backward error (absolute): "+MathUtils.round(backwardErrorAbs,2)+
                ",\t\tbackward error (relative): "+MathUtils.round(backwardErrorRel,4)
        )
        forwardErrorRel<tol && backwardErrorRel<tol
    }

    /** Test the solution of random system
      *      Hx + A'w = -q
      *      Ax = b
      * where H is nxn and A is pxn where the Cholesky factor H=LL' is given.
      *
      * @param tol tolerance (L2-norm) in forward and backward error
      * @return true if both forward and backward error (L2-norm) are less than tol, else false.
      */
    def testSolutionWithCholFactor(n:Int,p:Int,tol:Double):Boolean = {

        println("\nSolving random KKT system:")
        // uniform in (-1,1)
        val Q:DenseMatrix[Double] = DenseMatrix.tabulate(n,n)((i,j) => -5+10*Math.random())
        val L = lowerTriangular(Q)
        (0 until n).map(i => L(i,i) = L(i,i)+20.0)    // improve conditioning

        val A:DenseMatrix[Double] = DenseMatrix.rand(p,n)
        (0 until p).map(i => A(i,i) = A(i,i)+1.0)

        val x = DenseVector.tabulate(n)(i => -1+2*Math.random())     // uniform in (-1,1)
        val w = DenseVector.tabulate(p)(i => -2+4*Math.random())     // uniform in (-2,2)

        testSolutionWithCholFactor(L,A,x,w,tol)
    }


    /** Test the solution of m random systems
      *      Hx + A'w = -q
      *      Ax = b
      * where H is nxn and A is pxn where the Cholesky factor H=LL' is given.
      */
    def testSolutionWithCholFactor(m:Int,n:Int,p:Int,tol:Double):Boolean = {

        val results = (0 until m).map(i => testSolutionWithCholFactor(n,p,tol))
        results.forall(p => p)
    }



    //------------- Tests of type 0 solution ----------------//

    /** Test the solution of system
      *      Hx + A'w = -q
      *      Ax = b
      * where H is nxn and A is pxn.
      *
      * @param tol tolerance (L2-norm) in forward and backward error
      * @return true if both forward and backward error (relative, L2-norm) are less than tol, else false.
      */
    def testPositiveDefinite(
        H:DenseMatrix[Double], A:DenseMatrix[Double],
        x:DenseVector[Double], w:DenseVector[Double], tol:Double
    ):Boolean = {

        println("\nSolving KKT system:")
        val q = -(H*x+A.t*w)
        val b = A*x
        val condH = MatrixUtils.conditionNumber(H)
        println("Condition number of H: "+condH)

        val kktType = 0
        val (x1,w1) = KKTSystem(H,A,q,b,kktType).solve

        val q1 = -(H*x1+A.t*w1)
        val b1 = A*x1

        val forwardErrorAbs = norm(q1-q) + norm(b1-b)
        val backwardErrorAbs = norm(x1-x) + norm(w1-w)
        val forwardErrorRel = norm(q1-q)/norm(q) + norm(b1-b)/norm(b)
        val backwardErrorRel = norm(x1-x)/norm(x) + norm(w1-w)/norm(w)
        println(
            "Forward error (absolute): "+MathUtils.round(forwardErrorAbs,2)+
            ",\t\tforward error (relative): "+MathUtils.round(forwardErrorRel,4)
        )
        println(
            "Backward error (absolute): "+MathUtils.round(backwardErrorAbs,2)+
            ",\t\tbackward error (relative): "+MathUtils.round(backwardErrorRel,4)
        )
        forwardErrorRel<tol && backwardErrorRel<tol
    }

    /** Test the solution of random system
      *      Hx + A'w = -q
      *      Ax = b
      * where H is nxn and A is pxn.
      *
      * @param tol tolerance (L2-norm) in forward and backward error
      * @return true if both forward and backward error (L2-norm) are less than tol, else false.
      */
    def testPositiveDefinite(n:Int,p:Int,tol:Double):Boolean = {

        println("\nSolving random KKT system:")
        // uniform in (-1,1)
        val Q:DenseMatrix[Double] = DenseMatrix.tabulate(n,n)((i,j) => -5+10*Math.random())

        val M:DenseMatrix[Double] = Q*Q.t      // positive definite with probability 1
        (0 until n).map(i => M(i,i) = M(i,i)+20.0)    // improve conditioning

        // make exactly symmetric (processor or openblas problem?)
        val H = (M+M.t)*0.5

        val A:DenseMatrix[Double] = DenseMatrix.tabulate(p,n)((i,j) => -5+10*Math.random())
        (0 until p).map(i => A(i,i) = A(i,i)+20.0)

        val x = DenseVector.tabulate(n)(i => -1+2*Math.random())     // uniform in (-1,1)
        val w = DenseVector.tabulate(p)(i => -2+4*Math.random())     // uniform in (-2,2)

        testPositiveDefinite(H,A,x,w,tol)
    }


    /** Test the solution of m random systems
      *      Hx + A'w = -q
      *      Ax = b
      * where H is nxn and A is pxn.
      */
    def testPositiveDefinite(m:Int,n:Int,p:Int,tol:Double):Boolean = {

        val results = (0 until m).map(i => testPositiveDefinite(n,p,tol))
        results.forall(p => p)
    }
}
