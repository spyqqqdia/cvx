package cvx

import breeze.linalg.{DenseMatrix, DenseVector, lowerTriangular, norm, sum}
import breeze.stats.distributions.Rand

/**
  * Created by oar on 1/21/17.
  *
  * Test for solving KKT systems.
  */
object KktTest {


  


  /** This tests the function [KKTData.paddVector].
    */
  def testSolutionPadding(nTests:Int): Unit ={

    println("\nTesting KKT solution padding:\n")
    val n=10
    val randIntG = Rand.randInt(7)
    for(j <- 0 until nTests){

      val x:DenseVector[Double] = DenseVector.ones(n)
      val shift = -2+randIntG.draw()
      val nullIndices = Vector[Int](2+shift,4+shift,8+shift)

      println("\nTest "+j+"\n")
      println("Vector x:  "+x)
      println("NullIndices: "+nullIndices)

      val z = KKTData.paddVector(x,nullIndices)
      println("padded vector x: "+z)
    }
  }

  /** Allocates a KKTData object for a convex problem without inequality
    * conditions (e.g.: barrier method) where neither the objective function nor the
    * equality conditions depend on the variables x_j with j in nullIndices.
    *
    * This the Hessian H has zero rows and columns at all indices j in nullIndices,
    * zero columns in the matrix A of the equality conditions Ax=b and the vector g
    * satisfies g(j)=0, for all such j.
    *
    * We then reduce the system, solve the reduced system pad the solution back to
    * the original size by filling the value zero for the eliminated variables
    * x_j, j in nullIndices, and check if the padded solution satisfies the original
    * KKT system.
    */
  def testKktSystemReduction(
    nTests:Int,nullIndices:Vector[Int],delta:Double,tolEqSolve:Double,
    logger:Logger,debugLevel:Int
  ):Unit = for(tt <- 1 to nTests) {

    // check if the indices in nullIndices are strictly increasing
    val n = nullIndices.length
    assert((0 until (n-1)).foldLeft(true)((b:Boolean,j:Int)=> if(nullIndices(j)<nullIndices(j+1)) b else false),
    "\nVector nullIndices not strictly increasing: "+nullIndices+"\n")

    val dim = nullIndices(n-1)+5    // the largest index, number of variables x_j will be m+5
    val rng = scala.util.Random

    // allocate the Hessian
    val Q = DenseMatrix.tabulate[Double](dim,dim)((i,j) => -1+2*rng.nextDouble())
    val H:DenseMatrix[Double] = Q.t*Q    // symmetric, positive definite
    // wipe out j-th row and column
    for(j <- nullIndices)
      for(i <- 0 until dim){ H(i,j)=0; H(j,i)=0 }

    // allocate matrix A, 6 equations
    val A = DenseMatrix.tabulate[Double](6,dim)((i,j) => -1+2*rng.nextDouble())
    // wipe out j-th column
    for(j <- nullIndices)
      for(i <- 0 until A.rows) A(i,j) = 0

    // allocate vectors g,r
    val g = DenseVector.tabulate[Double](dim)(i => -2+4*rng.nextDouble())
    // wipe out the j-th coordinate
    for(j <- nullIndices) g(j)=0

    val r = DenseVector.tabulate[Double](A.rows)(i => -1+2*rng.nextDouble())

    val kktData = KKTData(H,A,g,r,Some(nullIndices))
    val reducedKktData = kktData.reduced
    val rA = reducedKktData.A
    val rH = reducedKktData.H
    val rg = reducedKktData.g
    val rr = reducedKktData.r
    val reducedKktSystem = KKTSystem(rH,rA,rg,rr)

    val (rdx,nu) = reducedKktSystem.solve(delta,logger,tolEqSolve,debugLevel)
    // pad the solution back to original size, nu is unaffected
    val dx = KKTData.paddVector(rdx,nullIndices)
    // the solution of the original system is now (dx,nu),
    // we check if Hdx+A'nu=-g, Adx=r
    val err1 = norm(H*dx+A.t*nu+g)
    val err2 = norm(A*dx-r)

    println("\nTest number "+tt+":\n")
    println("||Hdx+A'nu+g|| should equal zero and is = "+err1)
    println("||Anu-r|| should equal zero and is = "+err2+"\n")
  }


  //------------- Tests of type 1 solution ----------------//

  /** Test the solution of system
    *      Hx + A'w = -q
    *      Ax = b
    * where H is nxn and A is pxn where the Cholesky factor H=LL' is given.
    *
    * @param tolEqSolve tolerance (L2-norm) in forward and backward error
    * @return true if both forward and backward error (L2-norm) are less than tol, else false.
    */
  def testSolutionWithCholFactor(
      L:DenseMatrix[Double], A:DenseMatrix[Double],
      x:DenseVector[Double], w:DenseVector[Double], tolEqSolve:Double, debugLevel:Int
  ):Boolean = {

    val H = L*L.t
    val q = -(H*x+A.t*w)
    val b = A*x
    val condH = MatrixUtils.conditionNumber(H)
    println("Condition number of H: "+MathUtils.round(condH,1))

    val logger = Logger("logs/KktTestSolutionWithCholFactor.txt")
    val (x1,w1) = KKTSystem.solveWithCholFactor(L,A,q,b,logger,tolEqSolve,debugLevel)

    val q1 = -(H*x1+A.t*w1)
    val b1 = A*x1

    val forwardErrorAbs = norm(q1-q) + norm(b1-b)
    val backwardErrorAbs = norm(x1-x) + norm(w1-w)
    val forwardErrorRel = norm(q1-q)/norm(q) + norm(b1-b)/norm(b)
    val backwardErrorRel = norm(x1-x)/norm(x) + norm(w1-w)/norm(w)
    println(
      "Forward error (absolute): "+MathUtils.round(forwardErrorAbs,10)+
        ",\t\tforward error (relative): "+MathUtils.round(forwardErrorRel,10)
    )
    println(
      "Backward error (absolute): "+MathUtils.round(backwardErrorAbs,10)+
        ",\t\tbackward error (relative): "+MathUtils.round(backwardErrorRel,10)
    )
    forwardErrorRel<tolEqSolve && backwardErrorRel<tolEqSolve
  }

  /** Test the solution of random system
    *      Hx + A'w = -q
    *      Ax = b
    * where H is nxn and A is pxn where the Cholesky factor H=LL' is given.
    *
    * @param tolEqSolve tolerance (L2-norm) in forward and backward error
    * @return true if both forward and backward error (L2-norm) are less than tol, else false.
    */
  def testSolutionWithCholFactor(n:Int,p:Int,tolEqSolve:Double,debugLevel:Int):Boolean = {

    println("\n\nSolving random KKT system in dimensions n="+n+", p="+p)
    // uniform in (-1,1)
    val Q:DenseMatrix[Double] = DenseMatrix.tabulate(n,n)((i,j) => -5+10*Math.random())
    val L = lowerTriangular(Q)
    (0 until n).map(i => L(i,i) = L(i,i)+Math.sqrt(n))    // improve conditioning

    val A:DenseMatrix[Double] = DenseMatrix.rand(p,n)
    (0 until p).map(i => A(i,i) = A(i,i)+1.0)

    val x = DenseVector.tabulate(n)(i => -1+2*Math.random())     // uniform in (-1,1)
    val w = DenseVector.tabulate(p)(i => -2+4*Math.random())     // uniform in (-2,2)

    testSolutionWithCholFactor(L,A,x,w,tolEqSolve,debugLevel)
  }


  /** Test the solution of m random systems
    *      Hx + A'w = -q
    *      Ax = b
    * where H is nxn and A is pxn where the Cholesky factor H=LL' is given.
    */
  def testSolutionWithCholFactor(m:Int,n:Int,p:Int,tolEqSolve:Double,debugLevel:Int):Boolean = {

    val results = (0 until m).map(i => testSolutionWithCholFactor(n,p,tolEqSolve,debugLevel))
    results.forall(p => p)
  }



  //------------- Tests of type 0 solution ----------------//

  /** Test the solution of system
    *      Hx + A'w = -q
    *      Ax = b
    * where H is nxn and A is pxn.
    *
    * @return true if both forward and backward error (relative, L2-norm) are less than tol, else false.
    */
  def testPositiveDefinite(
    H:DenseMatrix[Double], A:DenseMatrix[Double],
    x:DenseVector[Double], w:DenseVector[Double],
    pars:SolverParams, logger:Logger, debugLevel:Int
  ):Boolean = {

    val q = -(H*x+A.t*w)
    val b = A*x
    val condH = MatrixUtils.conditionNumber(H)
    println("Condition number of H: "+MathUtils.round(condH,1))

    val tolEqSolve = pars.tolEqSolve
    val (x1,w1) = KKTSystem(H,A,q,b).solve(pars.delta,logger,tolEqSolve,debugLevel)

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
    forwardErrorRel < tolEqSolve  &&  backwardErrorRel < tolEqSolve
  }

  /** Test the solution of random system
    *      Hx + A'w = -q
    *      Ax = b
    * where H is nxn and A is pxn.
    *
    * @return true if both forward and backward error (L2-norm) are less than tol, else false.
    */
  def testPositiveDefinite(
    n:Int,p:Int,pars:SolverParams,logger:Logger,debugLevel:Int
  ):Boolean = {

    println("\n\nSolving random KKT system in dimensions n="+n+", p="+p)
    // uniform in (-1,1)
    val Q:DenseMatrix[Double] = DenseMatrix.tabulate(n,n)((i,j) => -5+10*Math.random())
    val L = lowerTriangular(Q)
    (0 until n).map(i => L(i,i) = L(i,i)+Math.sqrt(n))    // improve conditioning

    val M:DenseMatrix[Double] = L*L.t      // positive definite with probability 1

    // make exactly symmetric (processor or openblas problem?)
    val H = (M+M.t)*0.5

    val A:DenseMatrix[Double] = DenseMatrix.tabulate(p,n)((i,j) => -5+10*Math.random())
    (0 until p).map(i => A(i,i) = A(i,i)+20.0)

    val x = DenseVector.tabulate(n)(i => -1+2*Math.random())     // uniform in (-1,1)
    val w = DenseVector.tabulate(p)(i => -2+4*Math.random())     // uniform in (-2,2)

    testPositiveDefinite(H,A,x,w,pars,logger,debugLevel)
  }


  /** Test the solution of m random systems
    *      Hx + A'w = -q
    *      Ax = b
    * where H is nxn and A is pxn.
    */
  def testPositiveDefinite(
    m:Int,n:Int,p:Int,pars:SolverParams,logger:Logger,debugLevel:Int
  ):Boolean = {

    val results = (0 until m).map(i => testPositiveDefinite(n,p,pars,logger,debugLevel))
    results.forall(p => p)
  }
}