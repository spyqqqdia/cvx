package cvx

import breeze.linalg.DenseVector


/** Main class, runs all tests from main method.
  **/
object Runner extends App {

  /** Run the various tests and benchmarks.*/
  override def main(args: Array[String]) {

    println("Doing test problems."); Console.flush()

    val debugLevel=2

    val doTestMatrixUtils = false
    val doKktTests = false

    val doTestPowerProblems = false
    val doTestStandardProblems = true
    val doMinX1 = false
    val doTestKlProblems = false
    val doTestInfeasibleKlProblems = false
    val doFeasibilityTests = false

    // solver parameters
    val maxIter = 200           // max number of Newton steps computed
    val alpha = 0.05            // line search descent factor
    val beta = 0.75             // line search backtrack factor
    val tolSolver = 1e-7        // tolerance for norm of gradient, duality gap
    val tolEqSolve = 10         // tolerance in the solution of the KKT system
    val tolFeas = 1e-9          // tolerance in inequality and equality constraints
    val delta = 1e-8            // regularization A -> A+delta*I if ill conditioned
    val pars = SolverParams(maxIter,alpha,beta,tolSolver,tolEqSolve,tolFeas,delta)

    val tolSolution = 1e-2      // tolerance for solution identification


    if(doTestMatrixUtils){

      val dim = 100
      val reps= 10
      val tol = 1e-10
      //MatrixUtilsTests.runAll(dim,reps,tol)
      //MatrixUtilsTests.testSignCombinationMatrices
      MatrixUtilsTests.testRandomMatrixCondNum(1000,100)
    }

    if(doKktTests){

      val nTests = 5
      val debug = true
      val logFilePath = "logs/KktTestLog.txt"
      val logger = Logger(logFilePath)
      val nullIndices = Vector[Int](2,5,7)
      //KktTest.testSolutionWithCholFactor(5,1000,100,1e-10,logger,debug)
      //KktTest.testPositiveDefinite(5,1000,100,1e-10,logger,debugLevel)
      //KktTest.testSolutionPadding(nTests)
      KktTest.testKktSystemReduction(nTests,nullIndices,logger,tolEqSolve,debugLevel)
    }


    if(doTestPowerProblems ){

      MinimizationTests.testPowerProblems(pars,tolSolution,debugLevel)
    }

    if(doTestStandardProblems ){

      val dim = 100               // dimension of objective function
      val condNumber = 100   // condition number of matrix A
      MinimizationTests.testStandardProblems(dim,condNumber,pars,tolSolution,debugLevel)
    }

    if(doMinX1){

      MinimizationTests.testMinX1(pars,tolSolution,debugLevel)
    }

    if(doTestKlProblems){

      MinimizationTests.test_KL_problems(pars,tolSolution,debugLevel)
    }

    if(doTestInfeasibleKlProblems){

      MinimizationTests.test_infeasible_KL_problems(pars,tolSolution,debugLevel)
    }

    if(doFeasibilityTests){

      val p = 10; val q = 5
      val x0 = DenseVector.tabulate[Double](10)(j => 1.0)
      //FeasibilityTests.checkRandomFeasibleConstraints(x0,p,q,pars,debugLevel)
      FeasibilityTests.runAll(pars,debugLevel)
    }





  }
}