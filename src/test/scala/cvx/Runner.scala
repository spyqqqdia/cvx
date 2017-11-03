package cvx



/** Main class, runs all tests from main method.
  **/
object Runner extends App {

  /** Run the various tests and benchmarks.*/
  override def main(args: Array[String]) {

    println("Doing test problems."); Console.flush()

    val debugLevel=0

    val doTestMatrixUtils = false
    val doKktTests = false

    val doTestStandardProblems = false
    val doMinX1 = false
    val doTestKlProblems = false
    val doTestInfeasibleKlProblems = false
    val doFeasibilityTests = true

    // solver parameters
    val maxIter = 200           // max number of Newton steps computed
    val alpha = 0.05            // line search descent factor
    val beta = 0.75             // line search backtrack factor
    val tolSolver = 1e-8        // tolerance for norm of gradient, duality gap
    val tolSolution = 1e-2      // tolerance for solution identification
    val delta = 1e-8            // regularization A -> A+delta*I if ill conditioned
    val pars = SolverParams(maxIter,alpha,beta,tolSolver,delta)


    if(doTestMatrixUtils){

      val dim = 100
      val reps= 10
      val tol = 1e-10
      MatrixUtilsTests.runAll(dim,reps,tol)
    }

    if(doKktTests){

      val debug = true
      val logFilePath = "logs/KktTestLog.txt"
      val logger = Logger(logFilePath)
      //KktTest.testSolutionWithCholFactor(5,1000,100,1e-10,logger,debug)
      KktTest.testPositiveDefinite(5,1000,100,1e-10,logger,debugLevel)
    }

    if(doTestStandardProblems ){

      val dim = 100               // dimension of objective function
      MinimizationTests.testStandardProblems(dim,pars,tolSolution,debugLevel)
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

      FeasibilityTests.runAll(pars,debugLevel)
    }





  }
}