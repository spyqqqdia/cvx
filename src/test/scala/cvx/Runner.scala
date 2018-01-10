package cvx

import breeze.linalg.DenseVector


/** Main class, runs all tests from main method.
  **/
object Runner extends App {

  /** Run the various tests and benchmarks.*/
  override def main(args: Array[String]) {

    println("Doing test problems."); Console.flush()

    val debugLevel=2

    val doAdHoc = true

    val doTestMatrixUtils = false
    val doKktTests = false

    val doTestPowerProblems = false
    val doTestStandardProblems = false
    val doMinX1 = false
    val doTestKlProblems = false
    val doTestInfeasibleKlProblems = false
    val doFeasibilityTests = false

    // solver parameters
    val maxIter = 500           // max number of Newton steps computed
    val alpha = 0.07            // line search descent factor
    val beta = 0.6              // line search backtrack factor
    val tolSolver = 1e-8        // tolerance for norm of gradient, duality gap
    val tolEqSolve = 1e-2       // tolerance in the solution of the KKT system
    val tolFeas = 1e-9          // tolerance in inequality and equality constraints
    val delta = 1e-7            // regularization A -> A+delta*I if ill conditioned
    val pars = SolverParams(maxIter,alpha,beta,tolSolver,tolEqSolve,tolFeas,delta)

    val tolSolution = 1e-2      // tolerance for solution identification


    if(doAdHoc){

      val dim = 200
      val debugLevel=1
      //val problem = SimpleOptimizationProblems.normSquaredWithFreeVariables(dim,pars,debugLevel)
      //val problem = SimpleOptimizationProblems.joptP2(pars,debugLevel)
      //val problem = SimpleOptimizationProblems.probabilitySimplexProblem(dim,pars,debugLevel)
      val problem = SimpleOptimizationProblems.distanceFromOrigin1(dim,pars,debugLevel)

      if(false) {
        val a = DenseVector.fill[Double](dim)(1.0)
        val problem = SimpleOptimizationProblems.minDotProduct(a, pars, debugLevel)
      }
      MinimizationTests.runProblemWithKnownMinimizer(problem,tolSolution,debugLevel)

      //MatrixUtilsTests.testRuizEquilibration
    }


    if(doTestMatrixUtils){

      val dim = 100
      val dimKernel=0
      val nTests= 10
      val tol = 1e-10
      val condNum = 1e14
      //MatrixUtilsTests.runAll(dim,nTests,tol)
      //MatrixUtilsTests.testSignCombinationMatrices
      //MatrixUtilsTests.testRandomMatrixCondNum(500,100,condNum)
      MatrixUtilsTests.testEquationSolve(nTests,dim,condNum,dimKernel,tolEqSolve,debugLevel)
      //MatrixUtilsTests.diagonalizationTest(nTests,dim,condNum)
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
      KktTest.testKktSystemReduction(nTests,nullIndices,logger,pars,debugLevel)
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

      val dim = 12
      MinimizationTests.test_KL_problems(dim,pars,tolSolution,debugLevel)
    }

    if(doTestInfeasibleKlProblems){

      MinimizationTests.test_infeasible_KL_problems(pars,tolSolution,debugLevel)
    }

    if(doFeasibilityTests){

      val debugLevel = 4
      val n = 4
      val p = 10; val q = 5
      val x0 = DenseVector.tabulate[Double](10)(j => 1.0)

      FeasibilityTests.checkFeasibilityProbabilitySimplex(n,pars,debugLevel)
      //FeasibilityTests.checkRandomFeasibleConstraints(x0,p,q,pars,debugLevel)
      //FeasibilityTests.runAll(pars,debugLevel)
    }

  }
}