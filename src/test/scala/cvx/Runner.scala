package cvx

import breeze.linalg.DenseVector


/** Main class, runs all tests from main method.
  **/
object Runner extends App {

  /** Run the various tests and benchmarks.*/
  override def main(args: Array[String]) {

    println("Doing test problems."); Console.flush()

    val debugLevel=0

    val doAdHoc = false

    val doTestMatrixUtils = false
    val doKktTests = false

    val doTestPowerProblems = false
    val doTestStandardProblems = false
    val doMinX1 = false
    val doTestKlProblems = true
    val doTestInfeasibleKlProblems = false
    val doFeasibilityTests = false

    val tolEqSolve = 1e-2
    val tolSolution = 1e-2      // tolerance for solution identification
    val solverType = "BR"       // "BR", "PD0", "PD1"


    if(doAdHoc){

      val dim = 100   // at dim 41 distanceFromOrigin1 gets 20 times slower
      val debugLevel=1
      //val problem = SimpleOptimizationProblems.normSquaredWithFreeVariables(dim,solverType,debugLevel)
      //val problem = SimpleOptimizationProblems.joptP2(solverType,debugLevel)
      //val problem = SimpleOptimizationProblems.probabilitySimplexProblem(dim,solverType,debugLevel)
      //val problem = SimpleOptimizationProblems.distanceFromOrigin0(dim,solverType,debugLevel)
      val problem = SimpleOptimizationProblems.distanceFromOrigin1(dim,solverType,debugLevel)

      if(false) {
        val a = DenseVector.fill[Double](dim)(1.0)
        val problem = SimpleOptimizationProblems.minDotProduct(a,solverType,debugLevel)
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
      val delta = 1e-8
      val tolEqSolve = 1e-4
      //KktTest.testSolutionWithCholFactor(5,1000,100,1e-10,logger,debug)
      //KktTest.testPositiveDefinite(5,1000,100,1e-10,logger,debugLevel)
      //KktTest.testSolutionPadding(nTests)
      KktTest.testKktSystemReduction(nTests,nullIndices,delta,tolEqSolve,logger,debugLevel)
    }


    if(doTestPowerProblems ){

      MinimizationTests.testPowerProblems(tolSolution,debugLevel)
    }

    if(doTestStandardProblems ){

      val dim = 100               // dimension of objective function
      val condNumber = 100   // condition number of matrix A
      MinimizationTests.testStandardProblems(dim,condNumber,solverType,tolSolution,debugLevel)
    }

    if(doMinX1){

      MinimizationTests.testMinX1(solverType,tolSolution,debugLevel)
    }

    if(doTestKlProblems){

      val dim = 12
      MinimizationTests.test_KL_problems(dim,solverType,tolSolution,debugLevel)
    }

    if(doTestInfeasibleKlProblems){

      MinimizationTests.test_infeasible_KL_problems(solverType,tolSolution,debugLevel)
    }

    if(doFeasibilityTests){

      val doSOI = false
      val debugLevel = 1
      val n = 100
      val p = 10; val q = 5
      val x0 = DenseVector.tabulate[Double](10)(j => 1.0)
      val pars = SolverParams.standardParams(n)

      FeasibilityTests.checkFeasibilityProbabilitySimplex(doSOI,n,pars,debugLevel)
      //FeasibilityTests.checkRandomFeasibleConstraints(doSOI,x0,p,q,pars,debugLevel)
      //FeasibilityTests.runAll(doSOI,pars,debugLevel)
    }
  }
}