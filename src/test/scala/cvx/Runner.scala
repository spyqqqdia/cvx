package cvx

import breeze.linalg.DenseVector
import cvx.SimpleOptimizationProblems._


/** Main class, runs all tests from main method.
  **/
object Runner extends App {

  /** Run the various tests and benchmarks.*/
  override def main(args: Array[String]) {

    println("Doing test problems."); Console.flush()

    val debugLevel=0

    val doAdHoc = true

    val doTestMatrixUtils = false
    val doKktTests = false

    val doTestPowerProblems = false
    val doTestStandardProblems = false
    val doTestKlProblems = false
    val doTestInfeasibleKlProblems = false
    val doFeasibilityTests = false

    val tolEqSolve = 1e-1
    val tolSolution = 1e-2      // tolerance for solution identification
    val solverType = "PD"       // "BR", "PD"


    if(doAdHoc){

      val debugLevel=1
      val dim = 10   // at dim 41 distanceFromOrigin1 gets 20 times slower
      //val problem = minX1_no_FP(solverType,debugLevel)
      //val a = DenseVector.tabulate[Double](dim)(i => 1.0)
      //val problem = minDotProduct(a,solverType,debugLevel)
      //val problem = min_pNorm(dim,p=2.2,solverType,debugLevel)
      //val problem = min_pNorm(dim,p=4,solverType,debugLevel)
      //val problem = rankOneProblemSimplex(dim,solverType,debugLevel)
      //val problem = rankOneProblemSphere(dim,solverType,debugLevel)
      //val problem = joptP1(dim,solverType,debugLevel)
      //val problem = joptP2(solverType,debugLevel)
      //val problem = normSquaredWithFreeVariables(dim,solverType,debugLevel)
      val problem = OptimizationProblems.kl_2A(dim,solverType,debugLevel)

      MinimizationTests.runProblem(problem,tolSolution,debugLevel)

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

      val dim = 10               // dimension of objective function
      val condNumber = 100   // condition number of matrix A
      MinimizationTests.testStandardProblems(dim,condNumber,solverType,tolSolution,debugLevel)
    }

    if(doTestKlProblems){

      val dim = 120
      MinimizationTests.test_KL_problems(dim,solverType,tolSolution,debugLevel)
    }

    if(doTestInfeasibleKlProblems){

      MinimizationTests.test_infeasible_KL_problems(solverType,tolSolution,debugLevel)
    }

    if(doFeasibilityTests){

      val doSOI = false
      val debugLevel = 1
      val n = 10
      val p = 10; val q = 5
      val x0 = DenseVector.tabulate[Double](10)(j => 1.0)
      val pars = SolverParams.standardParams

      FeasibilityTests.checkFeasibilityProbabilitySimplex(doSOI,n,pars,debugLevel)
      //FeasibilityTests.checkRandomFeasibleConstraints(doSOI,x0,p,q,pars,debugLevel)
      //FeasibilityTests.runAll(doSOI,pars,debugLevel)
    }
  }
}