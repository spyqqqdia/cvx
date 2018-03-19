package cvx

import breeze.linalg.{DenseMatrix, DenseVector, _}

/**
  * Created by oar on 12/2/16.
  */
object MinimizationTests {

  /** Solve a list of unconstrained Optimization problems with known _unique_ solutions,
    * then report if the computed solutions are correct.
    *
    * @param tol tolerance for distance between an actual (usually the unique) solution and the
    * solution found by the solver, see [OptimizationSolution].
    */
  def runProblemsWithKnownMinimizer(
    optProblems:List[OptimizationProblem with KnownMinimizer], tol:Double, debugLevel:Int
  ):Unit = for(problem <- optProblems) try {

    print("\n\n#-----Problem: "+problem.id+":\n\n")
    Console.flush()

    val sol = problem.solve(debugLevel)
    val x = sol.x                       // minimizer, solution found
    val y = problem.objectiveFunction.valueAt(x)     // value at solution found
    problem.reportWithKnownSolution(sol,y,tol,problem.logger)

  } catch {

    case e:Exception => {

      print("\n\nException "+e.getClass+" occurred")
      print("\nMessage: "+e.getMessage)
      print("\nStacktrace:\n")
      e.printStackTrace()
      problem.logger.close()
    }
  }
  def runProblemWithKnownMinimizer(
    optProblem:OptimizationProblem with KnownMinimizer, tol:Double, debugLevel:Int
  ):Unit = runProblemsWithKnownMinimizer(List(optProblem),tol,debugLevel)

  /** Solve a list of Optimization problems where the solution may not be known
    * or is known to be infeasible.
    *
    * @param tol tolerance for all sorts of things.
    */
  def runProblems(
    optProblems:List[OptimizationProblem], tol:Double, debugLevel:Int
  ):Unit = for(problem <- optProblems) try {

    print("\n\n#-----Problem: "+problem.id+":\n\n")

    val sol = problem.solve(debugLevel)
    problem.report(sol,tol)

  } catch {

    case e:Exception => {

      print("\n\nException "+e.getClass+" occurred")
      print("\nMessage: "+e.getMessage)
      print("\nStacktrace:\n")
      e.printStackTrace()
      problem.logger.close()
    }
  }
  def runProblem(
    optProblem:OptimizationProblem, tol:Double, debugLevel:Int
  ):Unit = runProblems(List(optProblem),tol,debugLevel)

  /** Run the two simple power problems [OptimizationProblems.powerProblems].
    */
  def testPowerProblems(tol:Double,debugLevel:Int):Unit = {

    val problems = OptimizationProblems.powerProblems(debugLevel)
    runProblemsWithKnownMinimizer(problems,tol,debugLevel)
  }

  /** Test the standard list, [OptimizationProblems.standardProblems] in dimension dim.
    *
    * @param dim dimension of independent variable.
    * @param condNumber: condition number of the matrix A in the power problems.
    * @param solverType: "BR" (barrier solver), "PD0" (primal dual with one slack variable),
    *   "PD1" (primal dual with one slack variable for each inequality constraint), see
    *    docs/primaldual.pdf.
    * @param tol tolerance for deviation from the known solution.
    */
  def testStandardProblems(
    dim:Int,condNumber:Double,solverType:String,tol:Double,debugLevel:Int
  ):Unit = {

    val problems = SimpleOptimizationProblems.standardProblems(dim,condNumber,solverType,debugLevel)
    runProblemsWithKnownMinimizer(problems,tol,debugLevel)
  }

  /** Test the single problem OptimizationProblems.minX1.*/
  def testMinX1(solverType:String,tol:Double,debugLevel:Int):Unit = {


    val problems = List(
      SimpleOptimizationProblems.minX1_withFP(solverType,debugLevel),
      SimpleOptimizationProblems.minX1_no_FP(solverType,debugLevel)
    )
    runProblemsWithKnownMinimizer(problems,tol,debugLevel)
  }

  /** Test the KL-problems (geometric centering) with known solutions.*/
  def test_KL_problems(dim:Int,solverType:String,tol:Double,debugLevel:Int):Unit = {

    val problems = List(
      OptimizationProblems.kl_1(dim,solverType,debugLevel),
      OptimizationProblems.kl_2(dim,solverType,debugLevel)
    )
    runProblemsWithKnownMinimizer(problems,tol,debugLevel)
  }

  /** Test the KL-problems (geometric centering) known to be infeasible.*/
  def test_infeasible_KL_problems(solverType:String,tol:Double,debugLevel:Int):Unit = {

    val problems = List(
      OptimizationProblems.infeasible_kl_1(12,solverType,debugLevel)
    )
    runProblems(problems,tol,debugLevel)
  }


}