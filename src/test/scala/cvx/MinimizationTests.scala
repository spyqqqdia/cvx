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
  def testPowerProblems(pars:SolverParams,tol:Double,debugLevel:Int):Unit = {

    val problems = OptimizationProblems.powerProblems(pars,debugLevel)
    runProblemsWithKnownMinimizer(problems,tol,debugLevel)
  }

  /** Test the standard list, [OptimizationProblems.standardProblems] in dimension dim.
    *
    * @param dim dimension of independent variable.
    * @param condNumber: condition number of the matrix A in the power problems.
    * @param pars parameters controlling the solver behaviour (maxIter, backtracking line search
    * parameters etc, see [SolverParams].
    * @param tol tolerance for deviation from the known solution.
    */
  def testStandardProblems(
    dim:Int,condNumber:Double,pars:SolverParams,tol:Double,debugLevel:Int
  ):Unit = {

    val problems = SimpleOptimizationProblems.standardProblems(dim,condNumber,pars,debugLevel)
    runProblemsWithKnownMinimizer(problems,tol,debugLevel)
  }

  /** Test the single problem OptimizationProblems.minX1.*/
  def testMinX1(pars:SolverParams,tol:Double,debugLevel:Int):Unit = {


    val problems = List(
      SimpleOptimizationProblems.minX1_FP(pars,debugLevel),
      SimpleOptimizationProblems.minX1_no_FP(pars,debugLevel)
    )
    runProblemsWithKnownMinimizer(problems,tol,debugLevel)
  }

  /** Test the KL-problems (geometric centering) with known solutions.*/
  def test_KL_problems(dim:Int,pars:SolverParams,tol:Double,debugLevel:Int):Unit = {

    val problems = List(
      OptimizationProblems.kl_1(dim,pars,debugLevel),
      OptimizationProblems.kl_2(dim,pars,debugLevel)
    )
    runProblemsWithKnownMinimizer(problems,tol,debugLevel)
  }

  /** Test the KL-problems (geometric centering) known to be infeasible.*/
  def test_infeasible_KL_problems(pars:SolverParams,tol:Double,debugLevel:Int):Unit = {

    val problems = List(
      OptimizationProblems.infeasible_kl_1(12,pars,debugLevel)
    )
    runProblems(problems,tol,debugLevel)
  }


}