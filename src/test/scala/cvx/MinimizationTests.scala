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
    val y_opt = problem.minimumValue

    val newtonDecrement = sol.dualityGap      // Newton decrement at solution
    val normGrad = sol.normGrad        // norm of gradient at solution
    val iter = sol.iter
    val maxedOut = sol.maxedOut
    val isSolution = problem.isMinimizer(x,tol)

    var msg = "\n\nIterations = "+iter+"; maxiter reached: "+maxedOut+"\n"
    msg += "Newton decrement:  "+MathUtils.round(newtonDecrement,10)+"\n"
    msg += "norm of gradient:  "+MathUtils.round(normGrad,10)+"\n"
    msg += "value at solution y=f(x):  "+MathUtils.round(y,10)+"\n"
    msg += "value of global min:  "+MathUtils.round(y_opt,10)+"\n"
    msg += "Is global solution at tolerance "+tol+": "+isSolution+"\n"
    msg += "Solution x:\n"+x+"\n\n"
    print(msg)
    Console.flush()
    problem.logger.close()

  } catch {

    case e:Exception =>

      print("\n\nException of class "+e.getClass+"occurred")
      print("\nMessage: "+e.getMessage)
      print("\nStacktrace:\n")
      e.printStackTrace()
      problem.logger.close()

  }

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
    val x = sol.x                       // minimizer, solution found
    val y = problem.objectiveFunction.valueAt(x)     // value at solution found
    val newtonDecrement = sol.dualityGap      // Newton decrement at solution
    val normGrad = sol.normGrad        // norm of gradient at solution
    val iter = sol.iter
    val maxedOut = sol.maxedOut

    var msg = "\n\nIterations = "+iter+"; maxiter reached: "+maxedOut+"\n"
    msg += "Newton decrement:  "+MathUtils.round(newtonDecrement,10)+"\n"
    msg += "norm of gradient:  "+MathUtils.round(normGrad,10)+"\n"
    msg += "value at solution y=f(x):  "+MathUtils.round(y,10)+"\n"
    msg += "Solution x:\n"+x+"\n\n"
    print(msg)
    Console.flush()
    problem.logger.close()

  } catch {

    case e:Exception =>

      print("\n\nException of class "+e.getClass+"occurred")
      print("\nMessage: "+e.getMessage)
      print("\nStacktrace:\n")
      e.printStackTrace()
      problem.logger.close()

  }



  /** Test the standard list, [OptimizationProblems.standardProblems] in dimension dim.
    *
    * @param dim dimension of independent variable.
    * @param pars parameters controlling the solver behaviour (maxIter, backtracking line search
    * parameters etc, see [SolverParams].
    * @param tol tolerance for deviation from the known solution.
    */
  def testStandardProblems(dim:Int,pars:SolverParams,tol:Double,debugLevel:Int):Unit = {

    val problems = OptimizationProblems.standardProblems(dim,pars,debugLevel)
    runProblemsWithKnownMinimizer(problems,tol,debugLevel)
  }

  /** Test the single problem OptimizationProblems.minX1.*/
  def testMinX1(pars:SolverParams,tol:Double,debugLevel:Int):Unit = {


    val problems = List(
      OptimizationProblems.minX1_FP(pars,debugLevel),
      OptimizationProblems.minX1_no_FP(pars,debugLevel)
    )
    runProblemsWithKnownMinimizer(problems,tol,debugLevel)
  }

  /** Test the KL-problems (geometric centering) with known solutions.*/
  def test_KL_problems(pars:SolverParams,tol:Double,debugLevel:Int):Unit = {


    val problems = List(
      OptimizationProblems.kl_1(12,pars,debugLevel),
      OptimizationProblems.kl_2(20,pars,debugLevel)
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