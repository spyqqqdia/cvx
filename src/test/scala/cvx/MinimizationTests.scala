package cvx

import breeze.linalg.{DenseMatrix, DenseVector, _}

/**
  * Created by oar on 12/2/16.
  */
object MinimizationTests {

    /** Solve a list of unconstrained Optimization problems with known solutions, then report
      * if the computed solutions are correct.
      *
      * @param tol tolerance for distance between an actual (usually the unique) solution and the
      * solution found by the solver, see [OptimizationSolution].
      */
    def testList(optProblems:List[OptimizationProblem with KnownMinimizer], tol:Double):Unit =
        for(problem <- optProblems) try {

            print("\n\n#-----Problem: "+problem.id+":\n\n")

            val sol = problem.solve
            val x = sol.x                       // minimizer, solution found
            val y = problem.objF.valueAt(x)     // value at solution found
            val y_opt = problem.minimumValue

            val newtonDecrement = sol.gap      // Newton decrement at solution
            val normGrad = sol.normGrad        // norm of gradient at solution
            val iter = sol.iter
            val maxedOut = sol.maxedOut
            val isSolution = problem.isMinimizer(x,tol)

            var msg = "Iterations = "+iter+"; maxiter reached: "+maxedOut+"\n"
            msg += "Newton decrement:  "+MathUtils.round(newtonDecrement,10)+"\n"
            msg += "norm of gradient:  "+MathUtils.round(normGrad,10)+"\n"
            msg += "value at solution y=f(x):  "+MathUtils.round(y,10)+"\n"
            msg += "value of global min:  "+MathUtils.round(y_opt,10)+"\n"
            msg += "Is global solution at tolerance "+tol+": "+isSolution+"\n"
            print(msg)
        } catch {

            case e:Exception => {

                print("Exception of class "+e.getClass+"occurred")
                print("\nMessage: "+e.getMessage)
                print("\nStacktrace:\n")
                e.printStackTrace()
            }

        }


    /** Test the standard list, [OptimizationProblems.standardProblems] in dimension dim.
      *
      * @param dim dimension of independent variable.
      * @param pars parameters controlling the solver behaviour (maxIter, backtracking line search
      * parameters etc, see [SolverParams].
      * @param tol tolerance for deviation from the known solution.
      */
    def testStandardProblems(dim:Int,pars:SolverParams,tol:Double):Unit = {

        val problems = OptimizationProblems.standardProblems(dim,pars)
        testList(problems,tol)
    }

    /** Test the single problem OptimizationProblems.minX1.*/
    def testMinX1(pars:SolverParams,tol:Double):Unit = {

        val problems = List(OptimizationProblems.minX1(pars))
        testList(problems,tol)
    }
}
