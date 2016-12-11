package cvx

import breeze.linalg.{DenseMatrix, DenseVector, _}

/**
  * Created by oar on 12/2/16.
  */
object UnconstrainedMinimizationTests {

    /** Solve a list of unconstrained Optimization problems with known solutions, then report
      * if the computed solutions are correct.
      *
      * @param tol tolerance for distance between an actual (usually the unique) solution and the
      * solution found by the solver, see [OptimizationSolution].
      */
    def testList(optProblems:List[OptimizationProblem with OptimizationSolution],tol:Double):Unit =
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

            case e:breeze.linalg.NotConvergedException => print(e.getMessage())

        }


    /** Test minimizing the the function f(x)=pow(||x||,2) followed by a list of
      * k type 1 random test functions of power type.
      *
      * @param dim dimension of domain.
      */
    def testRandomType1Fcns(k:Int,dim:Int,tol:Double):Unit = {

        //--- FIX ME: implement the OptimizationProblems.randomPowerProblem function.

        /*
        val fncs_pow:List[OptimizationProblem with OptimizationSolution] =
            (1 to k).map(i => OptimizationProblems.randomPowerProblem(dim,1+randomDouble())).toList

        val fncs = List(OptimizationProblems.normSquared(dim)):::fncs_pow
        testList(fncs,tol)
        */
    }
}
