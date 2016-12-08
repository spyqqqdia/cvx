package cvx

import breeze.linalg.{DenseMatrix, DenseVector, _}

/**
  * Created by oar on 12/2/16.
  */
object UnconstrainedMinimizationTests {

    def testList(testFunctions:List[TestFunction],maxIter:Int):Unit = {

        val alpha = 0.07         // line search descent factor
        val beta = 0.75          // line search backtrack factor
        val delta = 1e-14        // regularizer Hd = -y --> (H+delta*I)d = -y if needed.
        val tol = 1e-7

        for(testFunction <- testFunctions) try {

            print("\n\n#-----"+testFunction.id+":\n\n")
            val gMin = testFunction.globalMin
            val dim = testFunction.dim
            val C = new FullSpace(dim)
            val x0 = DenseVector.rand[Double](dim)
            val solver = testFunction.solver(C)

            val sol = solver.solve(maxIter,alpha,beta,tol,delta)
            val x = sol.x                       // minimizer
            val y = testFunction.valueAt(x)
            val y_opt = testFunction.globalMin
            val newtonDecrement = sol.gap      // Newton decrement at solution
            val normGrad = sol.normGrad        // norm of gradient at solution
            val iter = sol.iter
            val maxedOut = sol.maxedOut
            val isSolution = testFunction.isMinimizer(x,tol)

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
    }

    /** Test minimizing the the function f(x)=||x||^2 followed by a list of
      * k type 1 random test functions of power type.
      *
      * @param dim dimension of domain.
      */
    def testRandomType1Fcns(k:Int,dim:Int,maxIter:Int):Unit = {

        val fncs_pow:List[TestFunction] =
            (1 to k).map(i => Type1TestFunction.randomPowerTestFunction(dim,1+randomDouble())).toList

        val fncs = List(TestFunction.normSquared(dim)):::fncs_pow
        testList(fncs,maxIter)
    }
}
