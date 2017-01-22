package cvx



/** Main class, runs all tests from main method.
  **/
object Runner extends App {

    /** Run the various tests and benchmarks.*/
    override def main(args: Array[String]) {

        val doTestMatrixUtils = true
        val doTestProblems = false
        val doMinX1 = false
        val doKktTests = false

        // solver parameters
        val maxIter = 200           // max number of Newton steps computed
        val alpha = 0.05            // line search descent factor
        val beta = 0.75             // line search backtrack factor
        val tolSolver = 1e-12       // tolerance for norm of gradient, duality gap
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

            //KktTest.testSolutionWithCholFactor(5,100,10,1e-10)
            KktTest.testPositiveDefinite(5,1000,100,1e-10)
        }

        if(doTestProblems){

            val dim = 100               // dimension of objective function
            MinimizationTests.testStandardProblems(dim,pars,tolSolution)
        }

        if(doMinX1){

            MinimizationTests.testMinX1(pars,tolSolution)
        }





    }
}