package cvx



/** Main class, runs all tests from main method.
  *
  * In the caliper benchmarking framework the class running main must be called
  * "Runner".
  * We also run all other tests from this class.
  **/
object Runner extends App {

    /** Run the various tests and benchmarks.*/
    override def main(args: Array[String]) {

        val doTestMatrixUtils = false
        val doTestUnconstrainedMinimization = true


        if(doTestMatrixUtils){

            val dim = 3
            val reps= 10
            MatrixUtilsTests.runAll(dim,reps)
        }

        if(doTestUnconstrainedMinimization){

            val k = 3           // number of random test functions of power type
            val dim = 100       // dimension of objective function
            val maxIter = 100   // max number of Newton steps computed
            UnconstrainedMinimizationTests.testRandomType1Fcns(k,dim,maxIter)
        }





    }
}