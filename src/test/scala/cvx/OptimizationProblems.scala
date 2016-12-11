package cvx

import breeze.linalg.{DenseVector, _}

/**
  * Created by oar on 12/11/16.
  *
  * Collection of convex minimization problems with and without constraints.
  */
object OptimizationProblems {

        /** f(x) = (1/2)*(x dot x).*/
        def normSquared(dim:Int,C:ConvexSet with SamplePoint):OptimizationProblem = {

            assert(C.dim==dim)
            val id = "f(x) = 0.5*||x||^2  in dimension "+dim
            val objF = ObjectiveFunctions.normSquared(dim)
            val maxIter = 200; val alpha = 0.1; val beta = 0.5; val tol = 1e-8; val delta = 1e-8
            val pars = SolverParams(maxIter,alpha,beta,tol,delta)
            OptimizationProblem.unconstrained(id,dim,objF,C,pars)
        }

        /** f(x) = (1/2)*(x dot x) on the full Euclidean Space*/
        def normSquared(dim:Int):OptimizationProblem with OptimizationSolution = {

            val optSol = OptimizationSolution(DenseVector.zeros[Double](dim),0.0)
            val problem = normSquared(dim,ConvexSet.fullSpace(dim))
            OptimizationProblem.addSolution(problem,optSol)
        }



}
