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
    def normSquared(dim:Int):OptimizationProblem with KnownMinimizer = {

        val minimizer = KnownMinimizer(DenseVector.zeros[Double](dim),0.0)
        val problem = normSquared(dim,ConvexSet.fullSpace(dim))
        OptimizationProblem.addSolution(problem,minimizer)
    }

    /** Unconstrained optimization problem with objective function as in docs/cvx_notes.pdf,
      * example 2.1, p5 with all functions $\phi_j(u)=pow(u*u,q)$ with $q>1$,
      * i.e. the objective function is globally defined in Euclidean space
      * of dimension dim and has the form
      *           \[ f(x)=\sum_j \alpha_j*pow((a_j dot x)*(a_j dot x),q) \]
      * with positive coefficients $\alpha_j$, $A$ a matrix of dimension m x n, where m <= n,
      * and $a_j=col_j(A)$.
      * Then $n$ is the dimension of the independent variable $x$ and the global minimum
      * is zero and is assumed at all points in the null space of A.
      * If m < dim this space is nontrivial and we can test how the algorithm behaves in such
      * a case.
      *
      * @param pars parameters controlling the solver behaviour (maxIter, backtracking line search
      * parameters etc, see [SolverParams].
      */
    def powerProblem(id:String,A:DenseMatrix[Double],alpha:DenseVector[Double],q:Double,pars:SolverParams):
    OptimizationProblem with KnownMinimizer = {

        val n=A.cols; val m=A.rows
        assert(m<=n)
        val objF:ObjectiveFunction = Type1Function.powerTestFunction(A,alpha,q)
        val C = ConvexSet.fullSpace(n)
        val minimizer = new KnownMinimizer {

            def isMinimizer(x:DenseVector[Double],tol:Double) = norm(A*x)<tol
            def minimumValue = 0.0
        }
        val problem = OptimizationProblem.unconstrained(id,n,objF,C,pars)
        OptimizationProblem.addSolution(problem,minimizer)
    }

    /** [powerProblem] in dimension dim with m x dim matrix A and coefficient vector alpha
      * having random entries in (0,1). In addition 1.0 is added to the diagonal entries of
      * A to improve the condition number.
      *
      * @param m we must have m<=dim.
      */
    def randomPowerProblem(id:String,dim:Int,m:Int,q:Double,pars:SolverParams):
    OptimizationProblem with KnownMinimizer = {

        assert(m<=dim)
        val A = DenseMatrix.rand[Double](m,dim)
        for(i <- 0 until m) A(i,i)+=1.0
        val alpha = DenseVector.rand[Double](m)
        powerProblem(id,A,alpha,q,pars)
    }


    /** @return list of OptimizationProblems in dimension dim with known solution as follows:
      * first the following unconstrained problems
      *     f(x) = x dot x, followed by
      *     3 [randomPowerProblem]s with one dimensional solution space (m = dim-1)
      *
      * No constrained problems as of yet. The list will be expanded continually.
      *
      * @param dim common dimension of all problems, must be >= 2.
      * @param pars parameters controlling the solver behaviour (maxIter, backtracking line search
      * parameters etc, see [SolverParams].
      */
    def standardProblems(dim:Int,pars:SolverParams):List[OptimizationProblem with KnownMinimizer] = {

        var theList:List[OptimizationProblem with KnownMinimizer] = List(normSquared(dim))
        for(j <- 1 to 5){

            val q = 1.0+rand()
            val m = dim-1        // rank of A, so solution space = ker(A) is one dimensional
            val id = "Random power problem in dimension "+dim+" with m="+dim+"-1 and exponent 2*"+q
            theList = theList :+ randomPowerProblem(id,dim,m,q,pars)
        }
        theList
    }
}
