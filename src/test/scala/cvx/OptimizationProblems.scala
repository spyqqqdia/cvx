package cvx

import breeze.linalg.{DenseVector, _}
import org.apache.commons.math3.optim.linear.LinearConstraint

/**
  * Created by oar on 12/11/16.
  *
  * Collection of convex minimization problems with and without constraints.
  */
object OptimizationProblems {


    /** @return list of OptimizationProblems in dimension dim with known solution as follows:
      * first the following unconstrained problems
      *     minX1,
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
        for(j <- 1 to 3){

            val q = 1.0+rand()
            val m = dim-1        // rank of A, so solution space = ker(A) is one dimensional
            val id = "Random power problem in dimension "+dim+" with m="+dim+"-1 and exponent 2*"+q
            theList = theList :+ randomPowerProblem(id,dim,m,q,pars)
        }
        minX1(pars) :: theList
    }

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


    /** Objective function f(x0,x1)=x0 subject to x1>=exp(x0) and x1=a+b*x0 with constant
      * r=0.5*(e+1/e), k=0.5*(e-1/e) chosen so that the line x1=r+k*x0 intersects x1=exp(x0)
      * at the points x0=1,-1. The minimum is thus attained at x0=-1, x1=r-k=1/e.
      *
      * @param pars parameters controlling the solver behaviour (maxIter, backtracking line search
      * parameters etc, see [SolverParams].
      */
    def minX1(pars:SolverParams):OptimizationProblem with KnownMinimizer = {

        val dim = 2

        // objective f(x0,x1)=x0
        val objF = new ObjectiveFunction(dim){

            def valueAt(x:DenseVector[Double]) = x(0)
            def gradientAt(x:DenseVector[Double]) = DenseVector(1.0,0.0)
            def hessianAt(x:DenseVector[Double]) = DenseMatrix.zeros[Double](dim,dim)
        }

        // set of inequality constraints

        // constraint x1 >= exp(x0)
        val ub = 0.0 // upper bound
        val ct1 = new Constraint("x2>=exp(x1)",dim,ub){

            def valueAt(x:DenseVector[Double]) = Math.exp(x(0))-x(1)
            def gradientAt(x:DenseVector[Double]) = DenseVector(Math.exp(x(0)),-1.0)
            def hessianAt(x:DenseVector[Double]) = DenseMatrix((Math.exp(x(0)),0.0),(0.0,0.0))
        }
        // linear inequality x1 <= r+k*x0
        val e = Math.exp(1.0); val r = 0.5*(e+1/e); val k = 0.5*(e-1/e)
        val a = DenseVector(-k,1.0)    // a dot x = x1-k*x0
        val ct2 = LinearConstraint("x1<=r+k*x0",dim,r,0.0,a)

        val x = DenseVector(0.0,0.0)     // point where all the constraints are defined
        val ineqs = ConstraintSet(dim,List(ct1,ct2),x)   // the inequality constraints

        // add a feasible point
        val x_feas = DenseVector(0.0,1.01)
        val ineqsF = ineqs.addFeasiblePoint(x_feas)

        val id = "f(x0,x1)=x0 on x1>=exp(x0), x1 <= r+k*x0, with feasible point."
        val doSOIAnalysis = false

        // val problem = OptimizationProblem.withBarrierMethod(id,dim,objF,ineqs,doSOIAnalysis,pars)
        val problem = OptimizationProblem.withBarrierMethod(id,dim,objF,ineqsF,pars)

        // add the known solution
        val x_opt = DenseVector(-1.0,1/e)    // minimizer
        val y_opt = -1.0                     // minimum value
        val minimizer = KnownMinimizer(x_opt,y_opt)
        OptimizationProblem.addSolution(problem,minimizer)
    }
}
