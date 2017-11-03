package cvx

import breeze.linalg.{DenseVector, _}


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
    * @param dim common dimension of all problems, must be >= 2.
    * @param pars parameters controlling the solver behaviour (maxIter, backtracking line search
    * parameters etc, see [SolverParams].
    */
  def standardProblems(dim:Int,pars:SolverParams,debugLevel:Int):List[OptimizationProblem with KnownMinimizer] = {

    var theList:List[OptimizationProblem with KnownMinimizer] = List(normSquared(dim,debugLevel))
    for(j <- 1 to 3){

      val q = 1.0+rand()
      val m = dim-1        // rank of A, so solution space = ker(A) is one dimensional
      val id = "Random power problem in dimension "+dim+" with m="+dim+"-1 and exponent 2*"+q
      theList = theList :+ randomPowerProblem(id,dim,m,q,pars,debugLevel)
    }
    minX1_FP(pars,debugLevel) :: theList
  }

  /** f(x) = (1/2)*(x dot x).*/
  def normSquared(dim:Int,C:ConvexSet,debugLevel:Int):OptimizationProblem = {

    assert(C.dim==dim)

    val id = "f(x) = 0.5*||x||^2  in dimension "+dim
    if(debugLevel>0) {
      println("\n\nAllocating problem " + id)
      Console.flush()
    }
    val logFilePath = "logs/"+id+"_log.txt"
    val logger = Logger(logFilePath)

    val startingPoint = DenseVector.tabulate[Double](dim)(j=>1+j)
    val objF = ObjectiveFunctions.normSquared(dim)
    val maxIter = 200; val alpha = 0.1; val beta = 0.5; val tol = 1e-8; val delta = 1e-8
    val pars = SolverParams(maxIter,alpha,beta,tol,delta)

    OptimizationProblem(id,objF,startingPoint,C,pars,logger)
  }

  /** f(x) = (1/2)*(x dot x) on the full Euclidean Space*/
  def normSquared(dim:Int,debugLevel:Int):OptimizationProblem with KnownMinimizer = {

    val minimizer = KnownMinimizer(DenseVector.zeros[Double](dim),ObjectiveFunctions.normSquared(dim))
    val problem = normSquared(dim,ConvexSet.fullSpace(dim),debugLevel)
    problem.addSolution(minimizer)
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
  def powerProblem( id:String,
                    A:DenseMatrix[Double], alpha:DenseVector[Double], q:Double,
                    pars:SolverParams, debugLevel:Int
    ): OptimizationProblem with KnownMinimizer = {

    if(debugLevel>0){
      println("\n\nAllocating problem "+id)
      Console.flush()
    }
    val n=A.cols; val m=A.rows
    assert(m<=n)

    val logFilePath = "logs/"+id+"_log.txt"
    val logger = Logger(logFilePath)

    val startingPoint = DenseVector.tabulate[Double](n)(j=>j*Math.sqrt(n))
    val objF:ObjectiveFunction = Type1Function.powerTestFunction(A,alpha,q)
    val C = ConvexSet.fullSpace(n)
    val minimizer = new KnownMinimizer {

      def isMinimizer(x:DenseVector[Double],tol:Double):Boolean = norm(A*x)<tol
      def minimumValue:Double = 0.0
    }
    val problem = OptimizationProblem(id,objF,startingPoint,C,pars,logger)
    problem.addSolution(minimizer)
  }

  /** [powerProblem] in dimension dim with m x dim matrix A and coefficient vector alpha
    * having random entries in (0,1). In addition 1.0 is added to the diagonal entries of
    * A to improve the condition number.
    *
    * @param m we must have m<=dim.
    */
  def randomPowerProblem(id:String,dim:Int,m:Int,q:Double,pars:SolverParams,debugLevel:Int):
  OptimizationProblem with KnownMinimizer = {

    assert(m<=dim)
    val A = DenseMatrix.rand[Double](m,dim)
    for(i <- 0 until m) A(i,i)+=1.0
    val alpha = DenseVector.rand[Double](m)
    powerProblem(id,A,alpha,q,pars,debugLevel)
  }


  /** Objective function f(x0,x1)=x0 subject to x1>=exp(x0) and x1=a+b*x0 with constant
    * r=0.5*(e+1/e), k=0.5*(e-1/e) chosen so that the line x1=r+k*x0 intersects x1=exp(x0)
    * at the points x0=1,-1. The minimum is thus attained at x0=-1, x1=r-k=1/e.
    *
    * The constraint set of the problem is allocated with feasible point.
    *
    * @param pars parameters controlling the solver behaviour (maxIter, backtracking line search
    * parameters etc, see [SolverParams].
    */
  def minX1_FP(pars: SolverParams,debugLevel:Int):OptimizationProblem with KnownMinimizer = {

    val id = "f(x0,x1)=x0 on x1>=exp(x0), x1 <= r+k*x0, with feasible point."
    if(debugLevel>0) {
      println("\n\nAllocating problem " + id)
      Console.flush()
    }
    val logFilePath = "logs/"+id+"_log.txt"
    val logger = Logger(logFilePath)

    val dim = 2
    // objective f(x0,x1)=x0
    val objF = new ObjectiveFunction(dim){

      def valueAt(x:DenseVector[Double]):Double = x(0)
      def gradientAt(x:DenseVector[Double]):DenseVector[Double] = DenseVector(1.0,0.0)
      def hessianAt(x:DenseVector[Double]):DenseMatrix[Double] = DenseMatrix.zeros[Double](dim,dim)
    }

    // set of inequality constraints

    // constraint x1 >= exp(x0)
    val ub = 0.0 // upper bound
    val ct1 = new Constraint("x2>=exp(x1)",dim,ub){

      def valueAt(x:DenseVector[Double]):Double = Math.exp(x(0))-x(1)
      def gradientAt(x:DenseVector[Double]):DenseVector[Double] = DenseVector(Math.exp(x(0)),-1.0)
      def hessianAt(x:DenseVector[Double]):DenseMatrix[Double] = DenseMatrix((Math.exp(x(0)),0.0),(0.0,0.0))
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

    val doSOIAnalysis = false

    // None: no equality constraints
    val problem = OptimizationProblem(id,objF,ineqsF,None,pars,logger,debugLevel)

    // add the known solution
    val x_opt = DenseVector(-1.0,1/e)    // minimizer
    val minimizer = KnownMinimizer(x_opt,objF)
    problem.addSolution(minimizer)
  }

  /** Objective function f(x0,x1)=x0 subject to x1>=exp(x0) and x1=a+b*x0 with constant
    * r=0.5*(e+1/e), k=0.5*(e-1/e) chosen so that the line x1=r+k*x0 intersects x1=exp(x0)
    * at the points x0=1,-1. The minimum is thus attained at x0=-1, x1=r-k=1/e.
    *
    * The constraint set of the problem is allocated without feasible point.
    *
    * @param pars parameters controlling the solver behaviour (maxIter, backtracking line search
    * parameters etc, see [SolverParams].
    */
  def minX1_no_FP(pars: SolverParams, debugLevel:Int):OptimizationProblem with KnownMinimizer = {

    val id = "f(x0,x1)=x0 on x1>=exp(x0), x1 <= r+k*x0, no feasible point"
    if(debugLevel>0) {
      println("\n\nAllocating problem " + id)
      Console.flush()
    }
    val logFilePath = "logs/"+id+"_log.txt"
    val logger = Logger(logFilePath)

    val dim = 2
    // objective f(x0,x1)=x0
    val objF = new ObjectiveFunction(dim){

      def valueAt(x:DenseVector[Double]):Double = x(0)
      def gradientAt(x:DenseVector[Double]):DenseVector[Double] = DenseVector(1.0,0.0)
      def hessianAt(x:DenseVector[Double]):DenseMatrix[Double] = DenseMatrix.zeros[Double](dim,dim)
    }

    // set of inequality constraints

    // constraint x1 >= exp(x0)
    val ub = 0.0 // upper bound
    val ct1 = new Constraint("x2>=exp(x1)",dim,ub){

      def valueAt(x:DenseVector[Double]):Double = Math.exp(x(0))-x(1)
      def gradientAt(x:DenseVector[Double]):DenseVector[Double] = DenseVector(Math.exp(x(0)),-1.0)
      def hessianAt(x:DenseVector[Double]):DenseMatrix[Double] = DenseMatrix((Math.exp(x(0)),0.0),(0.0,0.0))
    }
    // linear inequality x1 <= r+k*x0
    val e = Math.exp(1.0); val r = 0.5*(e+1/e); val k = 0.5*(e-1/e)
    val a = DenseVector(-k,1.0)    // a dot x = x1-k*x0
    val ct2 = LinearConstraint("x1<=r+k*x0",dim,r,0.0,a)

    val x = DenseVector(0.0,0.0)     // point where all the constraints are defined
    val ineqs = ConstraintSet(dim,List(ct1,ct2),x)   // the inequality constraints


    val problem = OptimizationProblem(id,objF,ineqs,None,pars,logger,debugLevel)

    // add the known solution
    val x_opt = DenseVector(-1.0,1/e)    // minimizer
    val minimizer = KnownMinimizer(x_opt,objF)
    problem.addSolution(minimizer)
  }

  /**********************************************************************************/
  /** Minimization of Kullback-Leibler distance from discrete uniform distribution **/
  /**********************************************************************************/

  /** A feasible problem with known analytic solution.
    * Minimize the Kullback-Leibler distance d_KL(x,p) from the uniform
    * distribution p_j=1/n on the set Omega={0,1,2,...,n-1} subject to the
    * constraints that
    *                     P^x(A)\geq 0.36 and P^x(B)\leq 0.1
    * where A={0,1,2} and B={n/2,n/2+1,...,n-1}.
    *
    * From manual optimization using symmetry and heuristics the optimum is given
    * as follows:
    *
    * IF 1.8/n>=0.12:
    *
    * x_j=1.8/n, j=0,1,2
    * x_j=0.2/n,       j=n/2,n/2+1,...,n-1
    * x_j=(1.8*n-10.8)/n(n-6), all other j
    *
    * IF 1.8/n <= 0.12:
    *
    * x_j=0.12, j=0,1,2
    * x_j=0.2/n,       j=n/2,n/2+1,...,n-1
    * x_j=1.08/(n-6), all other j
    *
    *
    * @param n must be even and bigger than 9 (to ensure feasibility).
    */
  def kl_1(n:Int, pars: SolverParams,debugLevel:Int):OptimizationProblem with KnownMinimizer = {

    assert(n>9 && n%2==0, "\n\nn must be even and > 9, but n = "+n+"\n\n")

    val id = "dist_KL problem 1"
    if(debugLevel>0) {
      println("\n\nAllocating problem " + id)
      Console.flush()
    }
    val logFilePath = "logs/"+id+"_log.txt"
    val logger = Logger(logFilePath)

    // objective f(x0,x1)=x0
    val objF = Dist_KL(n)

    // set up the constraints
    val positivityCnts:List[Constraint] = Constraints.allCoordinatesPositive(n)


    // indicator function 1_A
    val I_A = DenseVector.tabulate[Double](n)(j => if(j<3) 1.0 else 0.0)
    val ct1 = Constraints.expectation_lt_r(-I_A,-0.36,"P(A)>=0.36")

    // indicator function 1_B
    val I_B = DenseVector.tabulate[Double](n)(j => if(j>=n/2) 1.0 else 0.0)
    val ct2 = Constraints.expectation_lt_r(I_B,0.1,"P(B)<=0.1")

    val constraints:Seq[Constraint] = positivityCnts:::List[Constraint](ct1,ct2)

    // point where all constraints are defined
    val x = DenseVector.tabulate[Double](n)(j=>1.0/n)
    val ineqs = ConstraintSet(n,constraints,x)

    val probEq:EqualityConstraint = Constraints.sumToOne(n)
    val problem = OptimizationProblem(id,objF,ineqs,Some(probEq),pars,logger,debugLevel)

    // the heuristic optimal solution
    val x_opt = if(1.8/n>0.12) DenseVector.tabulate[Double](n)(
      j => if(j<3) 1.8/n else if (j>=n/2) 0.2/n else (1.8*n-10.8)/(n*(n-6))
    ) else DenseVector.tabulate[Double](n)(
      j => if(j<3) 0.12 else if (j>=n/2) 0.2/n else 1.08/(n-6)
    )
    val minimizer = KnownMinimizer(x_opt,objF)
    problem.addSolution(minimizer)
  }


  /** A feasible problem with known analytic solution.
    * Minimize the Kullback-Leibler distance d_KL(x,p) from the uniform
    * distribution p_j=1/n on the set Omega={0,1,2,...,n-1} subject to the
    * equality constraints
    *                     P^x(A)=0.36 and P^x(B)=0.1
    * where A={0,1,2} and B={n/2,n/2+1,...,n-1}.
    *
    * Using symmetry and it can be shown that the optimum occurs at the follwing
    * probability distribution x (see docs/Dist_KL.pdf):
    * IF 1.8/n>=0.12:
    *
    * x_j=0.36/3, j=0,1,2
    * x_j=0.2/n,       j=n/2,n/2+1,...,n-1
    * x_j=(1-0.36-0.1)/(n-n/2-3), all other j
    *
    * @param n must be even and bigger than 9 (to ensure feasibility).
    */
  def kl_2(n:Int, pars: SolverParams,debugLevel:Int):OptimizationProblem with KnownMinimizer = {

    assert(n>9 && n%2==0, "\n\nn must be even and > 9, but n = "+n+"\n\n")

    val id = "dist_KL problem 2"
    if(debugLevel>0) {
      println("\n\nAllocating problem " + id)
      Console.flush()
    }
    val logFilePath = "logs/"+id+"_log.txt"
    val logger = Logger(logFilePath)
    val probEq:EqualityConstraint = Constraints.sumToOne(n)

    // indicator function 1_A
    val I_A = DenseVector.tabulate[Double](n)(j => if(j<3) 1.0 else 0.0)
    val e1:EqualityConstraint = Constraints.expectation_eq_r(I_A,0.36)

    // indicator function 1_B
    val I_B = DenseVector.tabulate[Double](n)(j => if(j>=n/2) 1.0 else 0.0)
    val e2:EqualityConstraint = Constraints.expectation_eq_r(I_B,0.1)

    // add these to the basic equality constraint eqs
    val e3 = probEq.addEqualities(e1)
    val eqs = e3.addEqualities(e2)

    if(debugLevel>1) eqs.printSelf(logger,3)

    // KL-distance
    val objF = Dist_KL(n)

    // set up the constraints
    val positivityCnts:List[Constraint] = Constraints.allCoordinatesPositive(n)

    // point where all constraints are defined
    val x = DenseVector.tabulate[Double](n)(j=>1.0/n)
    val ineqs = ConstraintSet(n,positivityCnts,x)

    val problem = OptimizationProblem(id,objF,ineqs,Some(eqs),pars,logger,debugLevel)

    // the known optimal solution
    val x_opt = DenseVector.tabulate[Double](n)(
      j => if(j<3) 0.12 else if (j>=n/2) 0.2/n else 1.08/(n-6)
    )
    val minimizer = KnownMinimizer(x_opt,objF)
    problem.addSolution(minimizer)
  }


  /** A infeasible problem: sum of probabilities of disjoint events bigger than one.
    *
    * @param n must be even and bigger than 9 (to ensure feasibility).
    */
  def infeasible_kl_1(n:Int, pars: SolverParams,debugLevel:Int):OptimizationProblem = {

    assert(n>9 && n%2==0, "\n\nn must be even and > 9, but n = "+n+"\n\n")

    val id = "dist_KL problem 3 (infeasible)"
    if(debugLevel>0) {
      println("\n\nAllocating problem " + id)
      Console.flush()
    }
    val logFilePath = "logs/"+id+"_log.txt"
    val logger = Logger(logFilePath)

    val probEq:EqualityConstraint = Constraints.sumToOne(n)
    val I_A = DenseVector.tabulate[Double](n)(j => if(j<3) 1.0 else 0.0)
    val I_B = DenseVector.tabulate[Double](n)(j => if(j>=n/2) 1.0 else 0.0)
    val sgnA = -1.0; val sgnB = -1.0
    val pA=0.51; val pB=0.51
    val ineqs = ConstraintSets.probAB(n,I_A,pA,sgnA,I_B,pB,sgnB)   // p(A)>=0.51, p(B)>=0.51
    // KL-distance
    val objF = Dist_KL(n)

    OptimizationProblem(id,objF,ineqs,Some(probEq),pars,logger,debugLevel)
  }


}