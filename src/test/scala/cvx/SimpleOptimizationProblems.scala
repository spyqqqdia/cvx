package cvx

import breeze.linalg.{DenseMatrix, DenseVector, rand}
import breeze.numerics.{abs, sqrt}
import cvx.OptimizationProblems.{normSquared, randomPowerProblem}

/**
  * Created by oar on 14.12.17.
  */
object SimpleOptimizationProblems {

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
      println("Allocating problem " + id)
      Console.flush()
    }
    val logFilePath = "logs/"+id+"_log.txt"
    val logger = Logger(logFilePath)

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
      println("Allocating problem " + id)
      Console.flush()
    }
    val logFilePath = "logs/"+id+"_log.txt"
    val logger = Logger(logFilePath)

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


    val problem = OptimizationProblem(id,objF,ineqs,None,pars,logger,debugLevel)

    // add the known solution
    val x_opt = DenseVector(-1.0,1/e)    // minimizer
    val minimizer = KnownMinimizer(x_opt,objF)
    problem.addSolution(minimizer)
  }

  /** This is the problem
    *      min -a'x  subject to  |x_j|<=|a_j|.
    * Obviously the minimum is assumed at x=a.
    */
  def minDotProduct(a:DenseVector[Double], pars: SolverParams, debugLevel:Int):
  OptimizationProblem with KnownMinimizer = {

    val id = "Min f(x)=a'x subject to |x_j|<=|a_j|"
    val logFilePath = "logs/"+id+"_log.txt"
    val logger = Logger(logFilePath)

    if(debugLevel>0) {
      println("Allocating problem " + id)
      Console.flush()
    }

    val n = a.length
    val ub = abs(a)
    val cntList = Constraints.absoluteValuesBoundedBy(n,ub)
    val x0 = a*2.0   // point where all constraints are defined, deliberately infeasible
    val cnts = ConstraintSet(n,cntList,x0)

    val objF = LinearObjectiveFunction(-a)

    val problem = OptimizationProblem(id,objF,cnts,None,pars,logger,debugLevel)
    val theMinimizer = KnownMinimizer(a,objF)
    problem.addSolution(theMinimizer)
  }

  /** This is the problem
    *      min ||x||_p  subject to ||x||_1=1, x_j>=0.
    * where p>1. We know that generally ||x||_1 <= ||x||_p with equality
    * if an only if all |x_j| are equal.
    * Thus our problem has the unique solution x_j=1/n with n=length(x).
    */
  def min_pNorm(dim:Int, p:Double, pars: SolverParams, debugLevel:Int):
  OptimizationProblem with KnownMinimizer = {

    assert(p>=2,"\np-norm not twice differentiable unless p>=2 but p="+p+"\n")

    val id = "Min f(x)=||x||_p subject to ||x||_1=1"
    val logFilePath = "logs/"+id+"_log.txt"
    val logger = Logger(logFilePath)

    if(debugLevel>0) {
      println("Allocating problem " + id)
      Console.flush()
    }

    // the constraints x_j>=0, sum x_j=1
    val positivityCnts:List[Constraint] = Constraints.allCoordinatesPositive(dim)
    val probEq:EqualityConstraint = Constraints.sumToOne(dim)

    val x0 = DenseVector.zeros[Double](dim)     // vector where all constraints are defined
    val cnts = ConstraintSet(dim,positivityCnts,x0)
    val objF = ObjectiveFunctions.p_norm_p(dim,p)
    val problem = OptimizationProblem(id,objF,cnts,Some(probEq),pars,logger,debugLevel)

    // minimizer is the vector x_j=1/dim
    val w = DenseVector.tabulate[Double](dim)(j => 1.0/dim)
    problem.addSolution(KnownMinimizer(w,objF))
  }

  /** A problem with rank one Hessian
    * f(x)=(a_1x_1+...+a_nx_n)² = x'(aa')x subject to x_j>=0, sum(x_j)=1.
    * Solution x = e_j, where a_j = m:= min a_i.
    * If the minimum m is not uniquely determined there is a continuum
    * of solutions = convex_hull(e_j:a_j=m).
    *
    * We do this for a=linspace(1,10), so the unique solution is x=e_1.
    */
  def rankOneProblemSimplex(dim:Int, pars: SolverParams, debugLevel:Int):
  OptimizationProblem with KnownMinimizer = {

    assert(dim>1,"\n\ndim must be >1 , but dim = "+dim+"\n\n")

    val id = "Simplicial rankOneProblem"
    if(debugLevel>0) {
      println("Allocating problem " + id)
      Console.flush()
    }
    val logFilePath = "logs/"+id+"_log.txt"
    val logger = Logger(logFilePath)

    val q = DenseVector.zeros[Double](dim)
    val a:DenseVector[Double] = breeze.linalg.linspace(1,2,dim)
    val P:DenseMatrix[Double] = a*a.t
    val objF = QuadraticObjectiveFunction(dim,0.0,q,P)

    val positivityCnts:List[Constraint] = Constraints.allCoordinatesPositive(dim)
    // point where all constraints are defined
    val x = DenseVector.tabulate[Double](dim)(j=>1.0/dim)
    val ineqs = ConstraintSet(dim,positivityCnts,x)

    val probEq:EqualityConstraint = Constraints.sumToOne(dim)

    val problem = OptimizationProblem(id,objF,ineqs,Some(probEq),pars,logger,debugLevel)

    // the known optimal solution
    val x_opt = DenseVector.tabulate[Double](dim)(j => if(j==0) 1.0 else 0.0)
    val minimizer = KnownMinimizer(x_opt,objF)
    problem.addSolution(minimizer)
  }
  /** A problem with rank one Hessian
    * f(x)=(a_1x_1+...+a_nx_n)² = x'(aa')x subject to x_j>=0, ||x||²<=1.
    * Solution: all x orthogonal to a satisfying he constraints.
    * If this maximum is not uniquely determined there is a continuum
    * of solutions = convex_hull(e_j:a_j=m)
    *
    * We do this for a=linspace(1,10), so the unique solution is x=0.
    */
  def rankOneProblemSphere(dim:Int, pars: SolverParams, debugLevel:Int):
  OptimizationProblem with KnownMinimizer = {

    assert(dim>1,"\n\ndim must be >1 , but dim = "+dim+"\n\n")

    val id = "Spherical rankOneProblem"
    if(debugLevel>0) {
      println("Allocating problem " + id)
      Console.flush()
    }
    val logFilePath = "logs/"+id+"_log.txt"
    val logger = Logger(logFilePath)

    val q = DenseVector.zeros[Double](dim)
    val a:DenseVector[Double] = breeze.linalg.linspace(1,2,dim)
    val P:DenseMatrix[Double] = a*a.t
    val objF = QuadraticObjectiveFunction(dim,0.0,q,P)

    val cntSphere = Constraints.oneHalfNorm2BoundedBy(dim,1.0/2)
    val cnts:List[Constraint] = cntSphere::Constraints.allCoordinatesPositive(dim)

    // point where all constraints are defined
    val x = DenseVector.tabulate[Double](dim)(j=>1.0/dim)
    val ineqs = ConstraintSet(dim,cnts,x)

    val problem = OptimizationProblem(id,objF,ineqs,None,pars,logger,debugLevel)

    // the known optimal solution
    val x_opt = DenseVector.zeros[Double](dim)
    val minimizer = KnownMinimizer(x_opt,objF)
    problem.addSolution(minimizer)
  }

  /** This problem has many free variables which need to be eliminated
    * in phase I analysis:
    * f(x)=0.5*||x||² subject to x_0<=-1.
    */
  def normSquaredWithFreeVariables(dim:Int, pars: SolverParams, debugLevel:Int):
  OptimizationProblem with KnownMinimizer = {

    assert(dim>1,"\n\ndim must be >1 , but dim = "+dim+"\n\n")

    val id = "normSquaredWithFreeVariables"
    if(debugLevel>0) {
      println("Allocating problem " + id)
      Console.flush()
    }
    val logFilePath = "logs/"+id+"_log.txt"
    val logger = Logger(logFilePath)

    val objF = ObjectiveFunctions.normSquared(dim)

    val a = DenseVector.tabulate[Double](dim)(j => if(j==0) 1.0 else 0.0)
    val cnt = LinearConstraint("x_1<=-1",dim,-1.0,0.0,a)

    // point where all constraints are defined
    val x = DenseVector.fill[Double](dim)(-10.0)
    val ineqs = ConstraintSet(dim,List(cnt),x)

    val problem = OptimizationProblem(id,objF,ineqs,None,pars,logger,debugLevel)

    // the known optimal solution
    val x_opt = DenseVector.zeros[Double](dim)
    x_opt(0) = -1.0
    val minimizer = KnownMinimizer(x_opt,objF)
    problem.addSolution(minimizer)
  }

  /** f(x)= sum(x) subject to ||x||²<=1.
    * Solution is all x_j=-1/sqrt(dim).
    */
  def joptP1(dim:Int, pars: SolverParams, debugLevel:Int):
  OptimizationProblem with KnownMinimizer = {

    assert(dim>1,"\n\ndim must be >1 , but dim = "+dim+"\n\n")

    val id = "f(x)=sum(x) with ||x||²<=1"
    if(debugLevel>0) {
      println("Allocating problem " + id)
      Console.flush()
    }
    val logFilePath = "logs/"+id+"_log.txt"
    val logger = Logger(logFilePath)

    val a = DenseVector.fill[Double](dim)(1.0)
    val objF = LinearObjectiveFunction(dim,0.0,a)

    val cnt = Constraints.oneHalfNorm2BoundedBy(dim,1.0/2)
    // point where all constraints are defined
    val x = DenseVector.fill[Double](dim)(2.0)    // infeasible
    val ineqs = ConstraintSet(dim,List(cnt),x)

    val problem = OptimizationProblem(id,objF,ineqs,None,pars,logger,debugLevel)

    // the known optimal solution
    val x_opt = -a*1.0/sqrt(dim)
    val minimizer = KnownMinimizer(x_opt,objF)
    problem.addSolution(minimizer)
  }

  /** See docs/OptimizerExamples.pdf, example 1.5
    * f(x)=x'Px subject to x_j>=0, sum(x)=1 in dimension 2.
    */
  def joptP2(pars: SolverParams, debugLevel:Int):
  OptimizationProblem with KnownMinimizer = {

    val id = "example_1.5"
    if(debugLevel>0) {
      println("Allocating problem " + id)
      Console.flush()
    }
    val logFilePath = "logs/"+id+"_log.txt"
    val logger = Logger(logFilePath)

    val dim=2
    val a = DenseVector.zeros[Double](dim)
    val P = DenseMatrix((1.0,0.4),(0.4,1.0))
    val objF = QuadraticObjectiveFunction(dim,0.0,a,P)

    val probEq:EqualityConstraint = Constraints.sumToOne(dim)
    val cnt = Constraints.allCoordinatesPositive(dim)
    // point where all constraints are defined
    val x = DenseVector.fill[Double](dim)(2.0)    // infeasible
    val ineqs = ConstraintSet(dim,cnt,x)

    val problem = OptimizationProblem(id,objF,ineqs,Some(probEq),pars,logger,debugLevel)

    // the known optimal solution
    val x_opt = DenseVector(0.5,0.5)
    val minimizer = KnownMinimizer(x_opt,objF)
    problem.addSolution(minimizer)
  }






  /** @return list of OptimizationProblems in dimension dim with known solution as follows:
    * first the following unconstrained problems
    *     minX1,
    *     f(x) = x dot x, followed by
    *     3 [randomPowerProblem]s with one dimensional solution space (m = dim-1)
    *     followed by minDotProduct followed by min_pNorm with p=2.5 and p=5.
    *
    * This list will be expanded continually.
    * @param dim common dimension of all problems, must be >= 2.
    * @param condNumber: condition number of the matrix A in the power problems.
    * @param pars parameters controlling the solver behaviour (maxIter, backtracking line search
    * parameters etc, see [SolverParams].
    */
  def standardProblems(dim:Int, condNumber:Double, pars:SolverParams, debugLevel:Int):
  List[OptimizationProblem with KnownMinimizer] = {

    var theList:List[OptimizationProblem with KnownMinimizer]
      = List(normSquared(dim,debugLevel))
    for(j <- 1 to 3){

      val q = 2.0+rand()   // needs to be >= 2 for differentiability
      val id = "Random power problem in dimension "+dim+" with m="+dim+" and exponent "+q
      val p = randomPowerProblem(id,dim,dimKernel=0,condNumber,q,pars,debugLevel)
      theList = theList :+ p
    }
    val problem1 = minX1_no_FP(pars,debugLevel)
    val a = DenseVector.tabulate[Double](dim)(i => 1.0)
    val problem2 = minDotProduct(a,pars,debugLevel)
    val problem3 = min_pNorm(dim,p=2.2,pars,debugLevel)
    val problem4 = min_pNorm(dim,p=4,pars,debugLevel)
    val problem5 = rankOneProblemSimplex(dim,pars,debugLevel)
    val problem6 = rankOneProblemSphere(dim,pars,debugLevel)
    val problem7 = joptP1(dim,pars,debugLevel)
    //val problem8 = joptP2(pars,debugLevel)
    //val problem9 = normSquaredWithFreeVariables(dim,pars,debugLevel)
    problem1 :: problem2 :: problem3 :: problem4 ::
      problem5 :: problem6 :: problem7 :: theList
  }




}
