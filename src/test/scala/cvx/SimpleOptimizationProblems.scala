package cvx

import breeze.linalg.{DenseMatrix, DenseVector, rand}
import breeze.numerics.{abs, sqrt}
import cvx.OptimizationProblems.{normSquared, randomPowerProblem}

import scala.collection.mutable.ListBuffer

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
    * @param solverType: "BR" (barrier solver), "PD" (primal dual solver).
    */
  def minX1_withFP(solverType:String,debugLevel:Int):OptimizationProblem with KnownMinimizer = {

    val id = "f(x0,x1)=x0 on x1>=exp(x0), x1 <= r+k*x0, with feasible point."
    if(debugLevel>0) {
      println("\nAllocating problem " + id)
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

      def isDefinedAt(u:DenseVector[Double]):Boolean = true
      def valueAt(x:DenseVector[Double]):Double = Math.exp(x(0))-x(1)
      def gradientAt(x:DenseVector[Double]):DenseVector[Double] = DenseVector(Math.exp(x(0)),-1.0)
      def hessianAt(x:DenseVector[Double]):DenseMatrix[Double] = DenseMatrix((Math.exp(x(0)),0.0),(0.0,0.0))
    }
    // linear inequality x1 <= r+k*x0
    val e = Math.exp(1.0); val r = 0.5*(e+1/e); val k = 0.5*(e-1/e)
    val a = DenseVector(-k,1.0)    // a dot x = x1-k*x0
    val ct2 = LinearConstraint("x1<=r+k*x0",dim,r,0.0,a)

    val x0 = DenseVector(0.0,0.0)     // point where all the constraints are defined
    val setWhereDefined = ConvexSets.wholeSpace(dim)
    val ineqs = ConstraintSet(dim,List(ct1,ct2),setWhereDefined,x0)   // the inequality constraints

    // add a feasible point
    val x_feas = DenseVector(0.0,1.01)
    val ineqsF = ineqs.addFeasiblePoint(x_feas)

    val doSOIAnalysis = false

    // None: no equality constraints
    val pars = SolverParams.standardParams
    val problem = OptimizationProblem(
      id,setWhereDefined,objF,ineqsF,None,solverType,pars,logger,debugLevel
    )

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
    * @param solverType: "BR" (barrier solver), "PD" (primal dual solver).
    */
  def minX1_no_FP(solverType:String,debugLevel:Int):OptimizationProblem with KnownMinimizer = {

    val id = "f(x0,x1)=x0 on x1>=exp(x0), x1 <= r+k*x0, no feasible point"
    if(debugLevel>0) {
      println("\nAllocating problem " + id)
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

      def isDefinedAt(u:DenseVector[Double]):Boolean = true
      def valueAt(x:DenseVector[Double]):Double = Math.exp(x(0))-x(1)
      def gradientAt(x:DenseVector[Double]):DenseVector[Double] = DenseVector(Math.exp(x(0)),-1.0)
      def hessianAt(x:DenseVector[Double]):DenseMatrix[Double] = DenseMatrix((Math.exp(x(0)),0.0),(0.0,0.0))
    }
    // linear inequality x1 <= r+k*x0
    val e = Math.exp(1.0); val r = 0.5*(e+1/e); val k = 0.5*(e-1/e)
    val a = DenseVector(-k,1.0)    // a dot x = x1-k*x0
    val ct2 = LinearConstraint("x1<=r+k*x0",dim,r,0.0,a)

    val x0 = DenseVector(0.0,0.0)     // point where all the constraints are defined
    val setWhereDefined = ConvexSets.wholeSpace(dim)
    val ineqs = ConstraintSet(dim,List(ct1,ct2),setWhereDefined,x0)   // the inequality constraints

    val pars = SolverParams.standardParams
    val problem = OptimizationProblem.withoutFeasiblePoint(
      id,setWhereDefined,objF,ineqs,None,solverType,pars,logger,debugLevel
    )
    // add the known solution
    val x_opt = DenseVector(-1.0,1/e)    // minimizer
    val minimizer = KnownMinimizer(x_opt,objF)
    problem.addSolution(minimizer)
  }

  /** This is the problem
    *      min -a'x  subject to  |x_j|<=|a_j|.
    * Obviously the minimum is assumed at x=a.
    *
    * @param solverType: "BR" (barrier solver), "PD" (primal dual solver).
    */
  def minDotProduct(a:DenseVector[Double], solverType:String, debugLevel:Int):
  OptimizationProblem with KnownMinimizer = {

    val id = "Min f(x)=a'x subject to |x_j|<=|a_j|"
    val logFilePath = "logs/"+id+"_log.txt"
    val logger = Logger(logFilePath)

    if(debugLevel>0) {
      println("\nAllocating problem " + id)
      Console.flush()
    }

    val n = a.length  // the dimension
    val ub = abs(a)
    val cntList = Constraints.absoluteValuesBoundedBy(n,ub)
    val x0 = a*2.0   // point where all constraints are defined, deliberately infeasible
    val setWhereDefined = ConvexSets.wholeSpace(n)
    val ineqs = ConstraintSet(n,cntList,setWhereDefined,x0)

    val objF = LinearObjectiveFunction(-a)

    val pars = SolverParams.standardParams
    val problem = OptimizationProblem.withoutFeasiblePoint(
      id,setWhereDefined,objF,ineqs,None,solverType,pars,logger,debugLevel
    )
    val theMinimizer = KnownMinimizer(a,objF)
    problem.addSolution(theMinimizer)
  }

  /** This is the problem
    *      min ||x||_p  subject to ||x||_1=1, x_j>=0.
    * where p>1. We know that generally ||x||_1 <= ||x||_p with equality
    * if an only if all |x_j| are equal.
    * Thus our problem has the unique solution x_j=1/n with n=length(x).
    *
    * @param solverType: "BR" (barrier solver), "PD" (primal dual solver).
    */
  def min_pNorm(dim:Int, p:Double, solverType:String, debugLevel:Int):
  OptimizationProblem with KnownMinimizer = {

    assert(p>=2,"\np-norm not twice differentiable unless p>=2 but p="+p+"\n")

    val id = "Min f(x)=||x||_"+p+" subject to ||x||_1=1, x_j>=0"
    val logFilePath = "logs/"+id+"_log.txt"
    val logger = Logger(logFilePath)

    if(debugLevel>0) {
      println("\nAllocating problem " + id)
      Console.flush()
    }

    // the constraints x_j>=0, sum x_j=1
    val positivityCnts:List[Constraint] = Constraints.allCoordinatesPositive(dim)
    val probEq:EqualityConstraint = Constraints.sumToOne(dim)

    val x0 = DenseVector.zeros[Double](dim)     // vector where all constraints are defined
    val setWhereDefined = ConvexSets.wholeSpace(dim)
    val ineqs = ConstraintSet(dim,positivityCnts,setWhereDefined,x0)
    val objF = ObjectiveFunctions.p_norm_p(dim,p)

    val pars = SolverParams.standardParams
    val problem = OptimizationProblem.withoutFeasiblePoint(
      id,setWhereDefined,objF,ineqs,Some(probEq),solverType,pars,logger,debugLevel
    )
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
    *
    * @param solverType: "BR" (barrier solver), "PD" (primal dual solver).
    */
  def rankOneProblemSimplex(dim:Int, solverType:String, debugLevel:Int):
  OptimizationProblem with KnownMinimizer = {

    assert(dim>1,"\n\ndim must be >1 , but dim = "+dim+"\n\n")

    val id = "Simplicial rankOneProblem"
    if(debugLevel>0) {
      println("\nAllocating problem " + id)
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
    val x0 = DenseVector.tabulate[Double](dim)(j=>1.0/dim)
    val setWhereDefined = ConvexSets.wholeSpace(dim)
    val ineqs = ConstraintSet(dim,positivityCnts,setWhereDefined,x0)

    val probEq:EqualityConstraint = Constraints.sumToOne(dim)

    val pars = SolverParams.standardParams
    val problem = OptimizationProblem.withoutFeasiblePoint(
      id,setWhereDefined,objF,ineqs,Some(probEq),solverType,pars,logger,debugLevel
    )
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
    *
    * @param solverType: "BR" (barrier solver), "PD" (primal dual solver).
    */
  def rankOneProblemSphere(dim:Int, solverType:String, debugLevel:Int):
  OptimizationProblem with KnownMinimizer = {

    assert(dim>1,"\n\ndim must be >1 , but dim = "+dim+"\n\n")

    val id = "Spherical rankOneProblem"
    if(debugLevel>0) {
      println("\nAllocating problem " + id)
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
    val x0 = DenseVector.tabulate[Double](dim)(j=>1.0/dim)
    val setWhereDefined = ConvexSets.wholeSpace(dim)
    val ineqs = ConstraintSet(dim,cnts,setWhereDefined,x0)

    val pars = SolverParams.standardParams
    val problem = OptimizationProblem.withoutFeasiblePoint(
      id,setWhereDefined,objF,ineqs,None,solverType,pars,logger,debugLevel
    )
    // the known optimal solution
    val x_opt = DenseVector.zeros[Double](dim)
    val minimizer = KnownMinimizer(x_opt,objF)
    problem.addSolution(minimizer)
  }

  /** This problem has many free variables which need to be eliminated
    * in phase I analysis:
    * f(x)=0.5*||x||² subject to x_0<=-1.
    *
    * @param solverType: "BR" (barrier solver), "PD" (primal dual solver).
    */
  def normSquaredWithFreeVariables(dim:Int, solverType:String, debugLevel:Int):
  OptimizationProblem with KnownMinimizer = {

    assert(dim>1,"\n\ndim must be >1 , but dim = "+dim+"\n\n")

    val id = "normSquaredWithFreeVariables"
    if(debugLevel>0) {
      println("\nAllocating problem " + id)
      Console.flush()
    }
    val logFilePath = "logs/"+id+"_log.txt"
    val logger = Logger(logFilePath)

    val objF = ObjectiveFunctions.normSquared(dim)

    val a = DenseVector.tabulate[Double](dim)(j => if(j==0) 1.0 else 0.0)
    val cnt = LinearConstraint("x_1<=-1",dim,-1.0,0.0,a)

    // point where all constraints are defined
    val x0 = DenseVector.fill[Double](dim)(1.0)    // infeasible point
    val setWhereDefined = ConvexSets.wholeSpace(dim)
    val ineqs = ConstraintSet(dim,List(cnt),setWhereDefined,x0)

    val pars = SolverParams.standardParams
    val problem = OptimizationProblem.withoutFeasiblePoint(
      id,setWhereDefined,objF,ineqs,None,solverType,pars,logger,debugLevel
    )
    // the known optimal solution
    val x_opt = DenseVector.zeros[Double](dim)
    x_opt(0) = -1.0
    val minimizer = KnownMinimizer(x_opt,objF)
    problem.addSolution(minimizer)
  }

  /** f(x)= sum(x) subject to ||x||²<=1.
    * Solution is all x_j=-1/sqrt(dim).
    *
    * @param solverType: "BR" (barrier solver), "PD" (primal dual solver).
    */
  def joptP1(dim:Int, solverType:String, debugLevel:Int):
  OptimizationProblem with KnownMinimizer = {

    assert(dim>1,"\n\ndim must be >1 , but dim = "+dim+"\n\n")

    val id = "f(x)=sum(x) with ||x||²<=1"
    if(debugLevel>0) {
      println("\nAllocating problem " + id)
      Console.flush()
    }
    val logFilePath = "logs/"+id+"_log.txt"
    val logger = Logger(logFilePath)

    val a = DenseVector.fill[Double](dim)(1.0)
    val objF = LinearObjectiveFunction(dim,0.0,a)

    val cnt = Constraints.oneHalfNorm2BoundedBy(dim,1.0/2)
    // point where all constraints are defined
    val x0 = DenseVector.fill[Double](dim)(2.0)    // infeasible
    val setWhereDefined = ConvexSets.wholeSpace(dim)
    val ineqs = ConstraintSet(dim,List(cnt),setWhereDefined,x0)

    val pars = SolverParams.standardParams
    val problem = OptimizationProblem.withoutFeasiblePoint(
      id,setWhereDefined,objF,ineqs,None,solverType,pars,logger,debugLevel
    )
    // the known optimal solution
    val x_opt = -a*1.0/sqrt(dim)
    val minimizer = KnownMinimizer(x_opt,objF)
    problem.addSolution(minimizer)
  }

  /** See docs/OptimizerExamples.pdf, example 1.5
    * f(x)=x'Px subject to x_j>=0, sum(x)=1 in dimension 2.
    *
    * @param solverType: "BR" (barrier solver), "PD" (primal dual solver).
    */
  def joptP2(solverType:String, debugLevel:Int): OptimizationProblem with KnownMinimizer = {

    val id = "example_1.5"
    if(debugLevel>0) {
      println("\nAllocating problem " + id)
      Console.flush()
    }
    val logFilePath = "logs/"+id+"_log.txt"
    val logger = Logger(logFilePath)

    val dim=2
    val a = DenseVector.zeros[Double](dim)
    val P = DenseMatrix((1.0,0.4),(0.4,1.0))
    val objF = QuadraticObjectiveFunction(dim,0.0,a,P)

    val probEq:EqualityConstraint = Constraints.sumToOne(dim)
    val cnts = Constraints.allCoordinatesPositive(dim)
    // point where all constraints are defined
    val x0 = DenseVector.fill[Double](dim)(2.0)    // infeasible
    val setWhereDefined = ConvexSets.wholeSpace(dim)
    val ineqs = ConstraintSet(dim,cnts,setWhereDefined,x0)

    val pars = SolverParams.standardParams
    val problem = OptimizationProblem.withoutFeasiblePoint(
      id,setWhereDefined,objF,ineqs,Some(probEq),solverType,pars,logger,debugLevel
    )
    // the known optimal solution
    val x_opt = DenseVector(0.5,0.5)
    val minimizer = KnownMinimizer(x_opt,objF)
    problem.addSolution(minimizer)
  }

  /** This is the problem of minimizing f(x)=(x_1+x_2+...+x_n - 1)^^2 over
    * the positive orthant x_j>=0.
    * This problem comes up in the phase I analysis of the probability simplex.
    * We use it for debugging this very analysis.
    * The minimizer is obviously not uniquely determined and the set of minimizers
    * is exactly the probability simplex.
    *
    * @param solverType: "BR" (barrier solver), "PD" (primal dual solver).
    */
  def probabilitySimplexProblem(n:Int, solverType:String, debugLevel:Int):
  OptimizationProblem with KnownMinimizer = {

    val id = "Probability_simplex_problem"
    if(debugLevel>0) {
      println("\nAllocating problem " + id)
      Console.flush()
    }
    val logFilePath = "logs/"+id+"_log.txt"
    val logger = Logger(logFilePath)

    val a = DenseVector.fill[Double](n)(1.0)
    val P:DenseMatrix[Double] = a*a.t
    // a'x = x_1+x_2+...+x_n, x'Px = (x_1+x_2+...+x_n)^^2
    // f(x) = 0.5 - a'x + x'Px = 0.5*(x_1+x_2+...+x_n -1)^^2
    val objF = QuadraticObjectiveFunction(n,0.5,-a,P)

    val cnts = Constraints.allCoordinatesPositive(n)
    // point where all constraints are defined
    val x0 = DenseVector.fill[Double](n)(2.0)    // infeasible
    val setWhereDefined = ConvexSets.wholeSpace(n)
    val ineqs = ConstraintSet(n,cnts,setWhereDefined,x0)

    val pars = SolverParams.standardParams
    val problem = OptimizationProblem.withoutFeasiblePoint(
      id,setWhereDefined,objF,ineqs,None,solverType,pars,logger,debugLevel
    )
    // the known optimal solution
    val x_opt = DenseVector.fill[Double](n)(1.0/n)
    val minimizer = KnownMinimizer(x_opt,objF)
    problem.addSolution(minimizer)
  }

  /** A problem in R^^{n+1} with known solution x=e_{n+1} (where {e_j} denotes the
    * standard basis as usual):
    *           min f(x)=||x||^^2  on ||x-2e_{n+1}||^^2 <= 1.
    *
    * @param solverType: "BR" (barrier solver), "PD" (primal dual solver).
    */
  def distanceFromOrigin0(n:Int, solverType:String, debugLevel:Int):
  OptimizationProblem with KnownMinimizer = {

    val id = "DistanceFromOrigin_1"
    if(debugLevel>0) {
      println("\nAllocating problem " + id)
      Console.flush()
    }
    val logFilePath = "logs/"+id+"_log.txt"
    val logger = Logger(logFilePath)

    val e_np1 = DenseVector.zeros[Double](n+1); e_np1(n) = 1.0     //e_{n+1}

    // constraint 0.5*||x-a||^^2 <= 0.5 with a=2e_{n+1}, rewrite as
    // 0.5*xIx'-a'x+3/2 <= 0
    val I = DenseMatrix.eye[Double](n+1)
    val qc = QuadraticConstraint("Spherical constraint",n+1,0.0,1.5,-e_np1*2.0,I)

    val cnts = List[Constraint](qc)

    // point where all constraints are defined
    val x0 = DenseVector.fill[Double](n+1)(0.0)    // infeasible
    val setWhereDefined = ConvexSets.wholeSpace(n)
    val ineqs = ConstraintSet(n+1,cnts,setWhereDefined,x0)

    val objF = ObjectiveFunctions.normSquared(n+1)

    val pars = SolverParams.standardParams
    val problem = OptimizationProblem.withoutFeasiblePoint(
      id,setWhereDefined,objF,ineqs,None,solverType,pars,logger,debugLevel
    )
    // the known optimal solution
    val x_opt = e_np1
    val minimizer = KnownMinimizer(x_opt,objF)
    problem.addSolution(minimizer)
  }


  /** A problem in R^^{n+1} with known solution x=e_{n+1} (where {e_j} denotes the
    * standard basis as usual).
    *    min f(x)=||x||^^2  subject to ||x-2e_{n+1}|| <= 1 and
    *    x_j+x_{n+1} >= 1, -x_j+x_{n+1} >= 1, j=1,2,...,n.
    *
    * Note that the linear constraints are of the form a_j'x >= 1 with
    * a_j = +-e_j+e_{n+1}. I.e. we are slicing off pieces of the the ball
    * ||x-2e_{n+1}|| <= 1 at the bottom near the solution x = e_{n+1}.
    *
    * @param solverType: "BR" (barrier solver), "PD" (primal dual solver).
    */
  def distanceFromOrigin1(n:Int, solverType:String, debugLevel:Int):
  OptimizationProblem with KnownMinimizer = {

    val id = "DistanceFromOrigin_1"
    if(debugLevel>0) {
      println("\nAllocating problem " + id)
      Console.flush()
    }
    val logFilePath = "logs/"+id+"_log.txt"
    val logger = Logger(logFilePath)

    val e_np1 = DenseVector.zeros[Double](n+1); e_np1(n) = 1.0     //e_{n+1}

    // constraint 0.5*||x-a||^^2 <= 0.5 with a=2e_{n+1}, rewrite as
    // 0.5*xIx'-a'x+3/2 <= 0
    val I = DenseMatrix.eye[Double](n+1)
    val qc = QuadraticConstraint("Spherical constraint",n+1,0.0,1.5,-e_np1*2.0,I)

    val clb = ListBuffer[Constraint](qc)
    // now add all constraints a.x>=1, a=(+-e_j+e_{n+1})
    for(j <- 0 until n){

      val e_j = DenseVector.zeros[Double](n+1); e_j(j)=1.0
      val id_j = "a.x>=1, where a = e_j+e_{n+1}"
      val a:DenseVector[Double] = e_j+e_np1
      val lc1_j = LinearConstraint(id_j,n+1,-1.0,0.0,-a)
      clb += lc1_j
      val b:DenseVector[Double] = (-e_j)+e_np1
      val lc2_j = LinearConstraint(id_j,n+1,-1.0,0.0,-a)
      clb += lc2_j
    }
    val cnts = clb.toList
    // point where all constraints are defined
    val x0 = DenseVector.fill[Double](n+1)(0.0)    // infeasible
    val setWhereDefined = ConvexSets.wholeSpace(n)
    val ineqs = ConstraintSet(n+1,cnts,setWhereDefined,x0)

    val objF = ObjectiveFunctions.normSquared(n+1)

    val pars = SolverParams.standardParams
    val problem = OptimizationProblem.withoutFeasiblePoint(
      id,setWhereDefined,objF,ineqs,None,solverType,pars,logger,debugLevel
    )

    // the known optimal solution
    val x_opt = e_np1
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
    * @param solverType: "BR" (barrier solver), "PD" (primal dual solver).
    */
  def standardProblems(dim:Int, condNumber:Double, solverType:String, debugLevel:Int):
  List[OptimizationProblem with KnownMinimizer] = {

    var theList:List[OptimizationProblem with KnownMinimizer]
    = List(normSquared(dim,debugLevel))

    val q = 2.0+rand()   // needs to be >= 2 for differentiability
    val id = "Random power problem in dimension "+dim+" with m="+dim+" and exponent "+q
    val problem0 = randomPowerProblem(id,dim,dimKernel=0,condNumber,q,debugLevel)

    val problem1 = minX1_no_FP(solverType,debugLevel)
    val a = DenseVector.tabulate[Double](dim)(i => 1.0)
    val problem2 = minDotProduct(a,solverType,debugLevel)
    val problem3 = min_pNorm(dim,p=2.2,solverType,debugLevel)
    val problem4 = min_pNorm(dim,p=4,solverType,debugLevel)
    val problem5 = rankOneProblemSimplex(dim,solverType,debugLevel)
    val problem6 = rankOneProblemSphere(dim,solverType,debugLevel)
    val problem7 = joptP1(dim,solverType,debugLevel)
    val problem8 = joptP2(solverType,debugLevel)
    val problem9 = normSquaredWithFreeVariables(dim,solverType,debugLevel)
    problem0 :: problem1 :: problem2 :: problem3 :: problem4 ::
      problem5 :: problem6 :: problem7 :: problem8 :: problem9 :: theList
  }




}
