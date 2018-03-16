package cvx

import breeze.linalg.{DenseVector, _}
import breeze.numerics.{abs, sqrt}


/**
  * Created by oar on 12/11/16.
  *
  * Collection of convex minimization problems with and without constraints.
  */
object OptimizationProblems {


  /** f(x) = (1/2)*(x dot x).*/
  def normSquared(dim:Int,C:ConvexSet,debugLevel:Int):OptimizationProblem = {

    assert(C.dim==dim)

    val id = "f(x) = 0.5*||x||^2  in dimension "+dim
    if(debugLevel>0) {
      println("\nAllocating problem " + id)
      Console.flush()
    }
    val logFilePath = "logs/"+id+"_log.txt"
    val logger = Logger(logFilePath)

    val startingPoint = DenseVector.tabulate[Double](dim)(j=>1+j)
    val objF = ObjectiveFunctions.normSquared(dim)
    val maxIter = 200; val alpha = 0.1; val beta = 0.5; val delta = 1e-8
    val tol = 1e-8; val tolEqSolve = 1e-8; val tolFeas = 1e-9
    val pars = SolverParams(maxIter,alpha,beta,tol,tolEqSolve,tolFeas,delta)

    OptimizationProblem(id,objF,startingPoint,C,pars,logger)
  }

  /** f(x) = (1/2)*(x dot x) on the full Euclidean Space*/
  def normSquared(dim:Int,debugLevel:Int):OptimizationProblem with KnownMinimizer = {

    val minimizer = KnownMinimizer(DenseVector.zeros[Double](dim),ObjectiveFunctions.normSquared(dim))
    val problem = normSquared(dim,ConvexSet.fullSpace(dim),debugLevel)
    problem.addSolution(minimizer)
  }

  /** Unconstrained optimization problem with objective function as in docs/cvx_notes.pdf,
    * example 3.1, p6 with all functions $\phi_j(u)=u^^{2q}$ with $q>1$,
    * i.e. the objective function is globally defined in Euclidean space
    * of dimension dim and has the form
    *           \[ f(x)=\sum_j \alpha_j(a_j dot x)^^{2q} \]
    * with positive coefficients $\alpha_j$, $A$ a matrix of dimension m x n, where m <= n,
    * and $a_j=row_j(A)$.
    * Then $n$ is the dimension of the independent variable $x$ and the global minimum
    * is zero and is assumed at all points in the null space of A.
    * If m < dim this space is nontrivial and we can test how the algorithm behaves in such
    * a case.
    *
    * @param q exponent must be >= 1 for differentiability.
    * parameters etc, see [SolverParams].
    */
  def powerProblem( id:String,
                    A:DenseMatrix[Double], alpha:DenseVector[Double], q:Double,
                    debugLevel:Int
    ): OptimizationProblem with KnownMinimizer = {

    assert(q>=1,"\nExponent q needs to be at least 1 but q="+q+"\n")
    if(debugLevel>0){
      println("\nAllocating problem "+id)
      Console.flush()
    }
    val n=A.cols    // dimension of problem
    val m=A.rows
    assert(m<=n)

    val logFilePath = "logs/"+id+"_log.txt"
    val logger = Logger(logFilePath)

    val startingPoint = DenseVector.tabulate[Double](n)(j => -10+j*Math.sqrt(n))
    val objF:ObjectiveFunction = Type1Function.powerFunction(A,alpha,q)
    val C = ConvexSet.fullSpace(n)
    val minimizer = new KnownMinimizer {

      def theMinimizer:DenseVector[Double] = DenseVector.zeros[Double](n)
      def isMinimizer(x:DenseVector[Double],tol:Double):Boolean = norm(A*x)<tol
      def minimumValue:Double = 0.0
    }
    val pars = SolverParams.standardParams(n)
    val problem = OptimizationProblem(id,objF,startingPoint,C,pars,logger)
    problem.addSolution(minimizer)
  }

  /** [powerProblem] in dimension dim with dim x dim matrix A and coefficient vector alpha
    * having random entries in (0,1).
    *
    * @param dimKernel: dimension of solution space = ker(A).
    * @param condNumber: condition number of A.
    */
  def randomPowerProblem(
     id:String,dim:Int,dimKernel:Int,condNumber:Double,q:Double, debugLevel:Int
  ): OptimizationProblem with KnownMinimizer = {

    assert(dimKernel<=dim)
    val A = MatrixUtils.randomMatrix(dim,condNumber)
    val alpha = DenseVector.rand[Double](dim)
    val pars = SolverParams.standardParams(dim)
    powerProblem(id,A,alpha,q,debugLevel)
  }

  /** List of power problems, the first with A the 2x2 identity matrix,
    * the second with A={{1,0},{1,1}} and in each case q=2 and alpha=(1,1).
    */
  def powerProblems(debugLevel:Int):List[OptimizationProblem with KnownMinimizer] = {

    val q = 2.0
    val alpha = DenseVector(1.0,1.0)
    val A = DenseMatrix.eye[Double](2)

    val id1 = "Power problem A=I_2, alpha=(1,1), q=2"
    val p1 = powerProblem( id1,A,alpha,q,debugLevel)

    val B = DenseMatrix.tabulate[Double](2,2)((i,j)=>1.0); B(0,1)=0.0
    val id2 = "Power problem A={{1,0},{1,1}}, alpha=(1,1), q=2"
    val p2 = powerProblem( id2,B,alpha,q,debugLevel)
    List(p1,p2)
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
    * @param solverType: "BR" (barrier solver), "PD0" (primal dual with one slack variable),
    *   "PD1" (primal dual with one slack variable for each inequality constraint), see docs/primaldual.pdf.
    * @param n must be even and bigger than 9 (to ensure feasibility).
    */
  def kl_1(n:Int,solverType:String,debugLevel:Int):OptimizationProblem with KnownMinimizer = {

    assert(n>9 && n%2==0, "\n\nn must be even and > 9, but n = "+n+"\n\n")

    val id = "dist_KL problem 1"
    if(debugLevel>0) {
      println("\nAllocating problem " + id)
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
    val x0 = DenseVector.tabulate[Double](n)(j=>1.0/n)
    val setWhereDefined = ConvexSets.wholeSpace(n)
    val ineqs = ConstraintSet(n,constraints,setWhereDefined,x0)

    val probEq:EqualityConstraint = Constraints.sumToOne(n)
    val pars = SolverParams.standardParams(n)
    val problem = OptimizationProblem(
      id,setWhereDefined,objF,ineqs,Some(probEq),solverType:String,pars,logger,debugLevel
    )

    // the heuristic optimal solution
    val x_opt = if(1.8/n>0.12) DenseVector.tabulate[Double](n)(
      j => if(j<3) 1.8/n else if (j>=n/2) 0.2/n else (1.8*n-10.8)/(n*(n-6))
    ) else DenseVector.tabulate[Double](n)(
      j => if(j<3) 0.12 else if (j>=n/2) 0.2/n else (1.08)/(n-6)
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
    * Using symmetry and it can be shown that the optimum occurs at the following
    * probability distribution x (see docs/Dist_KL.pdf):
    * IF 1.8/n>=0.12:
    *
    * x_j=0.36/3, j=0,1,2
    * x_j=0.2/n,       j=n/2,n/2+1,...,n-1
    * x_j=(1-0.36-0.1)/(n-n/2-3), all other j
    *
    * @param solverType: "BR" (barrier solver), "PD0" (primal dual with one slack variable),
    *   "PD1" (primal dual with one slack variable for each inequality constraint), see docs/primaldual.pdf.
    * @param n must be even and bigger than 9 (to ensure feasibility).
    */
  def kl_2(n:Int, solverType:String, debugLevel:Int):OptimizationProblem with KnownMinimizer = {

    assert(n>9 && n%2==0, "\n\nn must be even and > 9, but n = "+n+"\n\n")

    val id = "dist_KL problem 2"
    if(debugLevel>0) {
      println("\nAllocating problem " + id)
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
    val x0 = DenseVector.tabulate[Double](n)(j=>1.0/n)
    val setWhereDefined = ConvexSets.wholeSpace(n)
    val ineqs = ConstraintSet(n,positivityCnts,setWhereDefined,x0)

    val pars = SolverParams.standardParams(n)
    val problem = OptimizationProblem(
      id,setWhereDefined,objF,ineqs,Some(eqs),solverType,pars,logger,debugLevel
    )
    // the known optimal solution
    val x_opt = DenseVector.tabulate[Double](n)(
      j => if(j<3) 0.12 else if (j>=n/2) 0.2/n else 1.08/(n-6)
    )
    val minimizer = KnownMinimizer(x_opt,objF)
    problem.addSolution(minimizer)
  }


  /** A infeasible problem: sum of probabilities of disjoint events bigger than one.
    *
    * @param solverType: "BR" (barrier solver), "PD0" (primal dual with one slack variable),
    *   "PD1" (primal dual with one slack variable for each inequality constraint), see docs/primaldual.pdf.
    * @param n must be even and bigger than 9 (to ensure feasibility).
    */
  def infeasible_kl_1(n:Int,solverType:String,debugLevel:Int):OptimizationProblem = {

    assert(n>9 && n%2==0, "\n\nn must be even and > 9, but n = "+n+"\n\n")

    val id = "dist_KL problem 3 (infeasible)"
    if(debugLevel>0) {
      println("\nAllocating problem " + id)
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

    val pars = SolverParams.standardParams(n)
    val setWhereDefined = ConvexSets.wholeSpace(n)
    OptimizationProblem(id,setWhereDefined,objF,ineqs,Some(probEq),solverType,pars,logger,debugLevel)
  }



}