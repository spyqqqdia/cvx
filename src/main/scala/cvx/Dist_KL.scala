package cvx

import breeze.linalg.{DenseMatrix, DenseVector, diag, sum}
import breeze.numerics.{exp, log}



/**
  * Created by oar on 10.10.17.
  *
  * Let $P=(p_j)_{j=1}^n$ and $Q=(q_j)_{j=1}^n$ denote discrete probability
  * distributions, i.e. $p_j>=0$ and $\sum p_j=1$ and likewise for $Q$.
  *
  * Given any matrices $H$, $A$ with n columns (i.e. such that the products
  * $HQ$ and $AQ$ are defined when $Q$ is viewed as a column vector) and vectors
  * u and r of appropriate size this class implements the problem
  *
  *   ? = argmin_Q dist_KL(P,Q) subject to HQ <= u and AQ=r.
  *
  * Here the inequality HQ >= u is to be interpreted coordinatewise
  * ($(HQ)_i>=u_i$, for all $i$) and
  *   [ dist_KL(Q,P)= E_Q[dQ/dP] = \sum_jq_j(log(q_j)-log(p_j)) ]
  * denotes the Kullback-Leibler distance of Q from P, where $E_Q$ denotes
  * the expectation with  respect to the probability Q.
  * This is also known as the negentropy of P with respect to Q.
  *
  * Here we average the difference of the probabilities of
  * atomic events $E={j}$ in the probabilities P and Q, where the average is taken
  * under Q since we consider Q to be the "true" probability.
  *
  * The matrix-vector product HQ has the following probabilistic interpretation:
  * the discrete probabilities P,Q are viewed as probabilities on the set
  * $\Omega = \{1,2,\dots,n\}$. Now consider the random vector $X:\Omega\mapsto R_n$
  * given by
  *   [ X(j) = col_j(H). ]
  * Then we have
  *   [ HQ = \sum_jq_jcol_j(H) = \sum_jq_jX(j) = E_Q(X) ].
  * Thus the constraints are expectation constraints
  *   [ E_Q(X)\leq u]  and [ E_Q(Y)=r ],
  * where the random vector Y is defined analogeously.
  *
  * Each row of the matrix H defines and expectation constraint for a scalar
  * random variable. Indeed $row_i(H)$ defines the constraint
  *   [ row_i(H)\cdot Q = \sum_jq_jH_{ij} \leq u_i. ]
  *
  * In other words $row_i(H)$ defines the constraint $E_Q(X_i)\leq u_i$
  * on the scalar random variable $X_i$ with values $X_i(j)=H_{ij}$, i.e. $X_i$
  * can be identified with $row_i(H)$.
  *
  * Similarly $row_i(A)$ defines the constraint $E_Q(Y_i)\leq u_i$ on the
  * scalar random variable $Y_i:j\in\Omega\mapsto A_{ij}$ which can be identified
  * with $row_i(A)$.
  *
  * The Kullback-Leibler distance dist_KL(Q,P) is convex as a function of Q
  * and the minimization problem is implemented as an OptimizationProblem with
  * Duality so that the solution can either be computed directly or via a solution
  * of the dual problem.
  *
  * When passing to the dual problem the constraints q_i>=0 can be dropped, since
  * they will be satisfied automatically from the way the q_i are computed from the
  * dual variables.
  * Thus the dimension of the dual problem (H.rows = number of inequality constraints
  * plus A.rows = number of equality constraints plus 1 (for the constraint sum(Q)=1))
  * is typically much smaller than the dimension n of the variable Q in the primal
  * problem. Consequently the preferred approach is the solution via the dual problem.
  */
class Dist_KL(
  override val id:String,
  val n:Int,
  val H:Option[DenseMatrix[Double]],
  val u:Option[DenseVector[Double]],
  val A:Option[DenseMatrix[Double]],
  val r:Option[DenseVector[Double]],
  override val solver:Solver,
  override val logger:Logger
)
extends OptimizationProblem(id,Dist_KL.objectiveFunction(n),solver,logger)
with Duality { self =>

  require(n>0,s"\nn=${n} is not positive.\n")
  require(if(H.nonEmpty) H.get.cols==n else true,
    s"\nH.cols=${n} required, but H.cols=${H.get.cols}.\n"
  )
  require(if(A.nonEmpty) A.get.cols==n else true,
    s"\nA.cols=${n} required, but H.cols=${A.get.cols}.\n"
  )
  // if H is given, the so must be u, if A is given we need r
  require(if(H.nonEmpty) u.nonEmpty else true,"\nH is given but u is missing.\n")
  require(if(A.nonEmpty) r.nonEmpty else true,"\nA is given but r is missing.\n")
  require(if (r.nonEmpty) A.nonEmpty else true, "\nr is given but A is missing.\n")
  require(H.nonEmpty || A.nonEmpty,"\nMust have some inequality or equality constraints")


  ////------------------- dual problem --------------------////
  //
  // Note that not both H and A can be empty since we require either equality
  // or inequality constraints. This we have either
  //
  // (a) both H,u and A,r   or
  // (b) A,r but not H,u    or
  // (c) H,u but not A,r
  //
  // and we will deal with all these cases simultaneously.
  //
  // Note that we must always add the constraint E_Q[1]=1 (i.e. Q is a probability).
  // Thus when we set up the problem we will always have a matrix A and corresponding
  // equality constraints.

  // summand 1 is for the constraint E_Q[1]=1
  val dualDim:Int = if(A.isEmpty) 1+H.get.rows else
                    if(H.isEmpty) 1+A.get.rows else 1+H.get.rows+A.get.rows
  val numInequalities:Int = if(H.isEmpty) 0 else H.get.rows

  val e = 2.7182811828459045    // exp(1.0)
  /** See docs/maxent.pdf, before eq.(20).*/
  val vec_R:DenseVector[Double] = DenseVector.fill[Double](n)(1.0/(n*e))

  /** The vector w=(u,r), see docs/maxent.pdf, after eq.(18).
    * Note that we have to add the right hand side of the equality constraint
    * E_Q[1]=sum(Q)=1 to the vector r (as first coordinate), even if we have no
    * matrix A and vector r.
    */
  val vec_w:DenseVector[Double] = {

    // extend r with new first coordinate 1.0 from the constraint E_Q[1]=sum(Q)=1.
    val r_extended = Dist_KL.r_with_probEQ(n,r)
    if (H.isEmpty) r_extended else DenseVector.vertcat(u.get,r_extended)
  }


  /** The vertically stacked matrix B=(H',A')' with A stacked below H, see
    * docs/maxent.pdf, after eq.(18). Note that A has to be augmented by the
    * constraint E_Q[1]=sum(Q)=1 (as first row).
    */
  val mat_B:DenseMatrix[Double] = {

    // extend A with new first row of 1s from the constraint E_Q[1]=sum(Q)=1.
    val A_extended = Dist_KL.A_with_probEQ(n,A)
    if (H.isEmpty) A_extended else DenseMatrix.vertcat(H.get,A_extended)
  }


  /** The dual objective function $L_*(z)$, where $z=\theta=(\lambda,\nu)$,
    * see docs/maxent.pdf, eq.(20).
    */
  def dualObjFAt(z:DenseVector[Double]):Double =
    -((vec_w dot z) + (vec_R dot exp(-mat_B.t*z)))

  /** The gradient of the dual objective function $-L_*(z)$, where $z=\theta=(\lambda,\nu)$,
    * see docs/maxent.pdf, eq.(21).
    */
  def gradientDualObjFAt(z:DenseVector[Double]):DenseVector[Double] =
    -vec_w + mat_B*(vec_R:*exp(-mat_B.t*z))

  /** The Hessian of the dual objective function $-L_*(z)$, where $z=\theta=(\lambda,\nu)$,
    * see docs/maxent.pdf, eq.(22).
    */
  def hessianDualObjFAt(z:DenseVector[Double]):DenseMatrix[Double] = {

    val y:DenseVector[Double] = vec_R:*exp(-mat_B.t*z)
    // B*diag(y): multiply col_j(B) with y(j):
    val Bdy = DenseMatrix.zeros[Double](mat_B.rows,mat_B.cols)
    for(j <- 0 until Bdy.cols; i <- 0 until Bdy.rows) Bdy(i,j) = mat_B(i,j)*y(j)
    -Bdy*mat_B.t
  }

  /** The function $Q=Q(z)=Q(\lambda,\nu)$ which computes the optimal primal
    * solution $Q_*$ from the optimal dual solution $z_*=(\lambda_*,\nu_*)$.
    * See docs/maxent.pdf
    */
  def primalOptimum(z:DenseVector[Double]):DenseVector[Double] = vec_R:*exp(-mat_B.t*z)

  /** Add the known (unique) solution to the minimization problem.
    * For testing purposes.
    */
  def addKnownMinimizer(optSol:KnownMinimizer):
  OptimizationProblem with Duality with KnownMinimizer  =
    new Dist_KL(id,n,H,u,A,r,solver,logger) with KnownMinimizer {

      override def theMinimizer: DenseVector[Double] = optSol.theMinimizer
      def isMinimizer(x:DenseVector[Double],tol:Double):Boolean = optSol.isMinimizer(x,tol)
      def minimumValue:Double = optSol.minimumValue
    }
}


object Dist_KL {

  /** Given optional equality constraints AQ=r
    * extend A with new first row of 1s from the equality constraint
    * E_Q[1]=sum(Q)=1.
    * @param n dimension dim(Q) of KL-problem.
    */
  def A_with_probEQ(n:Int,A:Option[DenseMatrix[Double]]):DenseMatrix[Double] = {

    val EQ1:DenseMatrix[Double] = DenseMatrix.fill(1,n)(1.0)
    if(A.isEmpty) EQ1 else DenseMatrix.vertcat(EQ1,A.get)
  }

  /** Given optional equality constraints AQ=r
    * extend r with new first coordinate 1.0 from the equality constraint
    * E_Q[1]=sum(Q)=1.
    * @param n dimension dim(Q) of KL-problem.
    */
  def r_with_probEQ(n:Int,r:Option[DenseVector[Double]]):DenseVector[Double] = {

    val r1 = DenseVector.fill(1)(1.0)
    if(r.isEmpty) r1 else DenseVector.vertcat(r1,r.get)
  }



  /** Kullback-Leibler distance
    *
    *      d_KL(x,p) = sum_jp_j\log(p_j/x_j) = c-sum_jp_j\log(x_j)
    *                = c-sum_j\log(x_j)/n
    *
    * from a discrete uniform distribution p on Omega={1,2,...,n}, p_j=1/n; j=1,2,...,n.
    * Here c is the constant c = -log(n)
    * and even though it is irrelevant in minimization we will not neglect it, since
    * the KL-distance has an information theoretic interpretation.
    */
  def objectiveFunction(n:Int) = new ObjectiveFunction(n) {

    override def valueAt(x: DenseVector[Double]): Double = {

      assert(x.length==n,"\nDimension mismatch x.length = "+x.length+"dim(d_KL) = "+n+"\n")
      x dot log(x*n.toDouble)
    }
    override def gradientAt(x: DenseVector[Double]): DenseVector[Double] =
      DenseVector.tabulate[Double](n)(j => 1+log(x(j))+log(n))

    override def hessianAt(x: DenseVector[Double]): DenseMatrix[Double] = {

      // diagonal
      val d = DenseVector.tabulate[Double](n)(j => 1.0/x(j))
      diag(d)
    }
  }

  def setWhereDefined(n:Int):ConvexSet = ConvexSets.firstQuadrant(n)

  /** The equality constraint Ax=r combined with the constraint sum(x)=1.
    */
  def equalityConstraint(
    n:Int, A:Option[DenseMatrix[Double]],r:Option[DenseVector[Double]]
  ):EqualityConstraint = {

    val probEq:EqualityConstraint = Constraints.sumToOne(n)
    if(A.isEmpty) probEq else {

      assert(r.nonEmpty)
      val eqsAr = EqualityConstraint(A.get,r.get)
      eqsAr.addEqualities(probEq)
    }
  }

  /** The problem of minimizing the Kullback-Leibler dist_KL(Q,P) distance from the
    * discrete uniform distribution P=(p_j) on Omega = {1,2,...,n} (all p_j=1/n) subject
    * to the constraints E_Q[H]<=u and E_Q[A]=r.
    *
    * Here the matrix H is identified with the random vector H on Omega given by
    *    H: j\in Omega --> col_j(H)
    * (thus with Q=(q_j) we have E_Q[H] = \sum_jq_j*col_j(H) = HQ) and similarly for the
    * matrix A.
    *
    * @param solverType: "BR" (barrier solver) or "PD" (primal dual solver).
    * @return the optimization problem with Duality.
    */
  def apply(
    id:String, n:Int,
    H:Option[DenseMatrix[Double]], u:Option[DenseVector[Double]],
    A:Option[DenseMatrix[Double]], r:Option[DenseVector[Double]],
    solverType:String, pars:SolverParams, logger:Logger, debugLevel:Int
  ):Dist_KL = {

    require(n > 0, s"\nn=${n} is not positive.\n")
    require(if (H.nonEmpty) H.get.cols == n else true,
      s"\nH.cols=${n} required, but H.cols=${H.get.cols}.\n"
    )
    require(if (A.nonEmpty) A.get.cols == n else true,
      s"\nA.cols=${n} required, but H.cols=${A.get.cols}.\n"
    )
    // if H is given, the so must be u, if A is given we need r
    require(if (H.nonEmpty) u.nonEmpty else true, "\nH is given but u is missing.\n")
    require(if (A.nonEmpty) r.nonEmpty else true, "\nA is given but r is missing.\n")
    require(if (r.nonEmpty) A.nonEmpty else true, "\nr is given but A is missing.\n")
    require(H.nonEmpty || A.nonEmpty, "\nMust have some inequality or equality constraints")

    require(solverType == "BR" || solverType == "PD", s"\nUnknown solver: ${solverType}\n")

    // set where the constraints are defined
    val C: ConvexSet = ConvexSets.wholeSpace(n)
    val pointWhereDefined = DenseVector.fill[Double](n)(1.0 / n)

    // note: equalityConstraint already contains the constraint E_Q[1] = sum(Q) = 1
    val eqs = equalityConstraint(n,A,r)

    val positivityCnts:List[Constraint] = Constraints.allCoordinatesPositive(n)

    // funnily dist_KL_2A only works if we don't add in the constraint E_Q[1]01
    // val eqs = equalityConstraint(n,A,r)
    val objF = objectiveFunction(n)

    val ineqs = if (H.isEmpty) ConstraintSet(n,positivityCnts,C,pointWhereDefined)
                else ConstraintSet(H.get,u.get,C).addConstraints(positivityCnts)
    val ineqsWithFeasiblePoint = ineqs.withFeasiblePoint(Some(eqs),pars,debugLevel) // phase I

    val solver: Solver = if (solverType == "BR")
        BarrierSolver(objF,ineqsWithFeasiblePoint,Some(eqs),pars,logger)
       else
        PrimalDualSolver(C,objF,ineqsWithFeasiblePoint,Some(eqs),pars,logger)

    new Dist_KL(id,n,H,u,A,r,solver,logger)
  }
}
