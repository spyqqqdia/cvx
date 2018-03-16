package cvx

import breeze.linalg.{DenseMatrix, DenseVector, _}
import breeze.numerics.{abs, log}




// WARNING: 
// in the barrier method handle the multiplications with the dimension reducing
// matrix x = x0+Fu _outside_ the sum in the barrier function (bilinear!) or else we will matrix
// multiply ourselves to death.
// This is the reason why we do not put this operation into the constraints themselves.

/** Solver for constrained convex optimization using the barrier method.
  * C.samplePoint will be used as the starting point of the optimization.
  *
  * Note: C, objF, constraintSet, eqs all are in the dimension of the
  * original variable with slack variables added as in docs/primaldual.pdf.
  * Below x always denotes this larger variable.
  *
  * @param C domain of definition of the Problem
  * @param objF objective function of the optimization problem (needed to
  *            monitor the optimization state)
  * @param eqs Optional equality constraint(s) of the form Ax=b
  * @param numIneqsWithoutSlacks original number of inequality constraints before the
  *              positivity constraints on slack variables, see docs/primaldual.pdf
  * @param numSlacks number of slack variables relaxing the constraints,
  *                  see docs/primaldual.pdf
  * @param pars see [SolverParams]
  */
class PrimalDualSolver(
  val C:ConvexSet, val startingPoint:DenseVector[Double],
  val objF:ObjectiveFunction,
  val constraintSet:ConstraintSet, val eqs:Option[EqualityConstraint],
  val numIneqsWithoutSlacks:Int, val numSlacks:Int, val pars:SolverParams, val logger:Logger
)
  extends Solver { self =>

  //  check if the pieces fit together
  assert(C.dim==startingPoint.length,
    "\nDimension mismatch C.dim="+C.dim+", startingPoint.length="+startingPoint.length+"\n"
  )
  assert(objF.dim==startingPoint.length,
    "\nDimension mismatch objF.dim="+objF.dim+", startingPoint.length="+startingPoint.length+"\n"
  )
  assert(C.isInSet(startingPoint),"Starting point x not in set C, x:\n"+startingPoint+"\n")

  eqs.map(eqCnt => {

    val A:DenseMatrix[Double] = eqCnt.A
    assert(C.dim==A.cols,  "\n\nDimension mismatch: C.dim = "+C.dim+", A.cols = "+A.cols+"\n")
  })

  override val dim:Int = C.dim
  /** Number of inequality constraints. This contains inequalities on
    * slack variables if such are present!*/
  val numIneqsWithSlacks:Int = constraintSet.numConstraints
  /** Number of equality constraints.*/
  val numEqConstraints:Int = if(eqs.nonEmpty) eqs.get.A.rows else 0

  assert(numIneqsWithSlacks == numIneqsWithoutSlacks + numSlacks,
    "\nTotal number of inequalities with slacks = "+numIneqsWithSlacks+" is not the sum of "+
    "\nthe number of original inequalities = "+numIneqsWithoutSlacks+" and"+
    "\nthe number of slack variables = "+numSlacks+"\n"
  )

  def checkDim(x:DenseVector[Double]):Unit =
    assert(x.length==dim,"Dimension mismatch x.length="+x.length+" unequal to dim="+dim)

  private def logStep(t:Double):Unit = {

    val border = "\n****************************************************************\n"
    val content =    "**          PrimalDualSolver: step t = "+t+"                  **"
    val msg = "\n"+border+content+border+"\n"
    println(msg); Console.flush()
    logger.println(msg)
  }

  /** Dual residual r_dual(x,lambda) when there are no equality constraints
    * (boyd-vandenberghe, p610).
    * @param x variable with slacks
    */
  private def dualResidual_noEqs(
    x:DenseVector[Double],lambda:DenseVector[Double]
  ): DenseVector[Double] = {

    val grad_f = objF.gradientAt(x)
    val dgx = constraintSet.gradientMatrixAt(x) // constraint gradients in rows

    grad_f + dgx.t*lambda
  }
  /** Dual residual r_dual(x,lambda) when there are equality constraints Ax=b
    * (boyd-vandenberghe, p610).
    * @param x variable with slacks
    */
  private def dualResidual_withEqs(
    x:DenseVector[Double],lambda:DenseVector[Double], nu:DenseVector[Double]
  ): DenseVector[Double] = {

    assert(eqs.nonEmpty,
      "\ndualResidual_withEqs undefined when no equality constraints present\n"
    )
    val grad_f = objF.gradientAt(x)
    val dgx = constraintSet.gradientMatrixAt(x)

    val A = eqs.get.A
    val b = eqs.get.b

    grad_f + dgx.t*lambda + A.t*nu
  }


  /** Central residual r_central(t,x,lambda), see boyd-vandenberghe, p610.
    * This does not depend on any equality constraints.
    * @param x variable with slacks
    */
  private def centralResidual(
    t:Double, x:DenseVector[Double], lambda:DenseVector[Double]
  ): DenseVector[Double] = {

    assert(lambda.length==numIneqsWithSlacks,
      "\ndim(lambda)="+lambda.length+"not equal to numConstraints="+numIneqsWithSlacks+"\n"
    )
    val g = constraintSet.constraintFunctionAt(x)
    DenseVector.tabulate[Double](numIneqsWithSlacks)(i => -lambda(i)*g(i)-1.0/t)
  }
  /** Primal residual r_prim(t,x,lambda), see boyd-vandenberghe, p610.
    * This is only defined equality constraints are present in which case
    * it is simply the residual Ax-b.
    */
  private def primalResidual(x:DenseVector[Double]): DenseVector[Double] = {

    assert(eqs.nonEmpty,
      "\nPrimal residual is not defined when no equality constraints are present.\n"
    )
    val A = eqs.get.A
    val b = eqs.get.b
    A*x-b
  }

  /** The complete residual vector (dual + central), when no equalities are present.
    */
  private def residual_noEqs(
    t:Double, x:DenseVector[Double],lambda:DenseVector[Double]
  ): DenseVector[Double] = {

    val r_dual = dualResidual_noEqs(x,lambda)
    val r_central = centralResidual(t,x,lambda)
    DenseVector.vertcat(r_dual,r_central)
  }
  /** The complete residual vector (dual + central), when no equalities are present.
    */
  private def residual_withEqs(
    t:Double, x:DenseVector[Double], lambda:DenseVector[Double], nu:DenseVector[Double]
  ): DenseVector[Double] = {

    assert(eqs.nonEmpty,
      "\nresidual_withEqs undefined when no equality constraints present\n"
    )
    val r_dual = dualResidual_withEqs(x,lambda,nu)
    val r_cent = centralResidual(t,x,lambda)
    val r_prim = primalResidual(x)
    DenseVector.vertcat(r_dual,r_cent,r_prim)
  }


  /** With the following functions we set up the KKT system for the primal dual
    * search as in boyd-vandenberghe, p610, equation (11.55).
    * A problem occurs when a constraint f_i(x)=g_i(x)-u_i<=0 is active, i.e.
    * when f_i(x)=0.
    * In that case the expressions in cited location are not defined.
    * We take the position that then dLambda(i) should be zero and we drop the
    * constraint from the computation.
    * the KKT system (11.53), p609, is then only solvable for t=+oo and
    * the middle portion of (11.53) puts no condition on lambda_i.
    */

  /** First component $\grad f(x)+(1/t)\sum_i\frac{1}{u_i-g_i(x)}\grad g_i(x)$
    * of the right hand side (rhs) of boyd-vandenberghe, p610, eq.(11.55).
    * Here the term A'nu is dropped and the constraints are f_i(x)=g_i(x)-u_i<=0.
    */
  private def rhs1(t:Double,x:DenseVector[Double]):DenseVector[Double] = {

    val cnts = constraintSet.constraints
    var result = -objF.gradientAt(x)
    assert(cnts.length==numIneqsWithSlacks)
    for(cnt <- cnts) {

      // constraint cnt: g(x)<=ub, f(x)=g(x)-ub <= 0
      val fx = cnt.valueAt(x) - cnt.ub
      val grad_gx = cnt.gradientAt(x)

      result += grad_gx /(t*fx)
    }
    result
  }

  /** $\delta\lambda_{pd}$ as eliminated from the KKT system of the primal-dual
    * method in terms of x, lambda and dx. See boyd-vandenberghe, section 11.7.1,
    * p610, above equation (11.55).
    *
    * @param t analogue of barrier penalty parameter.
    */
  private def deltaLambda(
    t:Double,x:DenseVector[Double],dx:DenseVector[Double],lambda:DenseVector[Double]
  ):DenseVector[Double] = {

    val nIneqs = numIneqsWithSlacks  // number of inequality constrains
    assert(lambda.length == nIneqs,
      "\n dim(lambda) = "+lambda.length+" is not equal to numIneqs = "+nIneqs+".\n"
    )
    assert(x.length==dx.length,
      "\ndim(x) = "+x.length+"not equal to dim(dx) = "+dx.length+".\n"
    )
    val r_cent = centralResidual(t,x,lambda)
    val gx = constraintSet.constraintFunctionAt(x)
    val Dgx = constraintSet.gradientMatrixAt(x)
    assert(gx.forall(_<0),
      "\ngx not < 0, gx = " + gx +
        "\nline search did not pull back into strictly feasible region!\n"
    )
    val w = Dgx*dx
    assert(w.length==lambda.length,
      "\ndim(w) = "+w.length+" not equal to dim(lambda) = "+lambda.length+".\n"
    )
    // skip the active constraints
    for(i <- 0 until nIneqs) w(i) = -w(i)*(lambda(i)/gx(i)) + r_cent(i)/gx(i)
    w
  }


  /** The matrix H_pd from boyd-vandenberghe, p611, equation (11.56)
    * This defines the KKT system with dLambda eliminated when no equality constraints
    * are present.
    */
  private def kktMatrix_noEqs(
    x:DenseVector[Double], lambda:DenseVector[Double]
  ):DenseMatrix[Double] = {

    assert(lambda.length==numIneqsWithSlacks,
      "\ndim(lambda)="+lambda.length+"not equal to numConstraints="+numIneqsWithSlacks+"\n"
    )
    val cnts = constraintSet.constraints
    // the upper left block containing the Hessians
    var hpd = objF.hessianAt(x)
    for(i <- 0 until numIneqsWithSlacks) {

      val li = lambda(i)
      val cnt_i = cnts(i)
      val fi = cnt_i.valueAt(x) - cnt_i.ub // constraint f_i(x)=g_i(x)-u_i <= 0
      val grad_gi = cnt_i.gradientAt(x)
      val grad2_gi = cnt_i.hessianAt(x)

      assert(fi<0,"\nfi = "+fi+" not < 0, line search pulled back into strictly feasible region?\n")
      hpd += grad2_gi * li - (grad_gi * grad_gi.t) * (li / fi)
    }
    hpd
  }


  /** The system (11.55), boyd-vandenberghe, p610, with no rows and columns
    * containing the matrix A.
    * This is a plain vanilly symmetric positive definite linear system.
    *
    * Note: the variable lambda has been eliminated from this system.
    * Therefore the solution only yields the vector dx. For the subsequent line
    * search in the variables u=(x,lambda) this has to be augmented to
    * du = (dx,dlambda) using the explicit formula for dlambda
    *
    * @param t analogue of barrier penalty parameter.
    */
  private def kktSystem_noEqs(
    t:Double, x:DenseVector[Double], lambda:DenseVector[Double]
  ):SymmetricLinearSystem = {

    val H = kktMatrix_noEqs(x,lambda)
    val rhs = rhs1(t,x)
    SymmetricLinearSystem(H,rhs,logger)
  }

  /** The system (11.55), boyd-vandenberghe, p610.
    * Note that dlambda has been eliminated from this system so it yields
    * dv = (dx,dnu) which then has to be augmented by dlambda computed from
    * [[deltaLambda]] to du = (dx,dlambda,dnu) for the subsequent line search
    * in the variables u = (x,lambda,nu).
    *
    * @param t analogue of barrier penalty parameter.
    */
  private def kktSystem_withEqs(
    t:Double, x:DenseVector[Double], lambda:DenseVector[Double], nu:DenseVector[Double]
  ):KKTSystem = {

    assert(eqs.nonEmpty,
      "\nkktMatrix_withEqs is not defined when no equality constraints are present.\n"
    )
    val A = eqs.get.A
    val H = kktMatrix_noEqs(x,lambda)
    val v = rhs1(t,x)
    val q = v+A.t*nu
    val r_pri = primalResidual(x)

    KKTSystem(H,A,q,-r_pri)     // FIX ME: check this!
  }

  /** The surrogate duality gap, boyd-vandenberghe, section 11.7.2, p612.
    */
  private def surrogateDualityGap(
    x:DenseVector[Double],lambda:DenseVector[Double]
  ): Double ={

    assert(lambda.length==numIneqsWithSlacks,
      "\ndim(lambda)="+lambda.length+"not equal to numConstraints="+numIneqsWithSlacks+"\n"
    )
    -constraintSet.constraintFunctionAt(x) dot lambda
  }




  /************************************************************************
    ***        Solution with no equality constraints                    ***
    ***********************************************************************/

  /** The backtracking line search, boyd-vandenberghe, section 11.7.3, p612,
    * when no equality constraints are present.
    * @param z current iterate (z=(x,lambda)).
    * @param dz current search direction (dz=(dx,dLambda)).
    */
  private def lineSearch_noEQs(
    t:Double, z:DenseVector[Double], dz:DenseVector[Double]
  ):DenseVector[Double]  = {

    val nIneqs = numIneqsWithSlacks
    assert(z.length == dim+nIneqs,
      "Dimension of z = "+z.length+" is not equal to dim(x)+dim(lambda),\n" +
      "dim(x) = "+dim+", dim(lambda) = numConstraints = "+nIneqs + ".\n"
    )
    assert(z.length == dz.length,
      "Dimension of z = "+z.length+" is not equal to dim(dz) = "+dz.length+".\n"
    )
    val x = z(0 until dim)
    val lambda = z(dim until dim+nIneqs)

    assert(lambda.forall(_>0.0),"\nlambda not positive,\nlambda = "+lambda+"\n")

    val dx = dz(0 until dim)
    val dLambda = dz(dim until dim+nIneqs)

    // compute s0=s_max
    var s0 = 1.0
    var i = 0
    while(i<lambda.length){

      if(dLambda(i)<0 && -lambda(i)/dLambda(i)<s0) s0 = -lambda(i)/dLambda(i)
      i+=1
    }
    var s = 0.99*s0
    // data for termination criterion
    val alpha = pars.alpha
    val beta = pars.beta
    val r_t = residual_noEqs(t,x,lambda)
    var x_s = x+dx*s
    var lambda_s = lambda+dLambda*s
    var r_ts = residual_noEqs(t,x_s,lambda_s)
    var isInC = C.isInSet(x_s)
    var isStrictlyFeasible = constraintSet.isSatisfiedStrictlyBy(x_s)
    var lambdaIsPositive = lambda_s.forall(_>0)
    var residualDecreased = norm(r_ts) < (1-alpha*s)*norm(r_t)
    var iter = 0
    val maxIter = -30/log(beta)

    while(!(
        isInC && isStrictlyFeasible && lambdaIsPositive && residualDecreased
      ) && iter <= maxIter
    ){

      s *= beta
      x_s = x+dx*s
      lambda_s = lambda+dLambda*s
      r_ts = residual_noEqs(t,x_s,lambda_s)
      isInC = C.isInSet(x_s)
      isStrictlyFeasible = constraintSet.isSatisfiedStrictlyBy(x_s)
      lambdaIsPositive = lambda_s.forall(_>0)
      residualDecreased = norm(r_ts) < (1-alpha*s)*norm(r_t)
      iter += 1
    }
    if(iter>=maxIter){
      val msg = "\nLine search unsuccessful.\n"
      throw LineSearchFailedException(msg)
    }
    DenseVector.vertcat(x_s,lambda_s)
  }

  /** Find the location $x$ of the minimum of f=objF over C by the Newton method
    * starting from the starting point x0 with no equality constraints.
    *
    * @return Solution object: minimizer with additional info.
    */
  def solve_noEQs(
    terminationCriterion:(OptimizationState)=>Boolean, debugLevel:Int=0
  ):Solution = {

    if(debugLevel>0){

      val msg = "\n#---- PrimalDualSolver::solve_noEqs: ----#\n"
      println(msg)
      logger.println(msg)
    }
    val tol=pars.tol // tolerance for duality gap
    val mu = 10.0    // factor to increase parameter t

    // primal dual starting points,
    // recall:
    // numIneqs: number of inequality constraints before constraints on slacks
    // numIneqConstraints: number of inequality constraints including constraints on slacks
    var x = startingPoint       // iterates x=x_k
    var lambda = DenseVector.ones[Double](numIneqsWithSlacks)

    var obfFcnValue = Double.MaxValue
    var dualityGap = surrogateDualityGap(x,lambda)
    val equalityGap = 0.0
    var normDualResidual = Double.MaxValue

    var t = mu*numIneqsWithSlacks/dualityGap      // barrier penalty analogue

    // None: norm of gradient, Newton decrement
    var optimizationState = OptimizationState(
      None,None,Some(dualityGap),Some(equalityGap),obfFcnValue,Some(normDualResidual)
    )
    // insurance against non terminating loop, normally terminates long before that
    val maxIter = 2000/mu
    var iter = 0
    var u:DenseVector[Double] = DenseVector.vertcat(x,lambda)  // starting iterate
    while(!terminationCriterion(optimizationState) && iter<maxIter){

      if(debugLevel>2) logStep(t)

      // recall: this yields only dx not all of du=(dx,dlambda)
      val LS = kktSystem_noEqs(t,x,lambda)

      // search direction for du=(dx,dLambda)
      val dx = LS.solve(pars.tolEqSolve,debugLevel)
      val dlambda = deltaLambda(t,x,dx,lambda)
      val du = DenseVector.vertcat(dx,dlambda)
      val u_next = lineSearch_noEQs(t,u,du)

      if(debugLevel>0){

        val msg = "\nPrimalDualSolver::solve_noEqs:\ndx = "+dx+"\ndLambda = "+dlambda+"\n"
        println(msg)
        logger.println(msg)
      }

      x = u_next(0 until dim)
      lambda = u_next(dim until dim+numIneqsWithSlacks)

      obfFcnValue = objF.valueAt(x)
      dualityGap = surrogateDualityGap(x,lambda)
      normDualResidual = norm(dualResidual_noEqs(x,lambda))
      // None: norm of gradient, Newton decrement
      optimizationState = OptimizationState(
        None,None,Some(dualityGap),Some(equalityGap),obfFcnValue,Some(normDualResidual)
      )
      if(debugLevel>3){
        print("\nOptimization state:"+optimizationState)
        Console.flush()
      }
      t = mu*numIneqsWithSlacks/dualityGap
      iter+=1
    }
    // split x into slacks s and original variables w
    val w = x(0 until (dim-numSlacks))
    val s = if(numSlacks==0) None else Some(x((dim-numSlacks) until dim))
    val maxedOut = iter==maxIter
    Solution(
      w,s,Some(lambda),None,
      None,Some(dualityGap),None,None,Some(normDualResidual),
      iter-1,maxedOut
    )
  }





  /************************************************************************
    ***           Solution with equality constraints                    ***
    ***********************************************************************/


  /** The backtracking line search, boyd-vandenberghe, section 11.7.3, p612,
    * when no equality constraints are present.
    * @param z current iterate (z=(x,lambda,nu)).
    * @param dz current search direction (dz=(dx,dLambda,dNu)).
    */
  private def lineSearch_withEQs(
    t:Double, z:DenseVector[Double], dz:DenseVector[Double]
  ):DenseVector[Double]  = {

    assert(eqs.nonEmpty,
      "\nlineSearch_withEQs undefined when no equality constraints are present.\n"
    )
    val nEqs = numEqConstraints     // number of equality constraints
    val nIneqs = numIneqsWithSlacks     // number of inequality constraints
    assert(z.length == dim+nIneqs+nEqs,
      "Dimension of z = "+z.length+" is not equal to dim(x)+dim(lambda)+dim(nu),\n" +
        "dim(x) = "+dim+", dim(lambda) = numIneqs = "+nIneqs + ", dim(nu) = numEqs = "+nEqs+".\n"
    )
    val x = z(0 until dim)
    val lambda = z(dim until dim+nIneqs)
    val nu = z(dim+nIneqs until dim+nIneqs+nEqs)

    assert(lambda.forall(_>0.0),"\nlambda not positive,\nlambda = "+lambda+"\n")

    val dx = dz(0 until dim)
    val dLambda = dz(dim until dim+nIneqs)
    val dNu = dz(dim+nIneqs until dim+nIneqs+nEqs)

    // compute s0=s_max
    var s0 = 1.0
    var i = 0
    while(i<lambda.length){

      if(dLambda(i)<0 && -lambda(i)/dLambda(i)<s0) s0 = -lambda(i)/dLambda(i)
      i+=1
    }
    s0 = 0.99*s0

    // termination criterion
    val r_t = residual_noEqs(t,x,lambda)
    val lineSearchTC:(Double)=>Boolean = (s:Double) => {

      val x_s = x+dx*s
      val lambda_s = lambda+dLambda*s
      val nu_s = nu+dNu*s

      val r_ts = residual_withEqs(t,x_s,lambda_s,nu_s)
      val alpha = pars.alpha
      C.isInSet(x_s) &&
        constraintSet.isSatisfiedStrictlyBy(x_s) &&
          lambda_s.forall(_>0) &&
            norm(r_ts) < (1-alpha*s)*norm(r_t)
    }
    CvxUtils.lineSearch(z,dz,lineSearchTC,pars.beta,s0)
  }

  /** Find the location $x$ of the minimum of f=objF over C by the Newton method
    * starting from the starting point x0 with no equality constraints.
    *
    * @return Solution object: minimizer with additional info.
    */
  def solve_withEQs(
    terminationCriterion:(OptimizationState)=>Boolean, debugLevel:Int=0
  ):Solution = {

    assert(eqs.nonEmpty,
      "\nsolve_withEQs undefined when no equality constraints are present.\n"
    )
    val A = eqs.get.A
    val b = eqs.get.b
    val nEqs = numEqConstraints         // number of equality constraints

    val tol=pars.tol // tolerance for duality gap
    val mu = 10.0    // factor to increase parameter t in barrier method.

    // primal dual starting points
    // iterates x=x_k, includes the slack variables s_j
    var x = startingPoint
    // only the original variables x_j without slacks s_j
    var w = x(0 until dim-numSlacks)
    var lambda = DenseVector.ones[Double](numIneqsWithSlacks)
    var nu = DenseVector.zeros[Double](A.rows)   // A.rows: number of equality constraints
    var u:DenseVector[Double] = DenseVector.vertcat(x,lambda,nu)  // starting iterate

    var obfFcnValue = Double.MaxValue
    var dualityGap = surrogateDualityGap(x,lambda)
    var equalityGap = Double.MaxValue
    var normDualResidual = Double.MaxValue

    var t = mu*numIneqsWithSlacks/dualityGap   // analogue of t in barrier method

    // None: norm of gradient, Newton decrement
    var optimizationState = OptimizationState(
      None,None,Some(dualityGap),Some(equalityGap),obfFcnValue,Some(normDualResidual)
    )
    // insurance against non terminating loop, normally terminates before that
    val maxIter = 1500/mu
    var iter = 0
    while(!terminationCriterion(optimizationState) && iter<maxIter){

      if(debugLevel>2) logStep(t)

      val KS = kktSystem_withEqs(t,x,lambda,nu)
      // recall lambda has been eliminated, search direction for (x,nu)
      val (dx,dNu) = KS.solve(pars.delta,logger,pars.tolEqSolve,debugLevel)
      val dLambda = deltaLambda(t,x,dx,lambda)
      val du = DenseVector.vertcat[Double](dx,dLambda,dNu)
      val u_next = lineSearch_withEQs(t,u,du)

      x = u_next(0 until dim)        // contains the slack variables!
      w = x(0 until dim-numSlacks)   // the original variables without slacks
      lambda = u_next(dim until dim+numIneqsWithSlacks)
      nu = u_next(dim+numIneqsWithSlacks until dim+numIneqsWithSlacks+nEqs)

      obfFcnValue = objF.valueAt(x)
      equalityGap = norm(A*w-b)
      dualityGap = surrogateDualityGap(x,lambda)
      normDualResidual = norm(residual_withEqs(t,x,lambda,nu))
      // None: norm of gradient, Newton decrement
      optimizationState = OptimizationState(
        None,None,Some(dualityGap),Some(equalityGap),obfFcnValue,Some(normDualResidual)
      )
      if(debugLevel>3){
        print("\nOptimization state:"+optimizationState)
        Console.flush()
      }
      t = mu*numIneqsWithSlacks/dualityGap
      iter+=1
    }
    // split x into slacks s and original variables w
    w = x(0 until (dim-numSlacks))
    val s = if(numSlacks==0) None else Some(x(dim until dim+numSlacks))
    val maxedOut = iter==maxIter
    Solution(
      w,s,Some(lambda),None,
      None,Some(dualityGap),Some(equalityGap),None,Some(normDualResidual),
      iter-1,maxedOut
    )
  }



  /** Solution based on standard termination criterion:
    * dualityGap < tol, equalityGap < tol and norm(dualResidual) < tol
    */
  def solve(debugLevel:Int=0):Solution = {

    val terminationCriterion = (os:OptimizationState) =>
      os.dualityGap.get < pars.tol && os.normDualResidual.get < pars.tol
    solveSpecial(terminationCriterion,debugLevel)
  }

  /** Solution based on externally defined  termination criterion.
    */
  def solveSpecial(
    terminationCriterion: (OptimizationState) => Boolean, debugLevel: Int = 0
  ): Solution =
    if(eqs.nonEmpty) solve_withEQs(terminationCriterion,debugLevel)
    else solve_noEQs(terminationCriterion,debugLevel)


  /** Version of _this_ solver which operates on the variable u related to
    * the original variable as x = z0+Fu.
    * This solves the minimization problem under the additional constraint that
    * x is of the form z0+Fu and operates on the variable u. Results are reported using
    * the variable u not x.
    *
    * The intended application is to problems with equality constraints Ax=b, where the solution
    * space of the equality constraints is parametrized as x=z0+Fu, u unconstrained.
    *
    * REMARK: affine transformation can induce catastrophic overhead via large numbers of
    * big matrix multiplications if not handled correctly. In our approach this does not
    * happen, since we transform the completed barrier function instead of transforming
    * all summands in the barrier function and then summing up the transformed parts.
    * Note for example that we do not need a method affineTransformed for the class
    * ConstraintSet.
    *
    * @param u0 a vector satisfying x0 = z0+F*u0, where x0 is the startingPoint of _this_
    *           solver.
    *
    */
  def affineTransformed(
    z0:DenseVector[Double], F:DenseMatrix[Double], u0:DenseVector[Double]
  ): PrimalDualSolver = {

    // pull the domain bs.C back to the u variable
    val dim_u = F.cols
    val x0 = startingPoint
    assert(
      norm(x0-(z0+F*u0))<pars.tolEqSolve,
      "\nu0 does not map to x0 under the variable transform.\n"
    )
    val D = C.affineTransformed(z0,F,u0)
    val transformedConstraintSet = constraintSet.affineTransformed(z0,F)
    val transformedObjF = objF.affineTransformed(z0,F)
    val transformedEqs = eqs.map(_.affineTransformed(z0,F))

    new PrimalDualSolver(
      D,u0,transformedObjF,transformedConstraintSet,transformedEqs,
      numIneqsWithoutSlacks,numSlacks,pars,logger
    )
  }
  /** As [[affineTransformed(z0:DenseVector[Double],F:DenseMatrix[Double], u0:DenseVector[Double])]]
    * with u0 computed as the solution of Fu=x0-z0 using the SVD of F.
    */
  def affineTransformed(z0:DenseVector[Double],F:DenseMatrix[Double]): PrimalDualSolver = {

    val x0 = startingPoint
    val debugLevel = 0
    val u0 = MatrixUtils.svdSolve(F,x0-z0,logger,pars.tolEqSolve,debugLevel)
    affineTransformed(z0,F,u0)
  }

  /** As As [[affineTransformed(z0:DenseVector[Double],F:DenseMatrix[Double], u0:DenseVector[Double])]]
    * with z0,F,u0 computed by the solution space sol. This usually implies a dimension reduction
    * in the independent variable ( x -> u ).
    */
  def reduced(sol:SolutionSpace): PrimalDualSolver = {

    val z0 = sol.z0
    val F  = sol.F
    val x0 = startingPoint
    val u0 = sol.parameter(x0)     // more efficient than MatrixUtils.svdSolve above.
    affineTransformed(z0,F,u0)
  }

}


/** Some factory functions.*/
object PrimalDualSolver {

  /** PrimalDualSolver for minimization with or without equality constraints Ax=b.
    * Here we have a starting point satisfying the inequalities strictly.
    * Therefore no slack variables s are introduced which relax these inequalities.
    *
    * @param startingPoint a point satisfying the inequality constraints in the
    *                      constraintSet strictly.
    */
  def apply(
    C:ConvexSet, startingPoint:DenseVector[Double], objF:ObjectiveFunction,
    constraintSet:ConstraintSet, eqs:Option[EqualityConstraint],
    pars:SolverParams, logger:Logger
  ): PrimalDualSolver = {

    val numSlacks = 0
    val numIneqsWithoutSlacks = constraintSet.numConstraints
    new PrimalDualSolver(
      C,startingPoint,objF,constraintSet,eqs,
      numIneqsWithoutSlacks,numSlacks,pars,logger
    )
  }

  //------------- Solvers with relaxed inequalities, no starting point ---------//




  /** PrimalDualSolver with global relaxation of constraints.
    *
    * PrimalDualSolver for minimization with or without equality constraints Ax=b.
    * Here we have no starting point satisfying the inequalities strictly.
    * The inequality constraints g_i(x) <= u_i are relaxed globally to
    * g_j(x) <= u_j+s with one additional slack variable s>=0 and the objective
    * function f(x) is changed to h(x,s)=f(x)+Ks.
    *
    * For this problem there always exists a strictly feasible point (if
    * the convex set C is the whole space as usual). The parameter K>0 controls the
    * tradeoff between minimizing the slack s on the constraints and the value f(x)
    * of the objective function.
    *
    * If the original problem has a feasible point, then the solution (x,s) of the
    * new problem satisfies s=0 and f(x) minimizes the original problem,
    * see docs/primaldual.pdf.
    *
    * Here C, objF, constraintSet and eqs denote the entities of the origina
    * problem before the slack variable for global relaxation of the constraints
    * has been added.
    *
    */
  def apply(
    C:ConvexSet, objF:ObjectiveFunction,
    constraintSet:ConstraintSet, eqs:Option[EqualityConstraint],
    pars:SolverParams, logger:Logger, K:Double
  ): PrimalDualSolver = {

    val numIneqsBeforeSlacks = constraintSet.numConstraints
    val numSlacks = 1
    val newConstraintSet = constraintSet.globallyRelaxed(eqs,pars,logger,debugLevel = 0)
    val newObjF:ObjectiveFunction = objF.forGloballyRelaxedProblem(K)
    val new_C = ConvexSets.cartesianProduct(C,ConvexSets.wholeSpace(numSlacks))

    val new_Eqs = eqs.map(eqCnt => eqCnt.withSlackVariables(numSlacks))

    val startingPoint = newConstraintSet.feasiblePoint
    new PrimalDualSolver(
      new_C, startingPoint, newObjF, newConstraintSet, new_Eqs,
      numIneqsBeforeSlacks, numSlacks, pars, logger
    )
  }
  /** PrimalDualSolver with local relaxation of constraints.
    *
    * PrimalDualSolver for minimization with or without equality constraints Ax=b.
    * Here we have no starting point satisfying the inequalities strictly.
    * The inequality constraints g_i(x) <= u_i are relaxed idividually to
    * g_j(x) <= u_j+s_j with one additional slack variable s_j for each inequality
    * constraint and the objective function f(x) is changed to h(x,s)=f(x)+(K dot s).
    *
    * For this problem there always exists a strictly feasible point (if
    * the convex set C is the whole space as usual). The parameter K with K_j>0
    * controls the tradeoff between minimizing the slacks s_j on the constraints
    * and the value f(x) of the objective function.
    *
    * If the original problem has a feasible point, then the solution (x,s) of the
    * new problem satisfies s=0 and f(x) minimizes the original problem,
    * see docs/primaldual.pdf.
    *
    * The individual relaxation of the constraints with dedicated slack variables
    * gives more fine grained control in case no feasible point exists.
    * We can prioritize a constraint g_j(x) <= u_j by making the constant K_j
    * large which will put more emphasis on minimizing s_j and hence the slack on
    * the relaxed constraint g_j(x) <= u_j+s_j.
    *
    * Here C, objF, constraintSet and eqs denote the entities of the original
    * problem before the slack variables for local relaxation of the constraints
    * have been added.
    *
    */
  def apply(
    C:ConvexSet, objF:ObjectiveFunction,
    constraintSet:ConstraintSet, eqs:Option[EqualityConstraint],
    pars:SolverParams, logger:Logger, K:Vector[Double]
  ): PrimalDualSolver = {

    val numIneqsBeforeSlacks = constraintSet.numConstraints
    val numSlacks = numIneqsBeforeSlacks
    require(K.length==numSlacks,
      "\nDimension of K = "+K.length+" is not equal to the number "+numSlacks+
        " of slack variables s_j.\n"
    )
    // the relaxed constraints g_j(x) <= u_j+s_j with s_j>=0 are exactly
    // the phase I SOI constraints
    val newConstraintSet = constraintSet.phase_I_SOI_Constraints
    val newObjF:ObjectiveFunction = objF.forLocallyRelaxedProblem(K)
    val new_C = ConvexSets.cartesianProduct(C,ConvexSets.wholeSpace(numSlacks))

    val new_Eqs = eqs.map(eqCnt => eqCnt.withSlackVariables(numSlacks))

    val startingPoint = newConstraintSet.feasiblePoint
    new PrimalDualSolver(
      new_C, startingPoint, newObjF, newConstraintSet, new_Eqs,
      numIneqsBeforeSlacks, numSlacks, pars, logger
    )
  }

}