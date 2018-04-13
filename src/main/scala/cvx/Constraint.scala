package cvx

import breeze.linalg.{DenseMatrix, DenseVector, _}


/** General inequality constraint of the form g(x)<=ub.
  *
  * @param dim: dimension of independent variable x
  * @param ub: upper bound on g.
  */
abstract class Constraint(val id:String, val dim:Int, val ub:Double){

  self: Constraint =>
  def isDefinedAt(x:DenseVector[Double]):Boolean
  /** This is g(x) _not_ g(x)-ub.*/
  def valueAt(x:DenseVector[Double]):Double
  def gradientAt(x:DenseVector[Double]):DenseVector[Double]
  def hessianAt(x:DenseVector[Double]):DenseMatrix[Double]

  def checkDim(x:DenseVector[Double]):Unit =
    assert(x.length==dim,"Dimension mismatch: x.length = "+x.length+", dim="+dim)
  def isSatisfied(x:DenseVector[Double]):Boolean = valueAt(x)<=ub
  def isSatisfiedStrictly(x:DenseVector[Double]):Boolean = valueAt(x)*(1+3e-16) < ub
  def isSatisfiedWithTolerance(x:DenseVector[Double],tol:Double):Boolean = valueAt(x)<ub+tol
  /** @return |g(x)-ub|<tol. */
  def isActive(x:DenseVector[Double], tol:Double=1e-12):Boolean = Math.abs(valueAt(x)-ub)<tol
  /** @return ub-g(x).*/
  def margin(x:DenseVector[Double]):Double = ub-valueAt(x)


  /** The constraint k(u) = g(z+Fu) <= ub, where g(x) <= ub is _this_ constraint.
    * In other words: _this_ constraint under an affine change of variables
    * x = z+Fu.
    *
    * @param z a vector of dimension dim-p (intended: special solution of Ax=b)
    * @param F a nxp matrix (intended: p = number of equality constraints)
    */
  def affineTransformed(z:DenseVector[Double],F:DenseMatrix[Double]):Constraint = {

    val reducedDim = dim-F.cols
    val reducedID = id+"_reduced"
    new Constraint(reducedID,reducedDim,ub) {


      override def isDefinedAt(u:DenseVector[Double]):Boolean = self.isDefinedAt(z + F * u)
      override def valueAt(u: DenseVector[Double]):Double = self.valueAt(z + F * u)
      override def gradientAt(u: DenseVector[Double]):DenseVector[Double] =
        F.t * self.gradientAt(z + F * u)
      override def hessianAt(u: DenseVector[Double]):DenseMatrix[Double] =
        (F.t * self.hessianAt(z + F * u)) * F
    }
  }

}


object Constraint {


  /** Version of constraint cnt for basic phase I analysis, [boyd], 11.4.1, p579.
    * Recall: one new variable s with upper bound zero and g_j(x)<=ub replaced with
    * g_j(x)-s<=ub.
    */
  def phase_I(cnt:Constraint):Constraint = new Constraint(cnt.id+"_phase_I",cnt.dim+1,cnt.ub){


    def isDefinedAt(u:DenseVector[Double]):Boolean = cnt.isDefinedAt(u(0 until cnt.dim))
    // dim = cnt.dim+1
    def valueAt(u:DenseVector[Double]):Double = {

      checkDim(u)
      cnt.valueAt(u(0 until dim-1))-u(dim-1)
    }
    def gradientAt(u:DenseVector[Double]):DenseVector[Double] = {

      checkDim(u)
      val grad = DenseVector.zeros[Double](dim)      // dim = cnt.dim+1
      grad(0 until (dim-1)) := cnt.gradientAt(u(0 until (dim-1)))
      grad(dim-1)= -1.0
      grad
    }
    def hessianAt(u:DenseVector[Double]):DenseMatrix[Double] = {

      checkDim(u)
      val hess = DenseMatrix.zeros[Double](dim,dim)
      hess(0 until (dim-1), 0 until (dim-1)) := cnt.hessianAt(u(0 until (dim-1)))
      hess
    }
  }
  /** Version of this constraint for sum of infeasibilities phase I analysis,
    * [boyd], 11.4.1, p580.
    * Recall: this is defined in the context of the full constraint set.
    * We have p new variables s_s,...,s_p and if s_j is the variable corresponding to this
    * constraint, then the new constraint has the form g_j(x)-s_j<=ub_j.
    *
    * @param p number of additional variables s_j (intended application: total number
    *          of constraints).
    * @param j index of additional variable s_j corresponding to this constraint
    *          (zero based).
    */
  def phase_I_SOI(cnt:Constraint,p:Int,j:Int):Constraint = new Constraint(cnt.id+"_phase_I",cnt.dim+p,cnt.ub){

    // u=(x,s_1,...,s_p) is the new variable, note dim = cnt.dim+p
    def isDefinedAt(u:DenseVector[Double]):Boolean = cnt.isDefinedAt(u(0 until cnt.dim))
    /** The variables of the original problem.*/
    def x(u:DenseVector[Double]):DenseVector[Double] = u(0 until dim-p)   // dim(x) = n = cnt.dim = dim-p
    // g_j(x(u))-s_j
    def valueAt(u:DenseVector[Double]):Double = cnt.valueAt(x(u))-u(dim-p+j)

    def gradientAt(u:DenseVector[Double]):DenseVector[Double] = {

      val grad = DenseVector.zeros[Double](dim)
      grad(0 until dim-p) := cnt.gradientAt(x(u))
      grad(dim-p+j)= -1.0
      grad
    }
    def hessianAt(u:DenseVector[Double]):DenseMatrix[Double] = {

      val hess = DenseMatrix.zeros[Double](dim,dim)
      hess(0 until dim-p, 0 until dim-p) := cnt.hessianAt(x(u))
      hess
    }
  }



  /** Turns each constraint cnt: g_j(x)<=ub_j in the list constraints into the constraint
    *         h_j(x,s) = g_j(x)-s_j <= ub_j
    *  for feasibility analysis via the _sum of infeasibilities_ method of
    *  [boyd], 11.4.1, p580. Then adds all the constraints s_j>=0.
    *
    * The independent variable is now u=(x,s), where s=(s_1,...,s_n) and n is the number of
    * constraints in the list constraints. Thus each new constraint has dimension dim+n,
    * where dim is the common dimension of all the constraints in the list.
    *
    * @param cnts list of constraints all in the same dimension n.
    *
    */
  def phase_I_SOI_Constraints(n:Int, cnts:Seq[Constraint]):List[Constraint] = {

    assert(cnts.forall(_.dim==n))
    // modify the constraints in constraints in cnts
    val p = cnts.size     // number of constraints
    var j = -1
    val cnts_SOI = cnts.map(cnt => { j+=1; phase_I_SOI(cnt,p,j) }).toList

    // list of constraints s_j>=0, i.e. -s_j<=0
    val sPositive:List[Constraint] = (0 until p).map(j => new Constraint("s_"+j+">=0",n+p,0.0) {

      def isDefinedAt(u:DenseVector[Double]) = true
      def valueAt(u:DenseVector[Double]):Double = -u(dim-p+j)    // note: dim = n+p
      def gradientAt(u:DenseVector[Double]):DenseVector[Double] =
        DenseVector.tabulate[Double](dim)(k => if(k==dim-p+j) -1.0 else 0.0 )
      // hessian is the zero matrix
      def hessianAt(u:DenseVector[Double]):DenseMatrix[Double] = DenseMatrix.zeros[Double](dim,dim)
    }).toList

    cnts_SOI:::sPositive
  }
}

