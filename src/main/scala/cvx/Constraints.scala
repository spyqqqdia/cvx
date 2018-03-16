package cvx

import breeze.linalg.{DenseMatrix, DenseVector, norm}
import breeze.numerics.pow

import scala.collection.mutable.ListBuffer

/**
  * Created by oar on 10.10.17.
  *
  * Some constraints we will use repeatedly.
  */
object Constraints {

  val rng = scala.util.Random


  /** Constraint: x_j>=0.
    * * Variables are numbered starting from zero (hence in dimension n,
    * x_{n-1} is the last variable).
    *
    * @param n dimension of problem.
    * @param j coordinate on which the constraint acts, must satisfy j < n.
    *
    */
  def singleCoordinatePositive(n:Int,j:Int):Constraint = {

    assert(j<n && j>=0,
      "\nMust have 0<=j<n but actually m="+j+" and n="+n+"\n"
    )
    val id = "x_" + j + ">0"
    val a = DenseVector.zeros[Double](n)
    a(j) = -1.0
    LinearConstraint(id, n, 0, 0, a)
  }

  /** Constraints: x_m,x_{m+1},...,x_{n-1}>0.
    * Variables are numbered starting from zero (hence in dimension n,
    * x_{n-1} is the last variable).
    *
    * @param n dimension of problem.
    */
  def lastCoordinatesPositive(n:Int,m:Int):List[Constraint] = {

    assert(m<n && m>=0,
      "\nMust have 0<=m<n but actually m="+m+" and n="+n+"\n"
    )
    (m until n).map(j => singleCoordinatePositive(n,j)).toList
  }

  /** Constraints: all x_j>0, j=1,...,n-1.
    *
    * @param n dimension of problem.
    */
  def allCoordinatesPositive(n:Int):List[Constraint] = lastCoordinatesPositive(n,0)


  /** Equality constraint sum(x_j)=1.0 in the form Ax=b.
    * @return (A,b), where A=(1,...,1) and b=(1).
    */
  def sumToOne(n:Int):EqualityConstraint = {

    val A = DenseMatrix.tabulate[Double](1,n)((i,j) => 1.0)
    val b = DenseVector.ones[Double](1)
    EqualityConstraint(A,b)
  }


  /**------------ Expectation equality constraints -----------------*/


  /** Expectation constraint EW=r, where W is a discrete random variable
    * with values W=w_1,w_2,...,w_n and probability distribution P(W=w_j)=x_j.
    * The constraint acting on the probabilities x=(x_j) is the linear
    * constraint
    *                      sum(x_j*w_j)=r,
    *
    * i.e. in the usual matrix form w'x=r.
    *
    * NOTE:
    *
    * A moment constraint of the form EW^p=r is simply a special case of
    * this where the random variable W>0 is replaced with W^p, i.e the vector of
    * values w is replaced with the vector w^^p.
    *
    * Likewise a probability constraint P[E]=r can be expressed as an expectation
    * constraint E[W]=r, where W is the indicator function W=1_E, i.e. the
    * corresponding vector w is 0-1 valued with
    *
    *    w_j=1 if j\in E and w_j=0 otherwise.
    *
    * @param w values of the random variable W.
    * @return constraint EW=r
    */
  def expectation_eq_r(w:DenseVector[Double], r: Double): EqualityConstraint = {

    val n = w.length
    val A = DenseMatrix.tabulate[Double](1,n)((i,j) => w(j))
    val b = DenseVector.fill(1){ r }
    EqualityConstraint(A,b)
  }


  /**------------ Expectation inequality constraints -----------------*/

  /** Expectation constraint EW<r, where W is a discrete random variable
    * with values W=w_1,w_2,...,w_n and probability distribution P(W=w_j)=x_j.
    * The constraint acts on the probabilities x=(x_j) and is the linear
    * constraint
    *                      sum(x_j*w_j)<r,
    *
    * i.e. in the usual matrix form w'x<r.
    *
    * NOTE:
    *
    * A moment constraint of the form EW^p<r is simply a special case of
    * this where the random variable W>0 is replaced with W^p, i.e the vector of
    * values w is replaced with the vector w^^p.
    *
    * Likewise a probability constraint P[E]<r can be expressed as an expectation
    * constraint E[W]=r, where W is the indicator function W=1_E, i.e. the
    * corresponding vector w is 0-1 valued with
    *
    *    w_j=1 if j\in E and w_j=0 otherwise.
    *
    * An expectation constraint of the form E[W]>r is simply rewritten as
    * E[-W]<-r, i.e. we need to replace r with -r and the vector w with -w.
    *
    * In particular a probability constraint P[E]>r is rewritten as
    * E[-1_E]<-r.
    *
    * @param w values of the random variable W.
    * @return constraint EW=r
    */
  def expectation_lt_r(w:DenseVector[Double], r: Double, id:String): LinearConstraint = {

    val dim = w.length
    LinearConstraint(id,dim,r,0,w)
  }

  /** The constraint a'(x-x0)<=e. This is satisfied by the point x=x0 if and
    * only if e>=0.
    */
  def linearIneqConstraint(id:String,x0:DenseVector[Double],a:DenseVector[Double],e:Double):LinearConstraint = {

    val dim:Int = x0.length
    assert(a.length==dim,"\nLengths of vectors x0 and v not equal, x0.length = "+dim+", v.length = "+a.length+".\n")
    LinearConstraint(id,dim,e,-(a dot x0),a)
  }
  /** The constraint a'(x-x0)<=e with a vector a with uniformly random components in [-1,1].
    * This is satisfied by the point x=x0 if and only if e>=0.
    */
  def randomLinearIneqConstraint(id:String,x0:DenseVector[Double],e:Double):LinearConstraint = {

    val a = MatrixUtils.randomVector(x0.length,-1.0,1.0)
    linearIneqConstraint(id,x0,a,e)
  }


  /** The constraint 05*||R(x-x0)||²<=e. This is infeasible if e<0, has only one solution x=x0 if e=0 (then expect
    * problems) and defines a nonempty open convex set containing x0 if e>0.
    */
  def quadraticIneqConstraint(id:String,x0:DenseVector[Double],R:DenseMatrix[Double],e:Double):QuadraticConstraint = {

    val dim=x0.length
    assert(R.rows==dim,"\nDimension mismatch R.rows = "+R.rows+" not equal to x0.length = "+dim+".\n")

    val Rx0 = R*x0
    val norm_Rx0 = norm(Rx0)
    val r = norm_Rx0*norm_Rx0/2

    val a:DenseVector[Double] = -R.t*Rx0
    val P:DenseMatrix[Double] = R.t*R

    QuadraticConstraint(id,dim,e,r,a,P)
  }
  /** The constraint 05*||R(x-x0)||²<=e where the matrix R has entries uniformly random in [-1,1].
    * This is infeasible if e<0, has only one solution x=x0 if e=0 (then expect problems) and defines
    * a nonempty open convex set containing x0 if e>0.
    */
  def randomQuadraticIneqConstraint(id:String,x0:DenseVector[Double],e:Double):QuadraticConstraint = {

    val dim = x0.length
    val R = MatrixUtils.randomMatrix(dim,dim,-1.0,1.0)
    quadraticIneqConstraint(id,x0,R,e)
  }

  /** An equality constraint Ax=b satisfied by x=x0, where the matrix A has m rows
    * with entries uniformly random in [-1,1] (and then of course b=Ax0).
    *
    * @param x0: must have length >= 2.
    */
  def randomEqualityConstraint(id:String,x0:DenseVector[Double],m:Int):EqualityConstraint = {

    val n = x0.length
    assert(n>=2,"\nx0.length = "+n+" not >= 2.\n")
    val A = MatrixUtils.randomMatrix(m,n,-1.0,1.0)
    val b = A*x0
    EqualityConstraint(A,b)
  }

  /*******************************************************************/
  /********************** Special constraints ************************/
  /*******************************************************************/

  /** List of linear inequality constraints r+Qx <= ub, where here "<="
    * is interpreted component by component.
    * In other words this is the list of constraints
    *    r_i + row_i(Q)'x <= ub_i,  i < A.rows.
    */
  def linearInequalityConstraints(
    r:DenseVector[Double], Q:DenseMatrix[Double], ub:DenseVector[Double]
  ):List[LinearConstraint] = {

    assert(r.length==Q.rows && r.length==ub.length)
    val res0 = ListBuffer[LinearConstraint]()
    val n = Q.rows
    for(i <- 0 until n){

      val id = "LinCnt"+i
      val dim = Q.cols
      val a:DenseVector[Double] = Q(i,::).t
      val cnt_j = LinearConstraint(id,dim,ub(i),r(i),a)
      res0 += cnt_j
    }
    res0.toList
  }

  /** The constraint |x_0|+...+|x_{n-1}| <= ub, where n is the full
    * dimension of the variable x (all x_j involved) resolved as a list
    * of constraints
    *  a_0x_0 +...+ a_{n-1}x_{n-1} <= ub,
    * where the coefficient vector (a_0,...,a_{n-1}) runs through all
    * combinations of signs +1,-1.
    * Note: this list has length pow(2,n) so can only be applied to
    * small n!!
    */
  def sumAbsoluteValuesBoundedBy(n:Int,ub:Double):List[LinearConstraint] = {

    val m:Int = pow(2,n)
    val r = DenseVector.zeros[Double](m)
    val Q = MatrixUtils.signCombinationMatrix(n)
    val ubs = DenseVector.tabulate[Double](m)(j => ub)
    linearInequalityConstraints(r,Q,ubs)
  }
  /** The constraint |x_p|+...+|x_{q-1}| <= ub acting on a vector
    * of variables x=(x_0,...,x_{n-1}).
    *
    * This is resolved as a list of constraints
    *  a_px_p +...+ a_{q-1}x_{q-1} <= ub,
    * where the coefficient vector (a_p,...,a_{q-1}) runs through all
    * combinations of signs +1,-1.
    * Note: this list has length pow(2,q-p) so can only be applied to
    * small values of q-p!!
    */
  def sumAbsoluteValuesBoundedBy(n:Int,p:Int,q:Int,ub:Double):List[LinearConstraint] = {

    val m:Int = pow(2,q-p)
    assert(q-p<=16,"\nTrying to allocate too many constraints, number = "+m+".\n")
    val r = DenseVector.zeros[Double](m)
    val Q = MatrixUtils.signCombinationMatrix(n,p,q)
    val ubs = DenseVector.tabulate[Double](m)(j => ub)
    linearInequalityConstraints(r,Q,ubs)
  }

  /** The constraints |x_j|<=ub(j), j=0,...,n-1, resolved as
    *  x_j<=ub_j and -x_j<=ub_j, j=0,...,n-1.
    *
    * @param n dimension of variable x
    * @param ub vector of upper bounds for |x_j|.
    */
  def absoluteValuesBoundedBy(n:Int,ub:DenseVector[Double]):List[LinearConstraint] = {

    assert(ub.length==n && ub.forall(_>=0))
    val res0 = ListBuffer[LinearConstraint]()
    for(j <- 0 until n){

      val cnts_j = sumAbsoluteValuesBoundedBy(n,j,j+1,ub(j))   // |x_j| <= ub(j)
      res0++=cnts_j
    }
    res0.toList
  }
  /** This is the constraint 0.5*||x||²<=ub.
    */
  def oneHalfNorm2BoundedBy(n:Int,ub:Double):Constraint = {

    val id = "x dot x <= "+ub
    new Constraint(id,n,ub) {

      def isDefinedAt(x: DenseVector[Double]): Boolean = true
      def valueAt(x:DenseVector[Double]):Double = (x dot x)/2
      def gradientAt(x:DenseVector[Double]):DenseVector[Double] = x
      def hessianAt(x:DenseVector[Double]):DenseMatrix[Double] = DenseMatrix.eye[Double](x.length)
    }
  }
}