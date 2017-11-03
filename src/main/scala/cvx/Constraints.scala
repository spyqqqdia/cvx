package cvx

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.pow

/**
  * Created by vagrant on 10.10.17.
  *
  * Some constraints we will use repeatedly.
  */
object Constraints {

  /** Constraints: all x_j>0, j=1,...,n.
    *
    * @param n dimension of problem.
    */
  def allCoordinatesPositive(n:Int):List[Constraint] =  (0 until n).map( j => {

    val  id = "x_"+j+">0"
    val a = DenseVector.zeros[Double](n)
    a(j)= -1.0
    LinearConstraint(id,n,0,0,a)
  }).toList

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
    * values w is replaced with the vector pow(w,p).
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
    * values w is replaced with the vector pow(w,p).
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


}