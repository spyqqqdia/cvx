package cvx

import breeze.linalg.{DenseMatrix, DenseVector, norm, sum}
import breeze.numerics.{abs, pow}

/** Examples of objective functions (test cases).*/
object ObjectiveFunctions {

  val rng = scala.util.Random

  def normSquared(dim:Int) = new ObjectiveFunction(dim) {

    def valueAt(x:DenseVector[Double]) = 0.5*(x dot x)
    def gradientAt(x:DenseVector[Double]) = x
    def hessianAt(x:DenseVector[Double]) = DenseMatrix.eye[Double](dim)
  }

  /** The objective function f(x) = 0.5*||R(x-x0)||².
    * This one has a unique global minimum at x=x0.
    */
  def quadraticObjectiveFunction(x0:DenseVector[Double],R:DenseMatrix[Double]):QuadraticObjectiveFunction = {

    val dim=x0.length
    assert(R.rows==dim,"\nDimension mismatch R.rows = "+R.rows+" not equal to x0.length = "+dim+".\n")

    val Rx0 = R*x0
    val norm_Rx0 = norm(Rx0)
    val r = norm_Rx0*norm_Rx0/2

    val a:DenseVector[Double] = -R.t*Rx0
    val P:DenseMatrix[Double] = R.t*R

    QuadraticObjectiveFunction(dim,r,a,P)
  }
  /** The objective function f(x) = 0.5*||R(x-x0)||² where the matrix R has entries uniformly random
    * in [-1,1].
    * This one has a unique global minimum at x=x0.
    */
  def randomQuadraticObjectiveFunction(x0:DenseVector[Double],R:DenseMatrix[Double]):QuadraticObjectiveFunction = {

    val dim = x0.length
    val R = DenseMatrix.tabulate[Double](dim,dim)((i,j) => -1+2*rng.nextDouble())
    quadraticObjectiveFunction(x0,R)
  }

  /** The L_p norm f(x)=||x||_p raised to power p, i.e.
    * $f(x)=\sum|x_j|^^p$.
    *
    * @param p must be >= 2 to ensure sufficient differentiability.
    */
  def p_norm_p(dim:Int,p:Double):ObjectiveFunction = new ObjectiveFunction(dim) {

    assert(p>=2,"\np-norm needs p>=2 but p="+p+"\n")

    def sgn(u:Double):Double = if(abs(u)<1e-14) 0 else if (u>0) 1.0 else -1.0

    def valueAt(x:DenseVector[Double]):Double = sum(pow(abs(x),p))
    def gradientAt(x:DenseVector[Double]):DenseVector[Double] =
      DenseVector.tabulate[Double](dim)(j => {val s=sgn(x(j)); s*p*pow(s*x(j),p-1)})
    def hessianAt(x:DenseVector[Double]):DenseMatrix[Double] =
      DenseMatrix.tabulate[Double](dim,dim)(
        (i,j) => if(i==j) p*(p-1)*pow(abs(x(i)),p-2) else 0.0
      )
  }
}