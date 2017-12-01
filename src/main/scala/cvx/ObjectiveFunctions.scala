package cvx

import breeze.linalg.{DenseMatrix, DenseVector, norm}

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
  def specialQuadraticObjectiveFunction(x0:DenseVector[Double], R:DenseMatrix[Double]):QuadraticObjectiveFunction = {

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
  def randomSpecialQuadraticObjectiveFunction(x0:DenseVector[Double], R:DenseMatrix[Double]):QuadraticObjectiveFunction = {

    val dim = x0.length
    val R = DenseMatrix.tabulate[Double](dim,dim)((i,j) => -1+2*rng.nextDouble())
    specialQuadraticObjectiveFunction(x0,R)
  }
  /** The function f(x) = r+a'x
    */
  def linearObjectiveFunction(r:Double,a:DenseVector[Double]):QuadraticObjectiveFunction = {

    val n = a.length
    val P = DenseMatrix.zeros[Double](n,n)
    QuadraticObjectiveFunction(n,r,a,P)
  }

  /** The function a dot x = a'x
    */
  def dotProduct(a:DenseVector[Double]):QuadraticObjectiveFunction = linearObjectiveFunction(0.0,a)
}