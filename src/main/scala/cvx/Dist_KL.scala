package cvx

import breeze.linalg.{DenseMatrix, DenseVector, diag, sum}
import breeze.numerics.log



/**
  * Created by vagrant on 10.10.17.
  *
  * Kullback-Leibler distance
  *
  *      d_KL(x,p) = sum_jp_j\log(p_j/x_j) = c-sum_jp_j\log(x_j)
  *                = c-sum_j\log(x_j)/n
  *
  * from a discrete uniform distribution p on Omega={1,2,...,n}, p_j=1/n; j=1,2,...,n.
  * Here c is the constant
  *
  *      c = -log(n)
  *
  * and even though it is irrelevant in minimization we will not neglect it, since
  * the KL-distance has an information theoretic interpretation.
  */
class Dist_KL(val n:Int) extends ObjectiveFunction(n) {

  override def valueAt(x: DenseVector[Double]): Double = {

    assert(x.length==n,"\nDimension mismatch x.length = "+x.length+"dim(d_KL) = "+n+"\n")
    -sum(log(x))/n - log(n)
  }
  override def gradientAt(x: DenseVector[Double]): DenseVector[Double] =
    DenseVector.tabulate[Double](n)(j => -1.0/x(j)/n)

  override def hessianAt(x: DenseVector[Double]): DenseMatrix[Double] = {

    // diagonal
    val d = DenseVector.tabulate[Double](n)(j => 1.0/(n*x(j)*x(j)))
    diag(d)
  }
}
object Dist_KL {

  def apply(n:Int): ObjectiveFunction ={ assert(n>0); new Dist_KL(n) }

}
