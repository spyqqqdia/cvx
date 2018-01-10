package cvx

import breeze.linalg.{DenseMatrix, DenseVector, _}

/** Objective function (should be convex, twice continuously differentiable).
  *  Provides value, gradient and hessian of self.
  */
abstract class ObjectiveFunction(val dim:Int){

  self: ObjectiveFunction =>

  def valueAt(x:DenseVector[Double]):Double
  def gradientAt(x:DenseVector[Double]):DenseVector[Double]
  def hessianAt(x:DenseVector[Double]):DenseMatrix[Double]

  def checkDim(x:DenseVector[Double]):Unit =
    assert(x.length==dim,"Dimension mismatch: x.length = "+x.length+", dim="+dim)

  /** The objective function h(u) = f(z+Fu), where f=f(x) is _this_ objective
    * function. In short _this_ objective function under change of variables
    * x = z + Fu.
    *
    * @param F a nxp matrix
    * @param z a vector of dimension F.rows
    */
  def affineTransformed(z:DenseVector[Double],F:DenseMatrix[Double]):ObjectiveFunction = {

    val  rDim = dim-F.cols
    new ObjectiveFunction(rDim){

      override def valueAt(u:DenseVector[Double]) = self.valueAt(z+F*u)
      override def gradientAt(u:DenseVector[Double]) = F.t*self.gradientAt(z+F*u)
      override def hessianAt(u:DenseVector[Double]) = (F.t*self.hessianAt(z+F*u))*F
    }
  }
}


