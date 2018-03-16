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

      override def valueAt(u:DenseVector[Double]):Double = self.valueAt(z+F*u)
      override def gradientAt(u:DenseVector[Double]):DenseVector[Double] = F.t*self.gradientAt(z+F*u)
      override def hessianAt(u:DenseVector[Double]):DenseMatrix[Double] = (F.t*self.hessianAt(z+F*u))*F
    }
  }
  /** Objective function g(x,s) = f(x)+Ks for optimization of the globally relaxed
    * problem with the PrimalDualSolver. See docs/primaldual.pdf.
    * This objective function has one additional slack variable s.
    */
  def forGloballyRelaxedProblem(K:Double): ObjectiveFunction = {

    require(K>=0,"\nK must be positive but K = "+K+".\n")
    val n = dim
    new ObjectiveFunction(n + 1) {

      override def valueAt(x: DenseVector[Double]): Double =
        self.valueAt(x(0 until n)) + K*x(n)

      override def gradientAt(x: DenseVector[Double]): DenseVector[Double] = {

        val grad_fx = self.gradientAt(x(0 until n))
        DenseVector.tabulate[Double](n + 1)(j => if (j < n) grad_fx(j) else K)
      }

      override def hessianAt(x: DenseVector[Double]): DenseMatrix[Double] = {

        val res = DenseMatrix.zeros[Double](n + 1, n + 1)
        res(0 until n,0 until n) := self.hessianAt(x(0 until n))
        res
      }
    }
  }
  /** Objective function g(x,s) = f(x)+(K dot s) for optimization of the
    * globally relaxed problem with the PrimalDualSolver.
    * See docs/primaldual.pdf.
    * This objective function has one additional slack variable s for each
    * inequality constraint of the original problem.
    */
  def forLocallyRelaxedProblem(K:Vector[Double]): ObjectiveFunction = {

    require(K.forall(_>=0),
      "\nK must have positive components but K = "+K+".\n"
    )
    val n = dim
    val m = K.length      // number of slack variables s_j
    new ObjectiveFunction(n + m) {

      override def valueAt(x: DenseVector[Double]): Double =
        self.valueAt(x(0 until n)) + (K dot x(n until n+m))

      override def gradientAt(x: DenseVector[Double]): DenseVector[Double] = {

        val grad_fx = self.gradientAt(x(0 until n))
        DenseVector.tabulate[Double](n + m)(j => if (j < n) grad_fx(j) else K(j-n))
      }

      override def hessianAt(x: DenseVector[Double]): DenseMatrix[Double] = {

        val res = DenseMatrix.zeros[Double](n + m, n + m)
        res(0 until n,0 until n) := self.hessianAt(x(0 until n))
        res
      }
    }
  }
}


