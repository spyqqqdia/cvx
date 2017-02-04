package cvx

import breeze.linalg.{DenseMatrix, DenseVector, _}

/** Objective function (should be convex, twice continuously differentiable).
 *  Provides value, gradient and hessian of self.
 */
abstract class ObjectiveFunction(val dim:Int){

    def valueAt(x:DenseVector[Double]):Double
	def gradientAt(x:DenseVector[Double]):DenseVector[Double]
    def hessianAt(x:DenseVector[Double]):DenseMatrix[Double]

    def checkDim(x:DenseVector[Double]):Unit =
        assert(x.length==dim,"Dimension mismatch: x.length = "+x.length+", dim="+dim)

    /** This objective function restricted to values of the original variable x of the form x=z+Fu
      * now viewed as a constraint on the variable u in dimension dim-p, where p is the rank
      * of F.
      * F is assumed to be of full rank and this condition is not checked.
      * The intended application is the case where the x=z+Fu are the solutions of
      * equality constraints Ax=b.
      *
      * In general this reduction will induce significant matrix multiplication overhead.
      * Elimination of equality constraints Ax=b by reduction will only be used in special
      * cases where this overhead can be avoided.
      *
      * @param z a vector of dimension dim-p (intended: special solution of Ax=b)
      * @param F a nxp matrix (intended: p = number of equality constraints)
      */
    def reduced(z:DenseVector[Double],F:DenseMatrix[Double]):ObjectiveFunction = {

        val  rDim = dim-F.cols
        new ObjectiveFunction(rDim){

            override def valueAt(u:DenseVector[Double]) = super.valueAt(z+F*u)
            override def gradientAt(u:DenseVector[Double]) = F.t*super.gradientAt(z+F*u)
            override def hessianAt(u:DenseVector[Double]) = (F.t*super.hessianAt(z+F*u))*F
        }
    }
}



