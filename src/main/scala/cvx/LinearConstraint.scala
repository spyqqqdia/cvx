package cvx

import breeze.linalg.{DenseMatrix, DenseVector}

/** Affine inequality constraint r + a'x <= ub
  */
class LinearConstraint(
                          override val id:String,
                          override val dim:Int,
                          override val ub:Double,
                          val r:Double,
                          val a:DenseVector[Double]
                      )
    extends Constraint(id,dim,ub){

    if(a.length!=dim){
        val msg = "Vector a must be of dimension "+dim+" but length(a) "+a.length
        throw new IllegalArgumentException(msg)
    }

    def valueAt(x:DenseVector[Double]) = { checkDim(x); r + (a dot x)	}
    def gradientAt(x:DenseVector[Double]) = { checkDim(x); a }
    def hessianAt(x:DenseVector[Double]) = { checkDim(x); DenseMatrix.zeros[Double](dim,dim) }

    /** This constraint restricted to values of the original variable x of the form x=z+Fu
      * now viewed as a constraint on the variable u in dimension dim-p, where p is the rank
      * of F.
      * F is assumed to be of full rank and this condition is not checked.
      * The intended application is the case where the x=z+Fu are the solutions of
      * equality constraints Ax=b.
      *
      * The result is another linear constraint, see docs/cvx_notes.pdf, p4, equation (5).
      *
      * @param z a vector of dimension dim-p (intended: special solution of Ax=b)
      * @param F a nxp matrix (intended: p = number of equality constraints)
      */
    override def reduced(z:DenseVector[Double], F:DenseMatrix[Double]) = {

        val rID = id + "_reduced"
        val rDim = dim - F.cols
        val rr = valueAt(z)
        val ra = F.t * a
        LinearConstraint(rID, rDim, ub, rr, ra)
    }
}
object LinearConstraint {

    /** Constraint r + (a dot x) <= ub. */
    def apply(id:String,dim:Int,ub:Double,r:Double,a:DenseVector[Double]) = new LinearConstraint(id,dim,ub,r,a)
}