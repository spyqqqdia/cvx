package cvx

import breeze.linalg.{DenseMatrix, DenseVector, _}


class StrictlyFeasibleSet(override val dim:Int,val constraints:List[Constraint])
extends ConvexSet(dim) {

    // this is called often in line search, may have to do this more efficiently
	def isInSet(x:DenseVector[Double]):Boolean =
	    constraints forall (cnt => cnt.isSatisfiedStrictly(x))
	def samplePoint = null
}
object StrictlyFeasibleSet {

    // factory method
	def apply(dim:Int, constraints:List[Constraint]):StrictlyFeasibleSet = {
	
	    // check that all constraints have the same dimension
		assert(constraints forall (cnt => cnt.dim==dim))
		new StrictlyFeasibleSet(dim,constraints)
	}
}