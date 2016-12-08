package cvx

import breeze.linalg.{DenseMatrix, DenseVector, _}


/** General inequality constraint of the form g(x)<=ub.
  *
  * @param dim: dimension of independent variable x
 * @param ub: upper bound on g.
 */
abstract class Constraint(val id:String, val dim:Int, val ub:Double){

    def valueAt(x:DenseVector[Double]):Double
	def gradientAt(x:DenseVector[Double]):DenseVector[Double]
    def hessianAt(x:DenseVector[Double]):DenseMatrix[Double]
	
	def checkDim(x:DenseVector[Double]):Unit =
        assert(x.length==dim,"Dimension mismatch: x.length = "+x.length+", dim="+dim)	
	def isSatisfied(x:DenseVector[Double]):Boolean = valueAt(x)<=ub
	def isSatisfiedStrictly(x:DenseVector[Double]):Boolean = valueAt(x)<ub
	/** @return |g(x)-ub|<tol. */
	def isActive(x:DenseVector[Double], tol:Double=1e-12):Boolean = Math.abs(valueAt(x)-ub)<tol
	/** @return ub-g(x).*/
	def margin(x:DenseVector[Double]):Double = ub-valueAt(x)	 
}
object Constraint {
	
	/** Constraint g(x)-s<=ub for phase I feasibility analysis of the given constraint
	 *  cnt: g(x)<=ub via the basic phase I feasibility method, [boyd], 11.4.1, p579.
	 * The independent variable is now u=(x,s) and the dimension is cnt.dim+1.
	 */
	def phase_I_Basic(cnt:Constraint):Constraint = new Constraint(cnt.id+"_phase_I",cnt.dim+1,cnt.ub){
	
	    def valueAt(u:DenseVector[Double]):Double = { 
		
		    checkDim(u)
			cnt.valueAt(u(0 until dim))-u(dim)
		}
        def gradientAt(u:DenseVector[Double]):DenseVector[Double] = {

		    checkDim(u)
			val grad = DenseVector.zeros[Double](dim)      // dim = cnt.dim+1
			grad(0 until (dim-1)) := cnt.gradientAt(u(0 until (dim-1)))
			grad(dim-1)= 1.0
			grad
		}
		def hessianAt(u:DenseVector[Double]):DenseMatrix[Double] = {
		
		    checkDim(u)
			val hess = DenseMatrix.zeros[Double](dim,dim)
			hess(0 until (dim-1), 0 until (dim-1)) := cnt.hessianAt(u(0 until (dim-1)))
		}
	}
	/** Turns each constraint cnt: g_j(x)<=ub_j in the list constraints into the constraint
	 *  g_j(x,s)-s<=ub_j for feasibility analysis via the basic phase I feasibility method of
	 *  [boyd], 11.4.1, p579.
	 * The independent variable is now u=(x,s) of dimension dim+1,
	 * where dim is the common dimension of all the constraints in the list.
	 *
	 * @param constraints list of constraints all in the same dimension dim.
     *
     */
    def phase_I_Basic(constraints:List[Constraint]):List[Constraint] = constraints.map(cnt => phase_I_Basic(cnt))
	
	
	
    /** Turns each constraint cnt: g_j(x)<=ub_j in the list constraints into the constraint
	 *  g_j(x,s)-s_j<=ub_j for feasibility analysis via the _Sum Of Infeasibilities_ method of
	 *  [boyd], 11.4.1, p580. The adds all the constraints s_j>=0.
	 *
	 * The independent variable is now u=(x,s), where s=(s_1,...,s_n) and n is the number of
	 * constraints in the list constraints. Thus each new constraint has dimension dim+n,
	 * where dim is the common dimension of all the constraints in the list.
	 *
	 * @param constraints list of constraints all in the same dimension dim.
     *
     */
    def phase_I_SOI(constraints:List[Constraint]):List[Constraint] = {

        val n = constraints.length
		val dim = constraints(0).dim   // common dimension of all constraints in list
		assert(constraints.forall(cnt => cnt.dim==dim))
		// list of constraints g_j(x)-s_j <= ub_j
		var j = -1      // number of constraint in list
        val cnts_SOI = constraints.map(cnt => {

				j = j + 1
				new Constraint(cnt.id + "phase_I_SOI", dim + n, cnt.ub) {

					def valueAt(u: DenseVector[Double]): Double = {

						checkDim(u)
						cnt.valueAt(u(0 until dim)) - u(dim + j)
					}

					def gradientAt(u: DenseVector[Double]): DenseVector[Double] = {

						checkDim(u)
						val grad = DenseVector.zeros[Double](dim)
						grad(0 until (dim - 1)) := cnt.gradientAt(u(0 until (dim - 1)))
						grad(dim - 1) = 1.0
						grad
					}

					def hessianAt(u: DenseVector[Double]): DenseMatrix[Double] = {

						checkDim(u)
						val hess = DenseMatrix.zeros[Double](dim, dim)
						hess(0 until (dim - 1), 0 until (dim - 1)) := cnt.hessianAt(u(0 until (dim - 1)))
					}
				}
		}) // end map
		
		// list of constraints s_j>=0, i.e. -s_j<=0
		val sPositive:List[Constraint] = (0 until n).map(j => new Constraint("s_"+j+">=0",dim+n,0.0) {
		
		    def valueAt(u:DenseVector[Double]):Double = { checkDim(u); -u(dim+j) }
			  
            def gradientAt(u:DenseVector[Double]):DenseVector[Double] = {

		        checkDim(u)
			    val grad = DenseVector.zeros[Double](dim+n)      
			    grad(dim+j) = -1.0
			    grad
		    }  
		    def hessianAt(u:DenseVector[Double]):DenseMatrix[Double] = {
		
		        checkDim(u)
			    DenseMatrix.zeros[Double](dim+n,dim+n)
		    }
		}).toList // end map

		cnts_SOI:::sPositive
    }	
}







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

    def valueAt(x:DenseVector[Double]) = { checkDim(x); r + (a dot x)	}
    def gradientAt(x:DenseVector[Double]) = { checkDim(x); a }
	def hessianAt(x:DenseVector[Double]) = { checkDim(x); DenseMatrix.zeros[Double](dim,dim) }
} 
/** Quadratic constraint r + a'x + (1/2)*x'Qx <= ub, where Q is a symmetric matrix.
 */
class QuadraticConstraint(
    override val id:String, 
	override val dim:Int, 
	override val ub:Double,
	val r:Double,
	val a:DenseVector[Double],
	val Q:DenseMatrix[Double]
) 
extends Constraint(id,dim,ub){

    MatrixUtils.checkSymmetric(Q,1e-13)
	
	def valueAt(x:DenseVector[Double]) = { checkDim(x); r + (a dot x) + (x dot (Q*x))/2 }
    def gradientAt(x:DenseVector[Double]) = { checkDim(x); a+Q*x }
	def hessianAt(x:DenseVector[Double]) = { checkDim(x); Q }
} 



