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
	def isSatisfiedStrictly(x:DenseVector[Double]):Boolean = valueAt(x)*(1+3e-16) < ub
	/** @return |g(x)-ub|<tol. */
	def isActive(x:DenseVector[Double], tol:Double=1e-12):Boolean = Math.abs(valueAt(x)-ub)<tol
	/** @return ub-g(x).*/
	def margin(x:DenseVector[Double]):Double = ub-valueAt(x)


}


object Constraint {


    /** Version of constraint cnt for basic phase I analysis, [boyd], 11.4.1, p579.
      * Recall: one new variable s with upper bound zero and g_j(x)<=ub replaced with
      * g_j(x)-s<=ub.
      */
    def phase_I(cnt:Constraint):Constraint = new Constraint(cnt.id+"_phase_I",cnt.dim+1,cnt.ub){

        // dim = cnt.dim+1
        def valueAt(u:DenseVector[Double]):Double = {

            checkDim(u)
            cnt.valueAt(u(0 until dim-1))-u(dim-1)
        }
        def gradientAt(u:DenseVector[Double]):DenseVector[Double] = {

            checkDim(u)
            val grad = DenseVector.zeros[Double](dim)      // dim = cnt.dim+1
            grad(0 until (dim-1)) := cnt.gradientAt(u(0 until (dim-1)))
            grad(dim-1)= -1.0
            grad
        }
        def hessianAt(u:DenseVector[Double]):DenseMatrix[Double] = {

            checkDim(u)
            val hess = DenseMatrix.zeros[Double](dim,dim)
            hess(0 until (dim-1), 0 until (dim-1)) := cnt.hessianAt(u(0 until (dim-1)))
            hess
        }
    }
    /** Version of this constraint for sum of infeasibilities phase I analysis,
      * [boyd], 11.4.1, p580.
      * Recall: this is defined in the context of the full constraint set.
      * We have p new variables s_s,...,s_p and if s_j is the variable corresponding to this
      * constraint, then the new constraint has the form g_j(x)-s_j<=ub_j.
      *
      * @param p number of additional variables s_j (intended application: total number
      *          of constraints).
      * @param j index of additional variable s_j corresponding to this constraint
      *          (zero based).
      */
    def phase_I_SOI(cnt:Constraint,p:Int,j:Int):Constraint = new Constraint(cnt.id+"_phase_I",cnt.dim+p,cnt.ub){

        // u=(x,s_1,...,s_p) is the new variable, note dim = cnt.dim+p

        /** The variables of the original problem.*/
        def x(u:DenseVector[Double]):DenseVector[Double] = u(0 until dim-p)   // dim(x) = n = cnt.dim = dim-p

        def valueAt(u:DenseVector[Double]):Double = cnt.valueAt(x(u))-u(dim-p+j)

        def gradientAt(u:DenseVector[Double]):DenseVector[Double] = {

            val grad = DenseVector.zeros[Double](dim)
            grad(0 until dim-p) := cnt.gradientAt(x(u))
            grad(dim-p+j)= -1.0
            grad
        }
        def hessianAt(u:DenseVector[Double]):DenseMatrix[Double] = {

            val hess = DenseMatrix.zeros[Double](dim,dim)
            hess(0 until dim-p, 0 until dim-p) := cnt.hessianAt(x(u))
            hess
        }
    }



    /** Turns each constraint cnt: g_j(x)<=ub_j in the list constraints into the constraint
	 *  g_j(x,s)-s_j<=ub_j for feasibility analysis via the _sum of infeasibilities_ method of
	 *  [boyd], 11.4.1, p580. Then adds all the constraints s_j>=0.
	 *
	 * The independent variable is now u=(x,s), where s=(s_1,...,s_n) and n is the number of
	 * constraints in the list constraints. Thus each new constraint has dimension dim+n,
	 * where dim is the common dimension of all the constraints in the list.
	 *
	 * @param cnts list of constraints all in the same dimension n.
     *
     */
    def phase_I_SOI_Constraints(n:Int, cnts:Seq[Constraint]):List[Constraint] = {

        assert(cnts.forall(_.dim==n))
		// modify the constraints in constraints in cnts
		val p = cnts.size     // number of constraints
		var j = -1
		val cnts_SOI = cnts.map(cnt => { j+=1; phase_I_SOI(cnt,p,j) }).toList

		// list of constraints s_j>=0, i.e. -s_j<=0
		val sPositive:List[Constraint] = (0 until p).map(j => new Constraint("s_"+j+">=0",n+p,0.0) {
		
		    def valueAt(u:DenseVector[Double]):Double = -u(dim-p+j)    // note: dim = n+p
			  
            def gradientAt(u:DenseVector[Double]):DenseVector[Double] =
				DenseVector.tabulate[Double](dim)(k => if(k==dim-p+j) -1.0 else 0.0 )
            // hessian is the zero matrix
		    def hessianAt(u:DenseVector[Double]):DenseMatrix[Double] = DenseMatrix.zeros[Double](dim,dim)
		}).toList

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
object LinearConstraint {

	/** Constraint r + (a dot x) <= ub. */
	def apply(id:String,dim:Int,ub:Double,r:Double,a:DenseVector[Double]) = new LinearConstraint(id,dim,ub,r,a)
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
object QuadraticConstraint {

	/** Constraint r + (a dot x) + x'Qx <= ub. */
	def apply(id:String,dim:Int,ub:Double,r:Double,a:DenseVector[Double], Q:DenseMatrix[Double]) =
		new QuadraticConstraint(id,dim,ub,r,a,Q)
}


/** A point strictly satisfying all constraints in a set of constraints.*/
trait FeasiblePoint {

	def feasiblePoint:DenseVector[Double]
}


/** Holder for a sequence of constraints with some additional methods.
  */
abstract class ConstraintSet(val dim:Int, val constraints:Seq[Constraint]) {

	assert(constraints.forall(cnt => cnt.dim == dim))

	/** A point x where all constraints in a set of constraints are defined,
	  * i.e. the functions g_j(x) defining the constraints a g_j(x)<=ub_j are all defined.
	  * The constraints do not have to be not satisfied at the point x.
	  * Will be used as starting point for phase_I feasibility analysis.
	  */
	def pointWhereDefined: DenseVector[Double]

	def numConstraints = constraints.size

	/** @return null. */
	def samplePoint = null

	def isSatisfiedStrictlyBy(x:DenseVector[Double]):Boolean = constraints.forall(_.isSatisfiedStrictly(x))

	/** Set of points where the constraints are satisfied strictly. */
	def strictlyFeasibleSet = new ConvexSet(dim) {

		def isInSet(x: DenseVector[Double]) = {

			assert(x.length == dim)
			constraints.forall(cnt => cnt.isSatisfiedStrictly(x))
		}
	}
    /** Turn this constraint set into a constraint set with a fesible point.*/
	def addFeasiblePoint(x0:DenseVector[Double]): ConstraintSet with FeasiblePoint = {

        // check if x0 is strictly feasible
		assert(x0.length==dim,"Feasible point does not have the right dimension")
		assert(constraints.forall(_.isSatisfiedStrictly(x0)))
		new ConstraintSet(dim,constraints) with FeasiblePoint {

			def pointWhereDefined = x0
		    def feasiblePoint = x0
		}
	}

	/** Perform a sum of infeasibilities (SOI) feasibility analysis on this constraint set
	  * with the feasibility solver [BarrierSolver.phase_I_Solver_SOI].
	  *
	  * If g_j(x) <= ub_j are the constraints in this constraint set this will
	  * minimize the function f(x,s)=s_1+...+s_p subject to s_j>=0 and g_j(x)-s_j <= ub_j.
	  *
	  * This gives an indication which constraints might be feasible (s_j=0) although the
	  * solution (x,s) of the feasibility solver is not guaranteed to yield a point x
	  * satisfying all or even many constraints (due to the constraint s_j>=0).
	  * However often x does solve many or even all constraints. This is simply a matter of luck.
	  *
	  * To get around this problem replace the upper bounds ub_j in the constraints with
	  * ub_j-epsilon and run the SOI feasibility analysis on this new constraint set.
	  * Then the point x in the solution (x,s) of the feasibility solver will satisfy the
	  * original constraint g_j(x) <= ub_j strictly whenever s_j<epsilon.
	  *
	  * @param pars solver parameters for the feasiblity solver ([BarrierSolver.phase_I_Solver_SOI]).
	  * @return a feasibility report containing both the vectors x and s above as well as other information.
      */
	def doSOIAnalysis(pars:SolverParams):FeasibilityReport = BarrierSolver.phase_I_Analysis_SOI(this,pars)
}


object ConstraintSet {

	/** Factory function,
	  *
	  * @param dim common dimension of all constraints in the list constraints
	  * @param constraints list of constraints
	  * @param x0 point at which all constraints are defined.
      * @return
      */
	def apply(dim:Int, constraints:Seq[Constraint],x0:DenseVector[Double]) =
		new ConstraintSet(dim,constraints){ def pointWhereDefined = x0 }

	//--- Objective function and constraints for basic feasibility analysis ---//

	/** Objective function for basic feasibility analysis, see [boyd], 11.4.1, p579.
	  * Recall: one new variable s and the function is f(x,s)=s.
      */
	def phase_I_ObjectiveFunction(cnts:ConstraintSet):ObjectiveFunction = {

		val n = cnts.dim
		new ObjectiveFunction(n+1){

			def valueAt(x:DenseVector[Double]):Double = x(n)
			def gradientAt(x:DenseVector[Double]):DenseVector[Double] =
				DenseVector.tabulate[Double](n+1)(j => if (j<n) 0 else 1 )

			/** Is the zero matrix in dimnsion 1+cnts.dim.*/
			def hessianAt(x:DenseVector[Double]):DenseMatrix[Double] = DenseMatrix.zeros[Double](n+1,n+1)
		}
	}


	/** Set of these constraints for basic phase I analysis, [boyd], 11.4.1, p579.
	  * Recall: one new variable s and each constraint g_j(x) <= ub_j replaced with g_j(x)-s <= ub_j.
	  *
	  * For these we can always find a point at which all these constraints are satisfied.
	  * (member function [feasiblePoint]).
	  */
	def phase_I_Constraints(cnts:ConstraintSet):ConstraintSet with FeasiblePoint = {

		val n = cnts.dim
		new ConstraintSet(n+1,cnts.constraints.map(cnt => Constraint.phase_I(cnt))) with FeasiblePoint {

			val x0:DenseVector[Double] = cnts.pointWhereDefined
			val y0:Double = cnts.constraints.map(_.valueAt(x0)).max  // max_j g_j(x0)
			def feasiblePoint = DenseVector.tabulate[Double](dim)(j => if (j<dim-1) x0(j) else 1+y0 )
			def pointWhereDefined = feasiblePoint
		}
	}


	//--- Objective function and constraints for sum of infeasibilities feasibility analysis ---//

	/** Objective function for sum of infeasibilities feasibility analysis, see
	  * [boyd], 11.4.1, p579.
	  * Recall: one new variable s_j for each constraint g_j(x) <= ub_j and the objective
	  * function is f(x,s) = s_1+...+s_p, where p is the number of constraints.
	  */
	def phase_I_SOI_ObjectiveFunction(cnts:ConstraintSet):ObjectiveFunction = {

		val n = cnts.dim
		val p = cnts.numConstraints
		val dim = n + p

		new ObjectiveFunction(dim){

			def valueAt(u:DenseVector[Double]):Double = sum(u(n until dim))
			def gradientAt(x:DenseVector[Double]):DenseVector[Double] =
				DenseVector.tabulate[Double](dim)(j => if (j<n) 0.0 else 1.0)
			/** Is the zero matrix in dimnsion 1+cnts.dim.*/
			def hessianAt(x:DenseVector[Double]):DenseMatrix[Double] = DenseMatrix.zeros[Double](dim,dim)
		}
	}



	/** Set of constraints cnts modified for sum of infeasibilities phase I analysis with added
	  * positivity constraints for the additional variables, see [Constraint..phase_I_SOI_Constraints].
	  * [boyd], 11.4.1, p580.
	  * Recall: one new variable s_j for each constraint g_j(x) <= ub_j which is then replaced with
	  * g_j(x)-s<=ub.
	  *
	  * For these we can always find a point at which all these constraints are satisfied.
	  * The member function _samplePoint_ returns such a point.
	  */
	def phase_I_SOI_Constraints(cnts:ConstraintSet):ConstraintSet with FeasiblePoint =  {

        val n = cnts.dim
		val p = cnts.numConstraints
		val cnts_SOI = Constraint.phase_I_SOI_Constraints(n,cnts.constraints)

		val x = cnts.pointWhereDefined

		new ConstraintSet(n+p,cnts_SOI) with FeasiblePoint {

			// new variable u = (x,s) = (x_1,...,x_n,s_1,...,s_p), where n=dim
			// feasibility if s_j > g_j(x) where g_j(x) <= ub_j is the jth original constraint
			def feasiblePoint =
				DenseVector.tabulate[Double](dim)(j => if (j<n) x(j) else 1+cnts.constraints(j-n).valueAt(x))

			def pointWhereDefined = feasiblePoint
		}
	}



}



/** Result of phase_I feasibility analysis.*/
case class FeasibilityReport(

	/** Point which satisfies many constraints.*/
	x0:DenseVector[Double],
	/** Vector of the additional variables s_j (subject ot g_j(x)-s_j <= ub_j)
	  * at the optimum of the feasibility solver.
	  *
	  * In case of a SOI feasibility analysis the s_j are constrained to be non negative
	  * and we can classify the constraint g_j(x) <= ub_j to be feasible if s_j is sufficiently
	  * close to zero.
	  *
	  * In case of a simple feasibility analysis there is only one additional variable
	  * which is unconstrained and the vector s has dimension one.
	  * In this case we only get the following information: all the constraints
	  * are strictly feasible if and only if s(0)<0.
	  */
	s:Vector[Double],
	/** Flag if x0 satisfies all constraints strictly. */
	isStrictlyFeasible:Boolean,
	/** List of constraints which x0 does not satisfy strictly.*/
	constraintsViolated:Seq[Constraint]
)



