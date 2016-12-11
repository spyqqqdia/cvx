package cvx

import breeze.linalg.{DenseMatrix, DenseVector, _}



// WARNING: 
// in the barrier method handle the multiplications with the dimension reducing
// matrix x = x0+Fu _outside_ the sum in the barrier function (bilinear!) or else we will matrix
// multiply ourselves to death.
// This is the reason why we do not put this operation into the constraints themselves.

/** Solver for constrained convex optimization using the barrier method.
  *
  * @param C open convex set known to contain the minimizer
  * @param pars see [SolverParams]
  */
abstract class BarrierSolver(val objF:ObjectiveFunction, val C:ConvexSet with SamplePoint, val pars:SolverParams)
extends Solver {
 
    val dim=C.dim

	def startingPoint:DenseVector[Double] = C.samplePoint
	def barrierFunction(t:Double,x:DenseVector[Double]):Double
    def gradientBarrierFunction(t:Double,x:DenseVector[Double]):DenseVector[Double]
	def hessianBarrierFunction(t:Double,x:DenseVector[Double]):DenseMatrix[Double]
	/** Number m of inequality constraints. */
	def numConstraints:Int

	def checkDim(x:DenseVector[Double]):Unit =
		assert(x.length==dim,"Dimension mismatch x.length="+x.length+" unequal to dim="+dim)

	/** Find the location $x$ of the minimum of f=objF over C by the newton method
	  * starting from the starting point x0.
	  *
	  * @return Solution object: minimizer with additional info.
	  */
	def solve():Solution = {
	
	    val tol=pars.tol // tolerance for duality gap
		val mu = 10.0    // factor to increase parameter t in barrier method.
		var t = 1.0
		var x = startingPoint       // iterates x=x_k
		var sol:Solution = null   // solutions at parameter t
	    while(numConstraints/t>=tol){
		
		    // solver for barrier function at fixed parameter t
			val objF_t = new ObjectiveFunction(dim){

				def valueAt(x:DenseVector[Double]) = { checkDim(x); barrierFunction(t,x) }
				def gradientAt(x:DenseVector[Double]) = { checkDim(x); gradientBarrierFunction(t,x) }
				def hessianAt(x:DenseVector[Double]) = { checkDim(x); hessianBarrierFunction(t,x) }
			}
			val solver = new UnconstrainedSolver(objF_t,C,pars){
			
			    override def startingPoint = x
			}
		    sol = solver.solve()
			x = sol.x
			t = mu*t
		}
	    sol
	}
}


/** Some factory functions.*/
object BarrierSolver {

	/** BarrierSolver for minimization without equality constraints.
	  */
	def apply(objF:ObjectiveFunction, cnts:ConstraintSet with FeasiblePoint, pars:SolverParams): BarrierSolver = {

		val Feas = cnts.strictlyFeasibleSet
		val C = ConvexSet.addSamplePoint(Feas,cnts.feasiblePoint)
		new BarrierSolver(objF,C,pars){

			def numConstraints = cnts.constraints.length
			def barrierFunction(t:Double,x:DenseVector[Double]):Double =
				cnts.constraints.foldLeft(t*objF.valueAt(x))((sum:Double,cnt:Constraint) => {

					val d = cnt.ub - cnt.valueAt(x)
					if (d <= 0)
						throw new IllegalArgumentException("barrierFunction(x) undefined, not strictly feasible x = "+x)
					sum-Math.log(d)
				})
			def gradientBarrierFunction(t:Double,x:DenseVector[Double]):DenseVector[Double] =
				cnts.constraints.foldLeft(objF.gradientAt(x)*t)((sum:DenseVector[Double],cnt:Constraint) => {

					val d = cnt.ub - cnt.valueAt(x)
					val G = cnt.gradientAt(x)
					if (d <= 0)
						throw new IllegalArgumentException("gradientBarrierFunction(x) undefined, not strictly feasible x = "+x)
					sum-G/d
				})
			def hessianBarrierFunction(t:Double,x:DenseVector[Double]):DenseMatrix[Double] =
				cnts.constraints.foldLeft(objF.hessianAt(x)*t)((sum:DenseMatrix[Double],cnt:Constraint) => {

				val d = cnt.ub - cnt.valueAt(x)
				val G = cnt.gradientAt(x)
				val H = cnt.hessianAt(x)
				if (d <= 0)
					throw new IllegalArgumentException("gradientBarrierFunction(x) undefined, not strictly feasible x = "+x)

				sum+G*G.t/(d*d)-H/d
			})
		}
	}
	/** Version of solver bs which operates on the changed variable u related to the original variable
	  *  as x = z0+Fu.
	  *  This solves the minimization problem of bs under the additional constraint that
	  * x is of the form z0+Fu and operates on the variable u. Results are reported using the variable x.
	  *
	  * @param sol solution space of Ax=b (then z0=sol.z0, F=sol.F).
	  */
	private def variableChangedSolver(bs:BarrierSolver, sol:SolutionSpace): BarrierSolver = {

		// pull the domain bs.C back to the u variable
		val C = bs.C
		val z0 = sol.z0
		val F = sol.F
		val dim_u = F.cols
		val x0 = bs.startingPoint
		val u0 = sol.parameter(x0)      // u0 with x0 = z0+F*u0

		val D = new ConvexSet(dim_u) with SamplePoint {

			def isInSet(u:DenseVector[Double]) = C.isInSet(z0+F*u)
			def samplePoint = u0
		}

		new BarrierSolver(bs.objF,D,bs.pars){

			def numConstraints = bs.numConstraints
			def barrierFunction(t:Double,u:DenseVector[Double]) = bs.barrierFunction(t,z0+F*u)
			def gradientBarrierFunction(t:Double,u:DenseVector[Double]) = F.t*(bs.gradientBarrierFunction(t,z0+F*u))
			def hessianBarrierFunction(t:Double,u:DenseVector[Double]) = (F.t*(bs.hessianBarrierFunction(t,z0+F*u)))*F

			override def solve() = {

				// 'super': with new X { ... } we automatically extend X
				val sol = super.solve()
				new Solution(z0+F*sol.x,sol.gap,sol.normGrad,sol.iter,sol.maxedOut)
			}
		}
	}

    /** BarrierSolver for minimization with equality constraints Ax=b, A is mxn with m < n and full rank m,
      * when no inequality constraints are present.
      * The solver will solve the system as x = z0 + Fu, and perform the minimization after a change of variables
      * x --> u.
      * It is assumed that the strictly feasible point returned by [cnts.feasiblePoint] also satisfies
      * the equality constraints.
      */
    def apply(objF:ObjectiveFunction, eqs:EqualityConstraints, pars:SolverParams): UnconstrainedSolver = {

        val sol = eqs.solutionSpace
        val F:DenseMatrix[Double] = sol.F
        val z0:DenseVector[Double] = sol.z0
        val dim_u = F.cols    // dimension of u-variable
        val dim_x = F.rows    // dimension of original x-variable

        //check if z0 has appropriate dimension (F is then automatically correct)
        assert(z0.size == dim_x, "Dimension mismatch: z0.size="+z0.size+" not equal dim(x)="+dim_x)

        UnconstrainedSolver(objF,eqs,pars)
    }


	/** BarrierSolver for minimization with equality constraints Ax=b, A is mxn with m < n and full rank m,
	  * when a strictly feasible point for the inequalities, which also satisfies the equality constraints,
	  * is known.
	  * The solver will solve the system as x = z0 + Fu, and perform the minimization after a change of variables
	  * x --> u.
	  * It is assumed that the strictly feasible point returned by [cnts.feasiblePoint] also satisfies
	  * the equality constraints.
	  */
	def apply(
        objF:ObjectiveFunction, eqs:EqualityConstraints, cnts:ConstraintSet with FeasiblePoint, pars:SolverParams
    ): BarrierSolver = {

		val x0 = cnts.feasiblePoint
		assert(eqs.isSatisfiedBy(x0))

		val sol = eqs.solutionSpace
		val F:DenseMatrix[Double] = sol.F
		val z0:DenseVector[Double] = sol.z0
		val dim_u = F.cols    // dimension of u-variable
		val dim_x = F.rows    // dimension of original x-variable

		//check if z0 has appropriate dimension (F is then automatically correct)
		assert(z0.size == dim_x, "Dimension mismatch: z0.size="+z0.size+" not equal dim(x)="+dim_x)
		// check if all constraints have the same dimension as x
		assert(
			cnts.constraints.forall(cnt => cnt.dim==dim_x),
			"Dimension mismatch: not all constraints have dimension dim(x)"
		)
		// the solver without equality constraints
		val bsNoEqs = apply(objF,cnts,pars)
		// change variables x = z0+Fu
		variableChangedSolver(bsNoEqs,sol)
	}


	//--------- Feasibility Analysis: basic algorithm ----------------//


	/** Barrier solver for phase I feasibility analysis according to basic algorithm,
	  *  [boyd], 11.4.1, p579. No equality constraints.
	  */
	private def phase_I_Solver(cnts:ConstraintSet,pars:SolverParams):BarrierSolver = {

		val dim = cnts.dim
		val feasObjF = ConstraintSet.phase_I_ObjectiveFunction(dim,cnts)
		val feasCnts = ConstraintSet.phase_I_Constraints(dim,cnts)

		apply(feasObjF,feasCnts,pars)
	}



	/** Phase I feasibility analysis according to basic algorithm, [boyd], 11.4.1, p579
	  *  using a BarrierSolver on the feasibility problem. No equality constraints.
	  *
	  * @param cnts constraints to be anaylzed for feasibility
	  * @param pars solver parameters, see [SolverParams].
	  */
	private def phase_I_Analysis(
		cnts:ConstraintSet, pars:SolverParams
	): FeasibilityReport = {

		val feasBS = phase_I_Solver(cnts,pars)
		val sol = feasBS.solve

		val dim = cnts.dim
		val w_feas = sol.x                  // w = c(x,s)
		val x_feas = w_feas(0 until dim)    // unclear how many constraints that satisfies, so we check
		val s_feas = w_feas(dim)            // if s_feas < 0, all constraints are strictly satisfied.
		val isStrictlySatisfied = s_feas < 0
		val violatedCnts = cnts.constraints.filter(!_.isSatisfiedStrictly(x_feas))

		FeasibilityReport(x_feas,isStrictlySatisfied,violatedCnts)
	}


	/** Phase I feasibility analysis according to basic algorithm, [boyd], 11.4.1, p579
	  *  using a BarrierSolver on the feasibility problem.
	  *  With equality constraints parameterized as x=z0+Fu.
	  *  Reports feasible candidate as u0 not as x0 = z0+Fu0.
	  *
	  * @param cnts inequality constraints
	  * @param eqs equality constraints
	  * @param pars solver parameters, see [SolverParams].
	  */
	private def phase_I_Analysis(cnts:ConstraintSet, eqs:EqualityConstraints, pars:SolverParams): FeasibilityReport = {

		val n = cnts.dim
		val solEqs = eqs.solutionSpace
		val F:DenseMatrix[Double] = eqs.F
		val z0 = eqs.z0
		assert(n==F.rows,"Dimension mismatch F.rows="+F.rows+" not equal to n=dim(problem)="+n)

		val k=F.cols     // dimension of variable u in x=z0+Fu

		val solverNoEqs = phase_I_Solver(cnts,pars)
		// change variables x = z0+Fu
		val solver = variableChangedSolver(solverNoEqs,solEqs)
		val sol = solver.solve

		val w_feas = sol.x             // w = c(u,s), dim(u)=m, s scalar
		val u_feas = w_feas(0 until k)
		val x_feas = z0+F*u_feas       // unclear how many constraints that satisfies, so we check
		val s_feas = w_feas(k)         // if s_feas < 0, all constraints are strictly satisfied.
		val isStrictlySatisfied = s_feas < 0
		val violatedCnts = cnts.constraints.filter(!_.isSatisfiedStrictly(x_feas))

		FeasibilityReport(u_feas,isStrictlySatisfied,violatedCnts)
	}


	//--------- Feasibility Analysis via sum of infeasibilities ----------------//


	/** Barrier solver for phase I feasibility analysis according to more detailled sum of infeasibilities
	  *  algorithm, [boyd], 11.4.1, p579. No equality constraints.
	  */
	private def phase_I_Solver_SOI(cnts:ConstraintSet,pars:SolverParams):BarrierSolver = {

		val n = cnts.dim
		val feasObjF = ConstraintSet.phase_I_SOI_ObjectiveFunction(n,cnts)
		val feasCnts = ConstraintSet.phase_I_SOI_Constraints(n,cnts)
		apply(feasObjF,feasCnts,pars)
	}

	/** Phase I feasibility analysis according to more soi (sum of infeasibilities) algorithm,
	  *  [boyd], 11.4.1, p580 using a BarrierSolver on the feasibility problem. This approach
	  *  generates points satisfying more constraints than the basic method, if the problem is infeasible.
	  *  No equality constraints.
	  *
	  * @param cnts constraints to be anaylzed for feasibility
	  * @param pars solver parameters, see [SolverParams].
	  */
	private def phase_I_Analysis_SOI(cnts:ConstraintSet, pars:SolverParams): FeasibilityReport = {

		val p = cnts.numConstraints
		val n = cnts.dim
		val solver = phase_I_Solver_SOI(cnts,pars)
		val sol = solver.solve

		val w_feas = sol.x                     // w = c(x,s),  dim(x)=n, s_j=g_j(x), j<p
		val x_feas = w_feas(0 until n)         // unclear how many constraints that satisfies, so we check
		val s_feas = w_feas(n until n+p)       // if s_feas(j) < 0, for all j<p, then all constraints are strictly satisfied.
		val isStrictlySatisfied = (0 until n).forall(j => s_feas(j) < 0)
		val violatedCnts = cnts.constraints.filter(!_.isSatisfiedStrictly(x_feas))

		FeasibilityReport(x_feas,isStrictlySatisfied,violatedCnts)
	}


	/** Phase I feasibility analysis according to more soi (sum of infeasibilities) algorithm,
	  *  [boyd], 11.4.1, p580 using a BarrierSolver on the feasibility problem. This approach
	  *  generates points satisfying more constraints than the basic method, if the problem is infeasible.
	  *  With equality constraints.
	  *
	  * @param cnts inequality constraints to be anaylzed for feasibility
	  * @param eqs inequality constraints to be anaylzed for feasibility
	  * @param pars solver parameters, see [SolverParams].
	  */
	private def phase_I_Analysis_SOI(
		cnts:ConstraintSet, eqs:EqualityConstraints, pars:SolverParams
	): FeasibilityReport = {

		val p = cnts.numConstraints
		val n = cnts.dim
		val solEqs = eqs.solutionSpace
		val F:DenseMatrix[Double] = eqs.F
		val z0:DenseVector[Double] = eqs.z0
		assert(n==F.rows,"Dimension mismatch F.rows="+F.rows+" not equal to n=dim(problem)="+n)

		val k=F.cols     // dimension of variable u in change of variables x = z0+Fu

		val solverNoEqs = phase_I_Solver_SOI(cnts,pars)
		// change variables x = z0+Fu
		val solver = variableChangedSolver(solverNoEqs,solEqs)
		val sol = solver.solve

		val w_feas = sol.x                 // w = c(u,s),   dim(u)=m, s_j=g_j(x), j<p
		val u_feas = w_feas(0 until k)
		val x_feas = z0+F*u_feas           // unclear how many constraints that satisfies, so we check
		val s_feas = w_feas(k until k+p)   // if s_feas(j) < 0, for all j<p, then all constraints are strictly satisfied.
		val isStrictlySatisfied = (0 until p).forall(j => s_feas(j) < 0)
		val violatedCnts = cnts.constraints.filter(!_.isSatisfiedStrictly(x_feas))

		FeasibilityReport(x_feas,isStrictlySatisfied,violatedCnts)
	}


	//---------- Solvers, when no feasible points are known ----------------- //

	/** Barrier solver for minimization under the inequality constraints in cnts when no
	  * _strictly feasible point_ for the constraints cnts is known. Will then perform
	  * a feasibility analysis.
	  *
	  * @param doSOIAnalysis perform a sum of infeasibilities analysis ([boyd], section 11.4.1, p580).
	  * If set to false, the simple analysis will be carried out ([boyd], section 11.4.1, p579).
	  * @param pars solver parameters (see [SolverParams] applied to both the feasibility and
	  * minimization problem.
	  */
	def apply(objF:ObjectiveFunction, cnts:ConstraintSet, doSOIAnalysis:Boolean, pars:SolverParams):BarrierSolver = {

		val feasibilityReport: FeasibilityReport =
			if(!doSOIAnalysis)
				BarrierSolver.phase_I_Analysis(cnts,pars)
			else
				BarrierSolver.phase_I_Analysis_SOI(cnts,pars)

		if(!feasibilityReport.isStrictlyFeasible) throw new InfeasibleProblemException(feasibilityReport)

		val x0 = feasibilityReport.x0
        // check if this is really a feasible point
        assert(cnts.isSatisfiedStrictlyBy(x0),"Inequality constraints not strictly satisfied")

		val feasCnts = cnts.addFeasiblePoint(x0)
		apply(objF,feasCnts,pars)
	}


	/** Barrier solver for minimization under the inequality constraints in cnts and equality constraints in eqs
	  * when no _strictly feasible point_ for the constraints cnts is known. Will then perform a feasibility analysis.
	  *
	  * @param doSOIAnalysis perform a sum of infeasibilities analysis ([boyd], section 11.4.1, p580).
	  * If set to false, the simple analysis will be carried out ([boyd], section 11.4.1, p579).
	  * @param pars solver parameters (see [SolverParams] applied to both the feasibility and
	  * minimization problem.
	  */
	def apply(
        objF:ObjectiveFunction, cnts:ConstraintSet, eqs:EqualityConstraints, doSOIAnalysis:Boolean,
		pars:SolverParams
    ):BarrierSolver = {

		val feasibilityReport: FeasibilityReport =
			if(!doSOIAnalysis)
				BarrierSolver.phase_I_Analysis(cnts,eqs,pars)
			else
				BarrierSolver.phase_I_Analysis_SOI(cnts,eqs,pars)

		if(!feasibilityReport.isStrictlyFeasible) throw new InfeasibleProblemException(feasibilityReport)

		val x0 = feasibilityReport.x0

        // check if this is really a feasible point
        assert(cnts.isSatisfiedStrictlyBy(x0),"Inequality constraints not strictly satisfied")
        assert(eqs.isSatisfiedBy(x0),"Equality constraints not strictly satisfied")

		val feasCnts = cnts.addFeasiblePoint(x0)
		apply(objF,eqs,feasCnts,pars)
	}

}