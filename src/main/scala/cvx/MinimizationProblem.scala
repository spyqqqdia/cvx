package cvx


/** Convex minimization problem with affine equality and convex inequality
 *  constraints. Allocates its own solver.
 */
class MinimizationProblem(val id:String, val dim:Int, val constraints:List[Constraint], val solver:Solver){

	/** Call the solver and report on the constraints at the solution (active or not).
	 *  @return value of Solver.solve
	 */
	def solve(maxIter: Int, alpha: Double, beta: Double, tol: Double, delta: Double, verbose:Boolean): Solution = {
	
	    if(verbose) print("\n\nSolving problem "+id)
		val sol = solver.solve(maxIter, alpha, beta, tol, delta)
		val x = sol.x
		if(verbose){
		
		    var msg:String = ""
			for(cnt <- constraints){
		
				val mrg = MathUtils.round(cnt.margin(x),8)
				msg += "Constraint "+cnt.id+", margin="+mrg+", active: "+cnt.isActive(x,tol)+"\n"
			}
			print(msg+"\n")
		}
		sol
	}
}
object MinimizationProblem {

    def apply(id:String, dim:Int, constraints:List[Constraint], solver:Solver) = {

        // check dimension of constraints
        assert(constraints.forall(cnt => cnt.dim==dim))
        new MinimizationProblem(id,dim,constraints,solver)
    }


}