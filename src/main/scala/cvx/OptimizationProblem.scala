package cvx

import breeze.linalg.{DenseVector, _}


/** Solution to an optimization problem, both location and value of the minimum.
  * Since the location of the minimum may not be unique it is formulated as a boolean check
  * (indicator function of the solution set).
  */
trait KnownMinimizer {

    /** Minimum value of the objective function: sum_j\phi_j(0).*/
    def isMinimizer(x:DenseVector[Double],tol:Double):Boolean
    /** Minimum value of the objective function: sum_j\phi_j(0).*/
    def minimumValue:Double
}
object KnownMinimizer {

    /** Uniquely determined solution at x=x0 with value objF(x0)=y0.*/
    def apply(x0:DenseVector[Double],y0:Double) = new KnownMinimizer {

        def isMinimizer(x:DenseVector[Double],tol:Double) = norm(x-x0) < tol
        def minimumValue = y0
    }

}

/**
  * Created by oar on 12/2/16.
  *
  * Constrained or unconstrained optimization problem. Contains objective function and abstract constraint
  * (convex set C).
  *
  * @param dim number of independent variables x_j.
  * @param solver solver used to find the minimizer.
  */
class OptimizationProblem(val id:String, val dim:Int, val solver:Solver) {

    def objF:ObjectiveFunction = solver.objF
    def solve:Solution = solver.solve

}

/** Factory functions to allocate problems and select the solver to use.
  *
  */
object OptimizationProblem {


    /** Allocates an optimization problem constrained only by $x\in C$, where C is an open convex set
      * known to contain a minimizer (typically the full Euclidean space).
      *
      * @param id ID for problem
      * @param dim dimension of independent variable
      * @param objF objective function
      * @param pars solver parameters, see [SolverParams].
      * @return problem minimizing objective function under the constraint $x\in C$ applying the parameters in pars
      * and starting the iteration at C.samplePoint.
      */
    def unconstrained(
        id:String, dim:Int, objF:ObjectiveFunction, C: ConvexSet with SamplePoint, pars:SolverParams
    ): OptimizationProblem = {

        assert(dim==objF.dim && dim==C.dim)
        val solver = UnconstrainedSolver(objF,C,pars)
        new OptimizationProblem(id,dim,solver)
    }


    /** Allocates an optimization problem with inequality constraints (no equality constraints) using
      * the barrier method if a strictly feasible point for the constraints is known.
      * Then phase I analysis in setting up the solver is not needed.
      *
      * @param id ID for problem
      * @param dim dimension of independent variable
      * @param objF objective function
      * @param ineqs inequality constraints
      * @param pars solver parameters, see [SolverParams].
      * @return problem minimizing objective function under constraints applying the parameters in pars
      * and starting the iteration at ineqs.feasiblePoint.
      */
    def withBarrierMethod(
        id:String, dim:Int, objF:ObjectiveFunction, ineqs: ConstraintSet with FeasiblePoint, pars:SolverParams
    ): OptimizationProblem = {

        assert(dim==objF.dim && dim==ineqs.dim)
        val solver = BarrierSolver(objF,ineqs,pars)
        new OptimizationProblem(id,dim,solver)
    }

    /** Allocates an optimization problem with equality and inequality constraints using
      * the barrier method if a strictly feasible point for the constraints is known.
      * Then phase I analysis in setting up the solver is not needed.
      *
      * @param id ID for problem
      * @param dim dimension of independent variable
      * @param objF objective function
      * @param ineqs inequality constraints
      * @param pars solver parameters, see [SolverParams].
      * @return problem minimizing objective function under constraints applying the parameters in pars
      * and starting the iteration at ineqs.feasiblePoint.
      */
    def withBarrierMethod(
        id:String, dim:Int, objF:ObjectiveFunction, ineqs: ConstraintSet with FeasiblePoint, eqs:EqualityConstraints,
        pars:SolverParams
    ): OptimizationProblem = {

        assert(dim==objF.dim && dim==ineqs.dim)
        val solver = BarrierSolver(objF,ineqs,eqs,pars)
        new OptimizationProblem(id,dim,solver)
    }


    /** Allocates an optimization problem with inequality constraints (no equality constraints) using
      * the barrier method if no strictly feasible point for the constraints is known.
      * Then phase I analysis to find a starting point for the iterations will be performed.
      *
      * @param id ID for problem
      * @param dim dimension of independent variable
      * @param objF objective function
      * @param ineqs inequality constraints
      * @param pars solver parameters, see [SolverParams].
      * If set to false, the simple analysis will be carried out ([boyd], section 11.4.1, p579).
      * @param printFeas print the value of the variable s in the simple feasibility analysis
      *                  if no feasible point is found.
      * @return problem minimizing objective function under constraints applying the parameters in pars
      * and starting the iteration at ineqs.feasiblePoint.
      */
    def withBarrierMethod(
        id:String, dim:Int, objF:ObjectiveFunction, ineqs: ConstraintSet, pars:SolverParams, printFeas:Boolean
    ): OptimizationProblem = {

        assert(dim==objF.dim && dim==ineqs.dim)
        val solver = BarrierSolver(objF,ineqs,pars,printFeas)
        new OptimizationProblem(id,dim,solver)
    }


    /** Allocates an optimization problem with inequality and equality constraints using
      * the barrier method if no strictly feasible point for the constraints is known.
      * Then phase I analysis to find a starting point for the iterations will be performed.
      *
      * @param id ID for problem
      * @param dim dimension of independent variable
      * @param objF objective function
      * @param ineqs inequality constraints
      * @param pars solver parameters, see [SolverParams].
      * @param printFeas print the value of the variable s in the simple feasibility analysis
      *                  if no feasible point is found.
      * If set to false, the simple analysis will be carried out ([boyd], section 11.4.1, p579).
      * @return problem minimizing objective function under constraints applying the parameters in pars
      * and starting the iteration at ineqs.feasiblePoint.
      */
    def withBarrierMethod(
        id:String, dim:Int, objF:ObjectiveFunction, ineqs: ConstraintSet, eqs:EqualityConstraints, pars:SolverParams,
        printFeas:Boolean
    ): OptimizationProblem = {

        assert(dim==objF.dim && dim==ineqs.dim)
        val solver = BarrierSolver(objF,ineqs,eqs,pars,printFeas)
        new OptimizationProblem(id,dim,solver)
    }


    /** Add the known solution to the minimization problem.
      * For testing purposes.
      */
    def addSolution(problem:OptimizationProblem,optSol:KnownMinimizer):
    OptimizationProblem with KnownMinimizer  =
        new OptimizationProblem(problem.id,problem.dim,problem.solver) with KnownMinimizer {

            def isMinimizer(x:DenseVector[Double],tol:Double) = optSol.isMinimizer(x,tol)
            def minimumValue = optSol.minimumValue

    }
}



