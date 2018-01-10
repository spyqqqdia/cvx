package cvx

import breeze.linalg.{DenseMatrix, DenseVector, NotConvergedException, _}

/**
  * Created by oar on 12/4/16.
  *
  * Class to minimize a convex function subject to equality and inequality constraints
  * using gradient and Hessian information
  *
  * In addition there might be an additional abstract constraint $x\in C$, where $C$ is an
  * open convex set. The intention is that $C$ is the full space (thus the constraint $x\in C$
  * vacuous) or it is known that the objective function approaches +oo as x approaches the boundary
  * of $C$.
  *
  * This is the case in the barrier method and can be treated similarly to the case where C
  * is the full space.
  *
  */
trait Solver {

  def startingPoint: DenseVector[Double]
  def dim = startingPoint.length

  def pars:SolverParams

  /** @return Solution object (minimizer with additional info based on standard termination criterion.
    */
  def solve(debugLevel: Int = 0): Solution

  /** @return Solution object (minimizer with additional info.
    */
  def solveSpecial(terminationCriterion: (OptimizationState) => Boolean, debugLevel: Int = 0): Solution

  /** The solver operating on the variable u related to the original variable x via the
    * affine transform x = z0+Fu. This means that the underlying problem has been similarly
    * transformed (objective function and constraints).
    *
    * This can be viewed as introducing and additional constraint that the solution x0 must
    * be of the form x0 = z0+F*u0.
    * Solution will be reported in the variable u not in x.
    *
    * @param u0 a vector satisfying this.startingPoint = z0 + F*u0.
    */
  def affineTransformed(z0:DenseVector[Double],F:DenseMatrix[Double],u0:DenseVector[Double]):Solver
}