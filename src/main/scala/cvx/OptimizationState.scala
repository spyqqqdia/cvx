package cvx

import breeze.linalg.DenseVector

/**
  * Created by or on 21.12.17.
  *
  * Current state of the optimization computation.
  * Used as input to various termination criteria.
  *
  * This will be expanded as needed (for example for primal dual infeasible
  * start solvers).
  *
  * @param equalityGap: ||Ax-b||
  * @param dualityGap: upper bound for the duality gap (e.g. numInequalities/t
  *                  at parameter t along the central path in the barrier solver)
  *
  */
case class OptimizationState(
              normGradient:Double,
              newtonDecrement:Double,
              dualityGap:Double,
              equalityGap:Double,
              objectiveFunctionValue:Double
)

