package cvx

import breeze.linalg.DenseVector

/**
  * Created by or on 21.12.17.
  *
  * Current state of the optimization computation.
  * Used as input to various termination criteria.
  */
case class OptimizationState(
              gradient:DenseVector[Double],
              newtonDecrement:Double,
              dualityGap:Double,
              objectiveFunctionValue:Double
)
