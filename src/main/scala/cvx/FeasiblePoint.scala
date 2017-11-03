package cvx

import breeze.linalg.DenseVector

/** A point strictly satisfying all constraints in a set of constraints.*/
trait FeasiblePoint {

  def feasiblePoint:DenseVector[Double]
}