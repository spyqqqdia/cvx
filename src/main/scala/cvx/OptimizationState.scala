package cvx

import breeze.linalg.DenseVector

/**
  * Created by oar on 21.12.17.
  *
  * Current state of the optimization computation.
  * Used as input to various termination criteria.
  *
  * Contains an option for every field that is relevant in any of the
  * optimization routines. But typically in each routine some of these
  * fields will remain empty.
  *
  * @param equalityGap: ||Ax-b||
  * @param dualityGap: upper bound for the duality gap (e.g. numInequalities/t
  *                  at parameter t along the central path in the barrier solver)
  * @param normDualResidual: norm of the dual residual r_dual in PrimalDualSolver.
  *                See boyd-vandenverghe, section 11.7.1, p610.
  *
  */
case class OptimizationState(
                              normGradient:Option[Double],
                              newtonDecrement:Option[Double],
                              dualityGap:Option[Double],
                              equalityGap:Option[Double],
                              objectiveFunctionValue:Double,
                              normDualResidual:Option[Double] = None
                            ) {

  override def toString:String =
      "\nnormGradient: "+normGradient.getOrElse("None") +
      "\nnewtonDecrement: "+newtonDecrement.getOrElse("None") +
      "\ndualityGap: "+dualityGap.getOrElse("None") +
      "\nequalityGap: "+equalityGap.getOrElse("None") +
      "\nnorm of dual residual: "+normDualResidual.getOrElse("None") +
      "\nobjectiveFunctionValue: "+objectiveFunctionValue + "\n"

}
