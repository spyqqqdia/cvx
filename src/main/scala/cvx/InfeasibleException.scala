/**
  * Created by vagrant on 10.10.17.
  */
package cvx

import breeze.linalg.{DenseVector}

/**
  * @param x0 point found by feasibility analysis
  * @param violatedConstraints list of constraints not satisfied
  * @param eqResidual
  */
case class InfeasibleException(
  x0:DenseVector[Double],
  violatedConstraints:Seq[Constraint],
  eqResidual:Option[Double]) extends Exception({

     val  msg ="\n\nPoint found by feasibility analysis:\n"+x0+"\n\nConstraints violated:\n"
     violatedConstraints.foldLeft(msg)((acc,cnt:Constraint) => acc+cnt.id+"\n")+"\n"+
     eqResidual.map(_ => "\nEqualities, residual: "+_)+"\n"
})