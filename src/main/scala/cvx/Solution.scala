package cvx

import breeze.linalg.{DenseVector, _}


/** Solution of minimization problem with additional information.
  * Created by oar on 12/6/16.
  *
  * Contains all quantities that may be relevant for the results of any one
  * of the solvers:
  * UnconstrainedSolver: x,newtonDecrement,normGradient
  * EqualityConstrainedSolver: x,newtonDecrement,normGradient,equalityGap
  * BarrierSolver: x,dualityGap,equalityGap (calls UnconstrainedSolver or
  * EqualityConstrainedSolver but the corresponding fields are not significant)
  * PrimalDualSolver: x,s,lambda,nu,dualityGap,normResidual.
  *
  * @param x minimizer
  * @param lambda dual variable attached to the inequality constraints
  * @param nu dual variable attached to the equality constraints
  * @param dualityGap duality gap at minimizer or Newton decrement (if unconstrained).
  * @param equalityGap ||Ax-b|| at minimizer x with equality constraints Ax=b or zero
  *   with no equality constraints.
  * @param normGrad norm of gradient at minimizer
  * @param normDualResidual: norm of the residual in PrimalDualSolver
  * @param iter iteration at termination (in outer loop for barrier method).
  * @param maxedOut  flag indicating that limit on number of iterations has been hit.
  *
  * Unconstrained minimization:
  * if ||gradF(x)|| < tol the algorithm terminates without computing the Newton increment at x,
  * in this case gap is the Newton increment at the last step and so does not have to be small.
  */
case class Solution(
                     x:DenseVector[Double],
                     lambda:Option[DenseVector[Double]],
                     nu:Option[DenseVector[Double]],
                     newtonDecrement:Option[Double],
                     dualityGap:Option[Double],
                     equalityGap:Option[Double],
                     normGrad:Option[Double],
                     normDualResidual:Option[Double],
                     iter:Int,
                     maxedOut:Boolean
){

  override def toString:String = {

    "\nSolution:"+
    "\nDecision variables x: "+x+
    "\nDual variable lambda: "+lambda.getOrElse("None")+
    "\nDual variable nu: "+nu.getOrElse("None")+
    "\nNewton decrement: "+newtonDecrement.getOrElse("None")+
    "\nDuality gap: "+dualityGap.getOrElse("None")+
    "\nEquality gap: "+equalityGap.getOrElse("None")+
    "\nNorm of gradient: "+normGrad.getOrElse("None")+
    "\nNorm of dual residual: "+normDualResidual.getOrElse("None")+
    "\nIterations: "+iter+
    "\nIteration limit reached: "+maxedOut+"\n"
  }
}

