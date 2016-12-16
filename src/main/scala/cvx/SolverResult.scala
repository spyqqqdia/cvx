package cvx

import breeze.linalg.{DenseVector, _}


/** Solution of minimization problem with additional information.
  * Created by oar on 12/6/16.
  *
  * @param x minimizer
  * @param gap duality gap at minimizer or Newton decrement (if unconstrained).
  * @param normGrad norm of gradient at minimizer
  * @param iter iteration at termination (in outer loop for barrier method).
  * @param maxedOut  flag indicating that limit on number of iterations has been hit.
  *
  * Unconstrained minimization:
  * if ||gradF(x)|| < tol the algorithm terminates without computing the Newton increment at x,
  * in this case gap is the Newton increment at the last step and so does not have to be small.
  */
case class Solution(x:DenseVector[Double], gap:Double, normGrad:Double, iter:Int, maxedOut:Boolean)

