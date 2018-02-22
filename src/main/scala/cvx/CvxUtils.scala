package cvx

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.log





/**
  * Created by oar on 10.10.17.
  */
object CvxUtils {

  /** Assert that A and b are either both defined or both undefined.
    */
  def check(A:Option[DenseMatrix[Double]], b:Option[DenseVector[Double]]): Unit =
     assert( A.isDefined && b.isDefined || A.isEmpty && b.isEmpty,
       "\n\nA is " + (if(A.isDefined) "defined" else "undefined\n")+
       "b is " + (if(b.isDefined) "defined" else "undefined.\n")
     )

  /** Starting from u(s)=x+s*dx with s=s0 backtrack s=beta*s
    * until the termination criterion tc(u(s)) is satisfied or
    * the maximum number of steps has been reached.
    *
    * Throw LineSearchFailedException if the final point u(s) does
    * not satisfy the criterion tc(u(s)).
    *
    * @return u(s) at final value s.
    */
  def lineSearch(
    x:DenseVector[Double],dx:DenseVector[Double],
    tc:(Double)=>Boolean, beta:Double, s0:Double=1.0
  ): DenseVector[Double] = {

    assert(0<beta && beta <1,"\nbeta = "+beta+" not in (0,1)\n")
    val maxSteps = -30/log(beta)    // then beta^^maxSteps < 1e-13
    var step = 0
    var s=s0
    var u = x+dx*s
    while(!tc(s) && step<maxSteps){

      s = beta*s
      u = x+dx*s
      step += 1
    }
    if(!tc(s)){

      val msg = "\nLine search unsuccessful.\n"
      throw LineSearchFailedException(msg)
    }
    u
  }


}
