package cvx

import breeze.linalg.{DenseMatrix, DenseVector}



/**
  * Created by vagrant on 10.10.17.
  */
object CvxUtils {

  /** Assert that A and b are either both defined or both undefined.
    */
  def check(A:Option[DenseMatrix[Double]], b:Option[DenseVector[Double]]): Unit =
     assert( A.isDefined && b.isDefined || A.isEmpty && b.isEmpty,
       "\n\nA is " + (if(A.isDefined) "defined" else "undefined\n")+
       "b is " + (if(b.isDefined) "defined" else "undefined.\n")
     )




}
