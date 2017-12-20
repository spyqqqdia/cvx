package cvx

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * Created by oar on 11.12.17.
  */
class LinearObjectiveFunction (
                                override val dim:Int,
                                val r:Double,
                                val a:DenseVector[Double]
                              )
extends ObjectiveFunction(dim) {

  if(a.length!=dim){
    val msg = "Vector a must be of dimension "+dim+" but length(a) "+a.length
    throw new IllegalArgumentException(msg)
  }
  def valueAt(x:DenseVector[Double]) = { checkDim(x); r + (a dot x)  }
  def gradientAt(x:DenseVector[Double]) = a
  def hessianAt(x:DenseVector[Double]) = DenseMatrix.zeros[Double](dim,dim)
}

object LinearObjectiveFunction {

  /** f(x) = r+a'x */
  def apply(dim:Int, r:Double, a:DenseVector[Double]) = new LinearObjectiveFunction(dim,r,a)
  /** Inner product f(x) = a#x */
  def apply(a:DenseVector[Double]) = new LinearObjectiveFunction(a.length,0.0,a)
}