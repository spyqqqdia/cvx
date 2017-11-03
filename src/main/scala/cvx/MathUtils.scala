package cvx

/**
  * Created by oar on 12/2/16.
  */
object MathUtils {

  /** u raised to power n, for n>=0.*/
  def pow(u:Double,n:Int):Double = {

    assert(n>=0, "pow(u,n) for n<0 not implmented, n="+n)
    if(n==0) 1.0 else u*pow(u,n-1)
  }
  def round(u:Double,d:Int):Double = {

    val f = pow(10,d)
    Math.round(u*f)/f
  }
}