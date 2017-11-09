package cvx

import breeze.linalg.{DenseVector, sum}
import breeze.numerics.{abs, log}
import breeze.stats.{median,_}

/**
  * Created by oar on 12/2/16.
  */
object MathUtils {

  /** u raised to power n, for n>=0.*/
  def pow(u:Double,n:Int):Double = {

    assert(n>=0, "pow(u,n) for n<0 not implemented, n="+n)
    if(n==0) 1.0 else u*pow(u,n-1)
  }
  def round(u:Double,d:Int):Double = {

    val f = pow(10,d)
    Math.round(u*f)/f
  }

  ////------------------- One dimensional minimization  ------------------------////

  /** Computes the location x of a local minimum of a function f of one
    * variable in the interval [a,b] by interval bisection.
    * This needs the derivative f'(x) and will return a or b if f is monotone
    * on [a,b] (except in rare cases where f has a saddle point).
    *
    * @param tol x is computed to precision tol (meaning bracketed in an interval
    *            of length tol, then estimated to be the midpoint.
    */
  def minBisect(df:(Double)=>Double,a:Double,b:Double,tol:Double):Double = {

    var l=a; var u=b
    while(u-l>tol){

      val m = (u+l)/2
      if(df(m)<0) u=m else l=m
    }
    (l+u)/2
  }

  /** Solves the equation g(t)=0 using Newton's method. This can fail.
    * User is responsible for reasonable application. Throws Exception
    * if tolerance is not hit within 100 iterations.
    *
    * @param tol iterations terminate as soon as |g(t)|<tol.
    * @param G function G(t)=(g(t),g'(t))
    * @param t0 starting point.
    */
  def solveNewton1D(G:(Double)=>(Double,Double), t0:Double, tol:Double):Double = {

    var iter=0; var t=t0; var continue=true
    while(iter<100 && continue){

      val Gt = G(t); val gt=Gt._1
      if(abs(gt)>tol){

        val dgt = Gt._2
        t = t - gt/dgt
        iter+=1

      } else continue=false
    }
    if(iter==100) throw new Exception("\nNewton1D failed to converge.\n")
    t
  }

  /** Computes the location of a local minimum of g(t) by solving the
    * equation g'(t)=0 using Newton's method. In general this can fail
    * in all sorts of was including computing a local maximum instead.
    * User is responsible for reasonable application.
    *
    * This function is redundant and is included here only to the correct
    * use of solveNewton1D in this context is documented.
    *
    * @param tol iterations terminate as soon as |g(t)|<tol.
    * @param G function G(t)=(g'(t),g"(t))
    * @param t0 starting point.
    */
  def minNewton1D(G:(Double)=>(Double,Double),t0:Double,tol:Double):Double = solveNewton1D(G,t0,tol)



  ////------------------- SVD regularization ------------------------////

  /** Auxilliary function computes the tuple
    * (Q(t),Q'(t),Q"(t),R(t),R'(t),R"(t))
    * from docs/convex_notes.pdf, p??, eq(??)-(??).
    *
    * Note that we are doing this for \mu_j=1 and \lambda_j=pow(s(j),2p+2),
    * where the s(j) are the singular values \sigma_j.
    * We will only sum over singular values > 1e-10.
    */
  def glmHelper(t:Double,s:DenseVector[Double],c:DenseVector[Double],p:Int):
  (Double,Double,Double,Double,Double,Double) = {

    assert(t>0,"\nNegative t = "+t+" not allowed.\n")
    val n=s.length; val k=c.length
    assert(n==k,"\nlength(s) = "+n+" not equal to length(c) = "+k+"\n")

    // number of singular values > 1e-10
    val r = sum(s.map(sj => if(sj>1e-10) 1 else 0))
    var Q=0.0; var dQ=0.0; var d2Q=0.0
    var R=0.0; var dR=0.0; var d2R=0.0
    var j=0
    while(j<n){

      if(s(j)>1e-10){

        val wj = t+pow(s(j),2*(p+1))    // \l_j+t\mu_j, recall \mu_j=1
        val wj2 = wj*wj
        val wj3 = wj2*wj
        val cj2 = c(j)*c(j)
        Q += cj2/wj
        R += log(wj)                    // factor 1/r added later
        dQ -= cj2/wj2
        dR += 1.0/wj                    // factor 1/r added later
        d2Q += cj2/wj3                  // factor 2 added later
        d2R -= 1.0/wj2                  // factor 1/r added later
      }
      j+=1
    }
    // (Q(t),Q'(t),Q"(t),R(t),R'(t),R"(t)), missing factors added
    (Q,dQ,2*d2Q,R/r,dR/r,d2R/r)
  }

  /** The GLM score function $g(t)$ for SVD regularization, see
    * docs/convex_notes, section ***.
    *
    * @param p smoothness parameter.
    * @param s vector of singular values (may or may not contain the zero
    *          singular values).
    */
  def svdGLM(t:Double,s:DenseVector[Double],c:DenseVector[Double],p:Int):Double = {

    assert(t>0,"\nNegative t = "+t+" not allowed.\n")
    val n=s.length; val k=c.length
    assert(n==k,"\nlength(s) = "+n+" not equal to length(c) = "+k+"\n")

    var sum_1=0.0; var sum_2=0.0; var j=0; var r=0
    // sum only over singular values above threshold 1e-10
    while(j<n){
      if(abs(s(j))>1e-12) {

        val w = t+pow(s(j),2*(p+1))    // \l_j+t\mu_j
        sum_1 += c(j)*c(j)/w
        sum_2 += log(w)
        r += 1
      }
      j+=1
    }
    log(sum_1)+sum_2/r
  }
  /** Returns the tuple (g(t),g'(t),g"(t)), where g(t) is the GLM
    * score function for SVD regularization.
    * See docs/convex_notes.pdf, section??, p??, eq(??)-(??).
    */
  def svdGLM3(t:Double,s:DenseVector[Double],c:DenseVector[Double],p:Int):
  (Double,Double,Double) = {

    val all_QR = glmHelper(t,s,c,p)  // (Q(t),Q'(t),Q"(t),R(t),R'(t),R"(t))
    val Q = all_QR._1; val dQ = all_QR._2; val d2Q = all_QR._2
    val R = all_QR._4; val dR = all_QR._5; val d2R = all_QR._6

    val g = log(Q)+R;  val dg = -dQ/Q+dR;  val d2g = d2R + (d2Q*Q-dQ*dQ)/(Q*Q)

    (g,dg,d2g)
  }

  /** Finds the location t of a local minimum of the GLM score function
    * g(t) for SVD regularization by solving the equation g'(t)=0 using
    * Newton's method. See docs/convex_notes, section ***.
    *
    * @param p smoothness parameter.
    * @param s vector of singular values (may or may not contain the zero
    *          singular values).
    */
  def min_svdGLM(t:Double,s:DenseVector[Double],c:DenseVector[Double],p:Int):Double = {

    val G = (t:Double) => {
      val Gt = svdGLM3(t,s,c,p);  // (g(t),g'(t),g"(t))
      (Gt._2,Gt._3)
    }
    val tol = 1e-6
    // median(s(j))^{2(p+1)}
    val t0 = pow(median(s),2*(p+1))
    solveNewton1D(G,t0,tol)
  }



}