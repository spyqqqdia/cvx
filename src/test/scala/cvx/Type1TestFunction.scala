package cvx

import breeze.linalg.{DenseMatrix, DenseVector, _}
/**
  * Created by oar on 12/2/16.
  *
  * Test function g(x) for unconstrained optimization as in example 1,
  * docs/cvx_notes, section Hessian:
  *
  * $g(x)=\sum_j\alpha_j\phi_j(a_j\cdot x)$, where $a_j=row_j(A)$.
  *
  * The dimension is n=A.cols.
  * These functions all assume global minima at all points x satisfying
  * Ax=0, where the mxn matrix A satisfies m<=n and is a parameter of the test
  * function.
  *
  * With the matrix A we control scaling of the independent variables and with
  * the coefficients alpha_j we scale the dependent variables.
  * With this we can examine the need for and effectiveness of preconditioners.
  *
  * @param A mxn matrix A satisfying m<=n.
  * @param alpha vector with positive entries
  */
abstract class Type1TestFunction(val A:DenseMatrix[Double],val alpha:DenseVector[Double])
extends TestFunction(A.cols) {

    val m = A.rows
    assert(m<=A.cols,"A.rows<=A.cols required, but A.rows="+A.rows+", A.cols="+A.cols)
    assert(alpha.length==m,"q=alpha.length must equal A.rows but q="+alpha.length+", A.rows="+A.rows)
    // coefficients must be nonnegative

    def id:String
    /** Function phi_j as in docs/cvx_notes, section Hessian, example 1.*/
    def phi(j:Int,u:Double):Double
    /** Derivative of function phi_j.*/
    def dphi(j:Int,u:Double):Double
    /** Second derivative of function phi_j.*/
    def d2phi(j:Int,u:Double):Double

    /** Global minimizers are the solutions of Ax=0. */
    def isMinimizer(x:DenseVector[Double],tol:Double) = norm(A*x)<tol*Math.sqrt(sum(A:*A))
    /** Minimum value of the objective function: sum_j\phi_j(0).*/
    def globalMin:Double = (0 until m).map(j => alpha(j)*phi(j,0)).sum

    def valueAt(x:DenseVector[Double]):Double = {

        var sum = 0.0
        var j = 0
        while(j<m){ sum += alpha(j)*phi(j,A(j,::).t dot x); j+=1 }
        sum
    }

    def gradientAt(x:DenseVector[Double]):DenseVector[Double] = {

        var sum = DenseVector.zeros[Double](x.length)
        var i = 0
        while(i<m){

            val a_i = A(i,::).t  // row_i(A) as col vec
            sum += a_i*alpha(i)*dphi(i,a_i dot x)
            i+=1
        }
        sum
    }

    def hessianAt(x:DenseVector[Double]):DenseMatrix[Double] = {

        var sum = DenseMatrix.zeros[Double](x.length,x.length)
        var i = 0
        while(i<m){

            val a_i = A(i,::).t     // row_i(A) as col vec
            sum += (a_i*a_i.t)*alpha(i)*d2phi(i,a_i dot x)
            i+=1
        }
        sum
    }
}


object Type1TestFunction {


    /** $f(x)=\sum_j\alpha_j[(a_j\cdot x)^2]^q$, where $a_j=row_j(A)$.
      * Here we shoud have q>=1 for this to be twice continuously differentiable.
      *
      */
    def powerTestFunction(A:DenseMatrix[Double], alpha:DenseVector[Double], q:Double):Type1TestFunction = {

        assert(q>=1,"q="+q+" is < 1.")
        new Type1TestFunction(A: DenseMatrix[Double], alpha: DenseVector[Double]) {

            def id = "Type 1 power test function with q=" + MathUtils.round(q,3)
            def phi(j: Int, u: Double) = Math.pow(u * u, q)
            def dphi(j: Int, u: Double) = {

                val y = (2*q)*Math.pow(u * u, q - 0.5)
                if (u>0) y else -y
            }
            def d2phi(j: Int, u: Double) = (2*q)*(2*q-1)*Math.pow(u * u, q - 1)
        }
    }

    /** Type 1 test function of power type with matrix A with random entries in
      * (0,1), coefficient vector alpha = 1:dim and power exponent 2*q.
      *
      * @param dim dimension will be dim, A will be dim x dim
      */
    def randomPowerTestFunction(dim:Integer,q:Double):Type1TestFunction = {

        assert(q>=1,"q="+q+" is < 1.")
        val B = DenseMatrix.rand[Double](dim,dim)
        // improve the condition number
        val A = B + DenseMatrix.eye[Double](dim)*2.0

        val alpha = DenseVector.tabulate(dim){i => i*1.0}
        powerTestFunction(A,alpha,q)
    }
}
