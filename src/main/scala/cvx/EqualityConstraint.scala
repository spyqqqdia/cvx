package cvx

import breeze.linalg.{DenseMatrix, DenseVector, _}

import scala.collection.mutable.ListBuffer


/** Affine equality constraints of the form Ax=b, where A is mxn with m < n
  *  and full rank m. The condition rank(A)=m will not be checked.

  *  Parametrizes solutions as x = z0+Fu, where x=z0 is the minimum norm solution
  *  and $Fu\perp z0$, for all u in dimension n-m.
  *  Used for change of variables x --> u to reduce dimension and get rid of explicit
  *  equality constraints.
  */
class EqualityConstraint(val A:DenseMatrix[Double], val b:DenseVector[Double]){

  assert(A.rows < A.cols, "m=A.rows="+A.rows+" is not less than n=A.cols="+A.cols)
  assert(A.rows == b.length, "m=A.rows="+A.rows+" != b.length = "+b.length)
  val dim = A.cols
  val solutionSpace = SolutionSpace(A,b)
  val F = solutionSpace.F
  val z0 = solutionSpace.z0

  /** @return ||Ax-b||.*/
  def errorAt(x:DenseVector[Double]):Double = norm(A*x-b)
  def isSatisfiedBy(x:DenseVector[Double],tol:Double=1e-14):Boolean = errorAt(x) < tol*A.rows

  /** Add the constraints eqs to the existing ones.
    */
  def addEqualities(eqs:EqualityConstraint):EqualityConstraint = {

    // stack horizontally
    val new_A = DenseMatrix.vertcat[Double](A,eqs.A)
    val new_b = DenseVector.vertcat[Double](b,eqs.b)
    EqualityConstraint(new_A,new_b)
  }

  /** The equality constraints mapped to the dimension of phase I analysis
    */
  def phase_I_EqualityConstraint: EqualityConstraint = {

    val zeroCol = DenseMatrix.zeros[Double](A.rows,1)
    val B = DenseMatrix.horzcat(A,zeroCol)
    EqualityConstraint(B,b)
  }
  /** The equality constraints mapped to the dimension of phase I SOI analysis
    * @param p number of inequality constraints the pahse I SOI analysis is applied to.
    */
  def phase_I_SOI_EqualityConstraint(p:Int): EqualityConstraint = {

    val zeroCols = DenseMatrix.zeros[Double](A.rows,p)
    val B = DenseMatrix.horzcat(A,zeroCols)
    EqualityConstraint(B,b)
  }
  def printSelf = {

    val msg = "\nEquality constraints, matrix A:\n"+A+"\nvector b:\n"+b+"\n\n"
    print(msg)
  }
  def printSelf(logger:Logger,digits:Int) = {

    logger.print("\nEquality constraints, matrix A:\n")
    MatrixUtils.print(A,logger,digits)
    logger.print("\nEquality constraints, vector b:\n")
    MatrixUtils.print(b,logger,digits)
  }

  /** The equality constraint Ax=b under change of variables x=z0+Fu.
    * This is then rewritten as A(z0+Fu)=b, i.e. (AF)u=b-Az0.
    */
  def affineTransformed(z0:DenseVector[Double],F:DenseMatrix[Double]):EqualityConstraint =
    EqualityConstraint(A*F,b-A*z0)

  /** Turns this equality constraint Ax=b into a list of inequalities
    * row_i(A)'x<=b_i and row_i(A)'x>=b_i.
    */
  def asInequalities:List[LinearConstraint] = {

    val n = A.rows
    val dim = A.cols
    val buff = ListBuffer[LinearConstraint]()
    for(i <- 0 until n){

      val row_iA = A(i,::)
      val id1 = "row_"+i+"(A)'x <= "+b(i)
      val ineq1 = LinearConstraint(id1,dim,b(i),0.0,row_iA.t)
      val id2 = "row_"+i+"(A)'x >= "+b(i)
      val ineq2 = LinearConstraint(id2,dim,-b(i),0.0,-row_iA.t)
      buff+=ineq1
      buff+=ineq2
    }
    buff.toList
  }
}



object EqualityConstraint {

  def apply(A:DenseMatrix[Double], b:DenseVector[Double]) = new EqualityConstraint(A,b)
}