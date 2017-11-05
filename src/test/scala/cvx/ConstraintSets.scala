package cvx

import breeze.linalg.DenseVector

/**
  * Created by vagrant on 02.11.17.
  *
  * Here we collect some feasible and infeasible constraint sets for
  * discrete probabilities (variable x satisfying x_j>0, sum_jx_j=1 is
  * interpreted as a probability distribution.
  *
  * The purpose is to have some examples to test feasibility analysis
  * and dist_KL minimization.
  */
object ConstraintSets {

  /** Constraint for probabilities x_0,...,x_{n-1}  on
    * Omega={0,1,2,...,n-1} as follows:
    * P(A)<=pA, if sgnA>0, and P(A)>=pA if sgnA<0. Same for the
    * inequality for P(B).
    *
    * This is feasible if and only if pA+pB<=1.
    * The constraints x_j>0 are included but the equality constraint
    * sum_jx_j=1 must be added separately.
    *
    * @param I_A indicator function of the subset A of Omega (as vector u,
    *            where u_j=0.0, if j is not in A, and u_j=1.0, if j is in A.
    * @param I_B indicator function of the subset B of Omega (similar to I_A).
    * @param sgnA: intended to be +-1, but anything nonzero works.
    *              If >0 the inequality is P(A)<=pA, if <0 it is P(A)>=pA.
    * @param sgnB: as above for the inequality for P(B).
    * @return the ConstraintSet specifying that x_j>0,
    *         as well as P(A)>=pA and P(B)>=pB.
    */
  def probAB(
    n:Int,
    I_A:DenseVector[Double],pA:Double, sgnA:Double,
    I_B:DenseVector[Double],pB:Double, sgnB:Double
  ):ConstraintSet = {

    assert(n>=12 && n%2==0,"\nn must be >= 12 and even, but n = "+n)
    assert(pA>0 && pB>0,"pA,pB must be positive but pA = "+pA+", pB = "+pB)

    val id1 = "P(A)"+(if(sgnA>0) " <= " else " >= ")+pA
    val ct1 = Constraints.expectation_lt_r(I_A*sgnA,pA*sgnA,id1)
    val id2 = "P(B)"+(if(sgnB>0) " <= " else " >= ")+pB
    val ct2 = Constraints.expectation_lt_r(I_B*sgnB,pB*sgnB,id2)

    // set up the constraints
    val cnts:List[Constraint] =
    ct2::ct1::Constraints.allCoordinatesPositive(n)

    // point where all constraints are defined
    val x = DenseVector.tabulate[Double](n)(j=>1.0/n)
    ConstraintSet(n,cnts,x)
  }

}
