package cvx

import breeze.linalg.DenseVector

/**
  * Created by vagrant on 02.11.17.
  */
object FeasibilityTests {


  /** Perform a feasibility analysis (first simple, then SOI) on the
    * constraints x_j>0, sum_jx_j=1, sgnA*P(A) <= sgnA*pA and
    * sgnB*P(B) <= sgnB*pB,
    * where the x_j are interpreted as probabilities on Omega={0,1,...,n-1}
    * determining the probabilities P(A), P(B) of the subsets A,B of Omega.
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
  def checkFeasibility(
    n:Int,
    I_A:DenseVector[Double],pA:Double,sgnA:Double,
    I_B:DenseVector[Double],pB:Double,sgnB:Double,
    pars:SolverParams, debugLevel:Int
  ):Unit = {

    val ct = ConstraintSets.probAB(n,I_A,pA,sgnA,I_B,pB,sgnB)
    val probCt = Constraints.sumToOne(n)

    print("\nDoing simple feasibility analysis")
    try {

      val fR:FeasibilityReport = ct.phase_I_Analysis(Some(probCt),pars,debugLevel)
      println(fR.toString(1e-9)); Console.flush()

    } catch {

      case e:InfeasibleException => print("\nInfeasible constraints:\n"+e.getMessage)
    }
    print("\n\nDoing SOI feasibility analysis")
    try {

      val fR:FeasibilityReport = ct.phase_I_Analysis_SOI(Some(probCt),pars,debugLevel)
      println(fR.toString(1e-9)); Console.flush()

    } catch {

      case e:InfeasibleException => print("\nInfeasible constraints:\n"+e.getMessage)
    }
  }


  /** Run all the tests with specific parameters.
    */
   def runAll(pars:SolverParams, debugLevel:Int):Unit = {

     print("\nChecking infeasible constraints P(A)>=0.51, P(B)>=0.51:\n")
     val n=20
     val I_A = DenseVector.tabulate[Double](n)(j => if(j<3) 1.0 else 0.0)
     val I_B = DenseVector.tabulate[Double](n)(j => if(j>=n/2) 1.0 else 0.0)
     val sgnA = -1.0; val sgnB = -1.0
     val pA=0.51; val pB=0.51
     checkFeasibility(n,I_A,pA,sgnA,I_B,pB,sgnB,pars,debugLevel)
   }
}
