package cvx

import breeze.linalg.DenseVector

/**
  * Created by oar on 02.11.17.
  */
object FeasibilityTests {


  /** Perform a feasibility analysis (first simple, then SOI) on the
    * constraints in cts subject additionally to the equality constraints
    * eqs.
    */
  def checkFeasibility(
    cts:ConstraintSet,eqs:Option[EqualityConstraint], pars:SolverParams, debugLevel:Int
  ):Unit = {

    print("\nDoing simple feasibility analysis with tolerance tol = "+pars.tolFeas+"\n")
    try {

      val fR:FeasibilityReport = cts.phase_I_Analysis(eqs,pars,debugLevel)
      if(debugLevel==0) println(fR.toString(1e-9)); Console.flush()

    } catch {

      case e:InfeasibleProblemException => print("\nInfeasible constraints:\n"+e.getMessage)
    }
    print("\n\nDoing SOI feasibility analysis")
    try {

      val fR:FeasibilityReport = cts.phase_I_Analysis_SOI(eqs,pars,debugLevel)
      if(debugLevel==0) println(fR.toString(1e-9)); Console.flush()

    } catch {

      case e:InfeasibleProblemException => print("\nInfeasible constraints:\n"+e.getMessage)
    }
  }

  /** Feasibility analysis of probability simplex x_j>=0, sum(x_j)=1.
    *
    * @param n
    * @param pars
    * @param debugLevel
    */
  def checkFeasibilityProbabilitySimplex(
    n:Int, pars:SolverParams, debugLevel:Int
  ):Unit = {

    // set up the constraints
    val cnts:List[Constraint] = Constraints.allCoordinatesPositive(n)

    // point where all constraints are defined.
    val x = DenseVector.tabulate[Double](n)(j=>1.0/n)
    val cts = ConstraintSet(n,cnts,x)
    val probCt = Constraints.sumToOne(n)

    checkFeasibility(cts,Some(probCt),pars,debugLevel)
  }

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
  def checkFeasibility_PAB(
    n:Int,
    I_A:DenseVector[Double],pA:Double,sgnA:Double,
    I_B:DenseVector[Double],pB:Double,sgnB:Double,
    pars:SolverParams, debugLevel:Int
  ):Unit = {

    val cts = ConstraintSets.probAB(n,I_A,pA,sgnA,I_B,pB,sgnB)
    val probCt = Constraints.sumToOne(n)

    checkFeasibility(cts,Some(probCt),pars,debugLevel)
  }

  /** Check feasibility via simple and SOI analysis of a random ConstraintSet
    * consisting of p linear inequalities and q quadratic inequalities
    * satisfied by the point x0.
    *
    * First we check these without equality constraints, then with an
    * added random equality constraint consisting of 3 equalities satisfied by x0.
    * Here x0 is always a feasible point and so the constraints are feasible.
    */
  def checkRandomFeasibleConstraints(
    x0:DenseVector[Double],p:Int,q:Int,pars:SolverParams,debugLevel:Int
  ):Unit = {

    print("\n\n\nChecking a random feasible ConstraintSet with "+p+" linear and "+q+
          " quadratic inequalities:\n\n")
    val cts = ConstraintSets.randomConstraintSet(x0,p,q)
    print("First without additional equality constraints:\n")
    checkFeasibility(cts,None,pars,debugLevel)
    print("\n\nNext with additional equality constraints:\n")
    val eqs = Constraints.randomEqualityConstraint("Random equality constraint",x0,3)
    checkFeasibility(cts,Some(eqs),pars,debugLevel)
  }


  /** Run all the tests with specific parameters.
    */
   def runAll(pars:SolverParams, debugLevel:Int) = {

     print("\n\nChecking infeasible probability constraints P(A)>=0.51, P(B)>=0.51:\n")
     val n=20
     val I_A = DenseVector.tabulate[Double](n)(j => if(j<3) 1.0 else 0.0)
     val I_B = DenseVector.tabulate[Double](n)(j => if(j>=n/2) 1.0 else 0.0)
     val sgnA = -1.0; val sgnB = -1.0
     val pA=0.51; val pB=0.51
     checkFeasibility_PAB(n,I_A,pA,sgnA,I_B,pB,sgnB,pars,debugLevel)

     val x0 = DenseVector.tabulate[Double](20)(j => 1.0)     // feasible point
     val p=10; val q=5;                                       // number of linear and quadratic inequalities

     for(tests <- 1 to 5)
       checkRandomFeasibleConstraints(x0,p,q,pars,debugLevel)
   }
}
