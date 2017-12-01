package cvx

/**
  * Created by oar on 12/10/16.
  */
class InfeasibleProblemException(val report:FeasibilityReport,tol:Double)
  extends Exception("\nNo feasible point found:\n"+report.toString(tol)+"\n\n")