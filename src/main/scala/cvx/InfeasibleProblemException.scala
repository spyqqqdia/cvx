package cvx

/**
  * Created by oar on 12/10/16.
  */
class InfeasibleProblemException(val report:FeasibilityReport)
    extends Exception("Feasibility of problem could not be established.")
