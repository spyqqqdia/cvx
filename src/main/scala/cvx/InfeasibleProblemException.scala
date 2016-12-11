package cvx

/**
  * Created by oar on 12/10/16.
  */
class InfeasibleProblemException(val report:FeasibilityReport)
    extends Exception("Problem could not be established.")
