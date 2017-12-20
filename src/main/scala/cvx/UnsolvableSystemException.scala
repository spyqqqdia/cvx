package cvx

/**
  * Created by oar on 15.11.17.
  */
case class UnsolvableSystemException(val msg:String) extends Exception(msg)