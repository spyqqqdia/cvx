package cvx

import java.io.IOException
import java.io.PrintWriter



/** Created by vagrant on 09.02.17.
  */
class Logger(val logFilePath: String) {

  private val writer: PrintWriter = try
    new PrintWriter(logFilePath, "UTF-8")
  catch {
    case e: Exception =>
      val msg: String = "\nUnable to allocate print writer: " + e.getMessage
      System.out.println(msg)
      throw e
  }
  def println(msg: String):Unit = { writer.println(msg); writer.flush() }
  def print(msg: String):Unit = { writer.print(msg); writer.flush() }

  def flush():Unit = writer.flush()
  def close():Unit = {
    writer.flush()
    writer.close()
  }
}
object Logger {

  def apply(logFilePath:String) = new Logger(logFilePath)

}
