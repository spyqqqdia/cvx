name := "cvx"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies  ++= Seq(
    "org.scalanlp" %% "breeze" % "0.10",
    // native libraries are not included by default. add this if you want them (as of 0.7)
    // native libraries greatly improve performance, but increase jar sizes.
    "org.scalanlp" %% "breeze-natives" % "0.10"
)
resolvers ++= Seq(
    // other resolvers here
    "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)
