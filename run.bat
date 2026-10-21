@echo off
echo Compiling VectorDB Java engine...
javac *.java
if %ERRORLEVEL% NEQ 0 (
    echo Compilation failed!
    exit /b %ERRORLEVEL%
)
echo Starting VectorDB server on http://localhost:8080...
java Main
