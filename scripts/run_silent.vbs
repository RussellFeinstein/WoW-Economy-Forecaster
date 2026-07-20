' run_silent.vbs -- Launch a batch file with no visible window.
' Usage: wscript.exe run_silent.vbs "C:\path\to\script.bat"
'
' Used by setup_tasks.bat to register silent Task Scheduler jobs.
' The WScript.Shell.Run third parameter (True) makes this script
' wait for the batch to finish so Task Scheduler sees the real exit code.

If WScript.Arguments.Count = 0 Then
    WScript.Echo "Usage: wscript.exe run_silent.vbs <path-to-bat>"
    WScript.Quit 1
End If

Dim shell, batPath, exitCode
Set shell = CreateObject("WScript.Shell")
batPath = WScript.Arguments(0)

' Run(command, windowStyle, waitOnReturn)
'   windowStyle 0 = SW_HIDE (no window)
'   waitOnReturn True = block until .bat exits
exitCode = shell.Run("cmd /c """ & batPath & """", 0, True)
WScript.Quit exitCode
