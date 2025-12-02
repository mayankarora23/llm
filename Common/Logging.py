from enum import Enum
import inspect
import os
import sys
import traceback
import threading
from datetime import datetime

class LogLevel(Enum):
    TRACE = 0
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4
    FATAL = 5

class Logging:
    _instance = None
    __logLevel : LogLevel = LogLevel.INFO
    __isDetailedLogNeeded : bool = True
    __lock = threading.Lock()

    @staticmethod
    def GetInstance():
        if Logging._instance is None:
            Logging._instance = Logging()
        return Logging._instance

    def SetLogLevel(self, logLevel):
        self.__logLevel = logLevel

    def SetDetailedLogNeeded(self, detailedLogNeeded):
        self.__isDetailedLogNeeded = detailedLogNeeded

    def Log(self, logLevel, message, logFileName="", printToConsole=True):
        if logLevel.value >= self.__logLevel.value:
            if self.__isDetailedLogNeeded:
                frame = inspect.currentframe().f_back.f_back
                lineNumber = frame.f_lineno
                functionName = frame.f_code.co_name
                fileNameWithPath = frame.f_code.co_filename
                fileName = os.path.basename(fileNameWithPath)
                currentTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                with self.__lock:
                    if logFileName:
                        with open(logFileName, 'a') as logFile:
                            logFile.write(f"{currentTime} [{fileName}:{lineNumber} ({functionName})] <{logLevel.name}>: {message}\n")
                    if printToConsole:
                        print(f"{currentTime} [{fileName}:{lineNumber} ({functionName})] <{logLevel.name}>: {message}", flush=True)
            else:
                with self.__lock:
                    if logFileName:
                        with open(logFileName, 'a') as logFile:
                            logFile.write(f"{message}\n")
                    if printToConsole:
                        print(f"{message}", flush=True)

    def Trace(self, message, logFileName="", printToConsole=True):
        self.Log(LogLevel.TRACE, message, logFileName=logFileName, printToConsole=printToConsole)

    def Debug(self, message, logFileName="", printToConsole=True):
        self.Log(LogLevel.DEBUG, message, logFileName=logFileName, printToConsole=printToConsole)

    def Info(self, message, logFileName="", printToConsole=True):
        self.Log(LogLevel.INFO, message, logFileName=logFileName, printToConsole=printToConsole)

    def Warning(self, message, logFileName="", printToConsole=True):
        self.Log(LogLevel.WARNING, message, logFileName=logFileName, printToConsole=printToConsole)

    def Error(self, message, logFileName="", printToConsole=True):
        self.Log(LogLevel.ERROR, message, logFileName=logFileName, printToConsole=printToConsole)

    def Fatal(self, message, logFileName="", printToConsole=True):
        self.Log(LogLevel.FATAL, message, logFileName=logFileName, printToConsole=printToConsole)
        traceback.print_stack()
        raise SystemExit("Fatal error occurred!")