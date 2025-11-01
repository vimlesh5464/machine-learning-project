# Importing the sys module for system-specific parameters and functions
import sys

# Importing custom logger from the networksecurity package
from networksecurity.logging import logger

# Defining a custom exception class
class NetworkSecurityException(Exception):
    # Constructor that takes error message and system details
    def __init__(self,error_message,error_details:sys):
        self.error_message = error_message
        
        # Extract exception type, value, and traceback using sys.exc_info()
        _,_,exc_tb = error_details.exc_info()
        
        # Get the line number where the exception occurred
        self.lineno=exc_tb.tb_lineno
        
        # Get the filename where the exception occurred
        self.file_name=exc_tb.tb_frame.f_code.co_filename 
    
    # Overriding the string representation of the exception
    def __str__(self):
        # Returning a formatted error message with file name, line number, and error message
        return "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        self.file_name, self.lineno, str(self.error_message))
        

# Main block for testing the custom exception
if __name__=='__main__':
    try:
        # Logging an info message before performing an operation
        logger.logging.info("Enter the try block")
        
        # Intentional error (division by zero) to test exception handling
        a=1/0
        print("This will not be printed",a)
    
    except Exception as e:
        # Raising the custom NetworkSecurityException with error details
        raise NetworkSecurityException(e,sys)
