# General Imports
from queue import Queue

# Two queues: 
#   (1) that holds data that can be pulled for real-time classification
#   (2) that holds data that is slowly pulled and placed into designated file
# One boolean dictating whether we are still collecting data
collecting_data = None
queue_classification = None
queue_file_data_transfer = None

"""
Initializes all global variables

params: None
output: None
"""
def init():
    global collecting_data
    global queue_classification
    global queue_file_data_transfer

    collecting_data = False
    queue_classification = Queue()
    queue_file_data_transfer = Queue()


"""
Sets the boolean collecting_data to True

params: None
output: None
"""
def startCollectingData():
    global collecting_data
    collecting_data = True


"""
Sets the boolean collecting_data to False

params: None
output: None
"""
def stopCollectingData():
    global collecting_data
    collecting_data = False


# TESTING
def printClassifyQueue():
    print("PRINTING CLASSIFYING QUEUE")
    while not queue_classification.empty():
        print(queue_classification.get())
