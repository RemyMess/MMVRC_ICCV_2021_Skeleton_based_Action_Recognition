# NOTE: store raw data in MMVRC_ICCV_2021_Skeleton_based_Action_Recognition/data_grabbing/raw_data

class DataGrabber:
    def __init__(self):
        raise NotImplementedError

    def getDataBatch(self):
        raise NotImplementedError

    def getAllData(self):
        raise NotImplementedError
