from .aws_util import S3Manager
from pyproj import Transformer
import csv
import math
import boto3
import io

class Convertor:
    """
    Interface for conversion and storage of results
    """
    def convert(self, result, worker_id=None):
        pass

    def set_top_left(self,top_left_lat,top_left_long):
        pass

    def save(self, filename, data):
        pass

def truncate(f, n):
    if f > 0:
        return math.floor(f * 10 ** n) / 10 ** n
    else:
        return math.ceil(f * 10 ** n) / 10 ** n

class LandCoverResultConvertor(Convertor):
    """
    Class for landstat result conversion
    """
    def __init__(self, pixels_per_file: int):
        self.pixels_per_file = pixels_per_file
        self.top_left_lat = None
        self.top_left_long = None
    
    def set_top_left(self,top_left_lat,top_left_long):
        self.top_left_lat = top_left_lat
        self.top_left_long = top_left_long

    def convert(self,result):
        transformer = Transformer.from_crs("+proj=utm +zone=19 +datum=WGS84 +units=m +no_defs", "+proj=longlat +datum=WGS84 +no_defs")
        returnData = {}
        print(result.shape)
        for i in range(result.shape[1]):
            for j in range(result.shape[2]):
                data = result[:,i,j]
                if data[0] == 0 or data[1] == 0:
                    continue
                init_x = self.top_left_lat+(30*i)
                init_y = self.top_left_long+(30*j)
                final_y,final_x = transformer.transform(init_x,init_y)
                csv_key = "converted_result_{}_{}.csv".format(str(truncate(final_x,2)),str(truncate(final_y,2)))
                if csv_key not in returnData:
                    returnData[csv_key] = []
                returnData[csv_key].append([final_x,final_y,data[0],data[1]])
        print("Conversion completed for the result. Keys: ",len(returnData.keys()))
        return returnData
    
    def save(self, filename, data):
        with open(filename, "w") as f:
            writer = csv.writer(f)
            writer.writerows(data)


class LandCoverResultS3Convertor(Convertor):
    """
    Class for landstat result conversion
    """

    def __init__(self, s3_client: S3Manager, bucket_name:str):
        self.top_left_lat = None
        self.top_left_long = None
        self.s3_client = s3_client
        self.bucket_name = bucket_name

    def set_top_left(self, top_left_lat, top_left_long):
        self.top_left_lat = top_left_lat
        self.top_left_long = top_left_long

    def convert(self, result,worker_id=None):
        transformer = Transformer.from_crs("+proj=utm +zone=19 +datum=WGS84 +units=m +no_defs",
                                           "+proj=longlat +datum=WGS84 +no_defs")
        returnData = {}
        print(result.shape)
        for i in range(result.shape[1]):
            for j in range(result.shape[2]):
                data = result[:, i, j]
                if data[0] == 0 or data[1] == 0:
                    continue
                init_x = self.top_left_lat + (30 * i)
                init_y = self.top_left_long + (30 * j)
                final_y, final_x = transformer.transform(init_x, init_y)
                csv_key = "converted_result_{}_{}.csv".format(str(truncate(final_x, 2)), str(truncate(final_y, 2)))
                if csv_key not in returnData:
                    returnData[csv_key] = []
                returnData[csv_key].append([final_x, final_y, data[0], data[1]])
        # if not worker_id is None:
        #     bucket_name = bucket_name + '-' + worker_id
        for csv_key in returnData:
            result = io.StringIO()
            writer = csv.writer(result)
            writer.writerows(returnData[csv_key])
            self.s3_client.save_data(self.bucket_name, csv_key, result.getvalue())
        print("Conversion completed for the result. Keys: ", len(returnData.keys()))
        return {}

    def save(self, filename, data):
        with open(filename, "w") as f:
            writer = csv.writer(f)
            writer.writerows(data)