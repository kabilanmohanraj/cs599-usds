from bz2 import compress
from lib.convertor import Convertor
import ray
import uuid
import socket
import os
import glob
from pathlib import Path
import wget
import time as clock
import logging
import re
import rioxarray
import xarray as xr
import numpy as np
from utils import get_parent_context
from opentelemetry import trace
from .aws_util import S3Manager
from .models import Model
import zlib
import io
import shutil
import itertools
from collections import defaultdict

# TODO: Trace this class
@ray.remote(num_cpus=1)
class LandCoverWorker(object):
    """A landcover worker to execute tasks.
    Attributes:
        id (UUID): The unique id of the worker.
        name (str): The worker's name
    """

    def __init__(self, id=uuid.uuid1(), name=""):
        # print("Constructor tryna construct")
        # otlpmodule = importlib.import_module("otlp")
        # print(otlpmodule)
        # print("Worker sys",sys.path)
        # sys.path.append("/home/ray/NASA-JPL-ec528")
        # The unique id of the actor
        self.id = id
        # Actor's name
        self.name = name
        # Host info
        self.hostname = socket.gethostname()
        self.ip = socket.gethostbyname(self.hostname)
        # The eosvault download command with most arguments hardcoded
        self.download_command = "cd /home/ray/eosvault/files; eosvault download --satellite landsat --asset-id LANDSAT/LC08/C01/T1_SR --max-workers {} --config-file /home/ray/eosvault/files/config.yaml --out-dir /home/ray/data/ --grid-file /home/ray/data/CONUS_grids_15.gpkg --grid {} --start-date 1999-03-01 --end-date 2020-11-01 --grid-crs  '+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs' --out-res 30"
        # Grid download times
        self.download_times = []
        # The COG handles
        self.COG_handles = None
        # Partition information
        self.partition_idx = None
        # The mapping from hostname to workers
        self.assignments = None
        # xarray partitions
        self.data = None
        # store times of observations
        self.times = None
        # id of partition received
        self.partition_id = 0
        # Adding global timer
        # self.global_timer=timer
        self.timestamps = []
        #S3 Bucket to store intermediate data to
        self.bucket = None
        self.aws_client: S3Manager = None
        #Set total number of nodes
        self.total_nodes = None
        self.node_id = None
        self.top_left = None
        self.top_left_lat = None
        self.top_left_long = None
        self.path = None
        self.row = None
        self.global_timers = defaultdict(list)
        self.downloaded_files = []
        self.total_uncompressed_size = 0

    # Store the mapping from hostname to workers
    def set_assignments(self, assignments):
        self.assignments = assignments

    # Returns the unique id of the worker
    def id(self):
        return self.id

    def set_id(self, id):
        self.id = id

    # Set total nodes
    def set_total_nodes(self, total_nodes):
        self.total_nodes = total_nodes

    def get_total_nodes(self):
        return self.total_nodes

    # Set bucket Name
    def set_bucket_name(self, bucket_name):
        self.bucket = bucket_name

    def get_bucket_name(self):
        return self.bucket

    # Set AWS Client
    def set_aws_client(self, aws_client: S3Manager):
        self.aws_client = aws_client

    def get_aws_client(self):
        return self.aws_client

    #Set Path row
    def set_path_row(self, path, row):
        self.path = path
        self.row = row

    def get_path_row(self):
        return self.path, self.row

    # Set Node Id
    def set_node_id(self, node_id: int):
        self.node_id = node_id

    def get_node_id(self):
        return self.node_id

    # Returns the name of the worker's host
    def hostname(self):
        return self.hostname

    # Returns the ip of the worker's host
    def ip(self):
        return self.ip

    #Return the timers from the worker
    def get_global_timers(self):
        return self.global_timers

    #Get Total uncompressed size
    def get_total_uncompressed_size(self):
        return self.total_uncompressed_size

    # TODO: Trace this function
    # Pulls data from a list of URLs (using wget)
    # Download to /home/ray/raw
    def pull_data(self, urls):
        """
        Todo: Move this method to Archive class to get data and return handle to data
        """
        # Make sure folder exists
        os.makedirs('/home/ray/raw', exist_ok=True)
        os.chdir('/home/ray/raw')
        # Scan for existing tifs
        all_tifs = list(glob.glob('/home/ray/raw/*.TIF'))

        def extract_filename(path: str):
            path = Path(path)
            return path.name

        filenames = set(map(extract_filename, all_tifs))
        print(f'Downloading {len(urls)} TIFs')
        for url in urls:
            for attempt in range(10): #10 attempts of retry 
                try:
                    url_file_name = wget.filename_from_url(url)
                    if url_file_name in filenames:
                        # Don't download if it already exists
                        continue
                    else:
                        wget.download(url)
                    self.downloaded_files.append(url_file_name)
                    break
                except Exception as e:
                    print("Download {} failed: Exception {}".format(url,e))
                    continue
        print("Final downloaded files for the worker {}-{}".format(self.node_id,self.id),self.downloaded_files)


    # Deletes raw data downloaded with pull_data()
    def delete_raw_data(self):
        # os.system("rm -f /home/ray/raw/*.TIF")
        # os.system("rm -f /home/ray/*.TIF")
        # os.system("rm -f /home/ray/NASA-JPL/*.TIF")
        return True

    # Pulls data using eosvault
    def eosvault_pull(self, grid_ids, max_workers=1):
        for grid_id in grid_ids:
            cmd = "cd /home/ray/eosvault/files; eosvault download --satellite landsat --asset-id LANDSAT/LC08/C01/T1_SR --max-workers {} --config-file /home/ray/eosvault/files/config.yaml --out-dir /home/ray/data/ --grid-file /home/ray/data/CONUS_grids_15.gpkg --grid {} --start-date 1999-03-01 --end-date 2020-11-01 --grid-crs  '+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs' --out-res 30".format(
                max_workers, grid_id)
            logging.info("  Downloading grid {} using {} workers...".format(grid_id, max_workers))

            start = clock.time()
            res = os.system(cmd)
            self.download_times.append((grid_id, clock.time() - start))
        return True

    # Returns eosvault download times
    def get_download_times(self):
        return self.download_times

    # Returns timeline
    def get_timers(self):
        return self.timestamps

    # Clears eosvault download times
    def clear_download_times(self):
        self.download_times.clear()

    # Deletes raw data downloaded with pull_data()
    def delete_eosvault_data(self):
        os.system("rm -r /home/ray/data/0*")
        return True

    # Stores data into the given destination
    def store_data(self, data, destination):
        pass

    # Creates time series from the raw data partition
    # NOTE (john): This method is called after calling 'create_time_series()' on
    # the dataset object
    def create_time_series(self):
        pass

    def store_cog_handle(self, cog_handle):
        self.COG_handles = cog_handle
        # Partition information
        self.partitions = None

    # TODO: Trace this function
    def redistribute(self, compression=False):
        # Set directory of local TIFF files
        self.global_timers["{}-{}-getFileNames".format(self.node_id,self.id)].append(clock.time())
        os.chdir('/home/ray/raw/')
        filenames_disk = sorted(glob.glob('/home/ray/raw/*.TIF'))
        # print("{} Length of files: {} Start: {} end: {} --> ".format(self.hostname,len(filenames),startidx,startidx + urls_per_worker),filenames)
        # filenames = filenames[startidx:startidx + urls_per_worker]
        # print("Worker's share of URL ", self.hostname, "-->", filenames)
        filenames = []
        for file in self.downloaded_files:
            filenames.append("/home/ray/raw/{}".format(file))
        result = all(file in filenames_disk for file in filenames)
        print("All files are avaiable in disk: {}".format(result))
        self.global_timers["{}-{}-getFileNames".format(self.node_id,self.id)].append(clock.time())

        # Import them lazily as a dask array
        def index_from_filenames(filenames):
            return [re.split("[_.]", f)[-2] for f in filenames]

        # TODO: Trace this function
        def index_time_from_filenames(filenames):
            return [re.split("[_.]", f)[3] for f in filenames]

        # send data to specified worker
        # TODO: Trace this function
        def send_data_to_worker(data, times, top_left, worker,top_left_lat,top_left_long):
            # print("Size of data before compression: ",data.nbytes)
            if compression:
                print("Size of data before compression: ",data.nbytes)
                print("Dtype:                           ",data.dtype)
                f = io.BytesIO()
                np.save(f, data)
                compressed_data = zlib.compress(f.getvalue(),level=-1)
                print("Size of data after compression:  ",len(compressed_data))
                return worker.store_partition.remote(compressed_data, times, top_left, top_left_lat, top_left_long ,True)
            else:
                return worker.store_partition.remote(data, times, top_left,top_left_lat, top_left_long)

        # TODO: Trace this function
        def chunk(chunks={'x': 5000, 'y': 5000}):
            # TODO: Move this somewhere better
            xmin = 247530
            xmax = 445980
            ymin = 4539000
            ymax = 4718970

            # Create time dimension from file name
            times = index_time_from_filenames(filenames)  # This will hold the times to make the time dim
            data_list = []
            if len(filenames) == 0:
                print('filenames in chunk() is empty')
            self.global_timers["{}-{}-rioxarrayFile".format(self.node_id,self.id)].append(clock.time())
            for file in filenames:
                tif = rioxarray.open_rasterio(file,
                                              chunks=chunks)  # Create Pointer to data on disk, and established the chunks
                # Adjust the size of the data on disk to be over the tile in question
                tif = tif.rio.pad_box(minx=xmin, maxx=xmax, miny=ymin, maxy=ymax,
                                      constant_values=0)
                tif = tif.rio.slice_xy(minx=xmin, maxx=xmax, miny=ymin, maxy=ymax)

                data_list.append(tif)  # append Pointer to data list
            self.global_timers["{}-{}-rioxarrayFile".format(self.node_id,self.id)].append(clock.time())
            # Create one big pointer to data on disk by adding the time axis, which indicates when the picture was taken
            self.global_timers["{}-{}-finalData".format(self.node_id,self.id)].append(clock.time())
            time_dim = xr.Variable('time', times)
            all_data = xr.concat(data_list, dim=time_dim).astype(
                'uint16')  # make sure the max size is 16 bits to save space
            print("{}-{}: Data size uncompressed: ".format(self.node_id,self.id),all_data.nbytes)
            # get x and y from da
            # then do node and worker partition
            (y, x) = all_data.shape[2:4]
            self.global_timers["{}-{}-finalData".format(self.node_id,self.id)].append(clock.time())
            self.global_timers["{}-{}-partitionData".format(self.node_id,self.id)].append(clock.time())
            # Partition nodes and worker
            node_partition = self.partition_nodes(x)
            worker_partition = self.partition_workers(y)
            self.global_timers["{}-{}-partitionData".format(self.node_id,self.id)].append(clock.time())
            return node_partition, worker_partition, all_data

        # TODO: Trace this function
        def reshuffle(node_partition, worker_partition, da):
            res = []
            times = da.time.values
            # print("Total Data size uncompressed: ",da.nbytes)
            # Get all the other host assigned  to this worker
            self.global_timers["{}-{}-reshuffle".format(self.node_id,self.id)].append(clock.time())
            for hostname in self.assignments:
                # Get the node grid assigned for every host
                node_grid = node_partition[hostname]
                # print("Hostname: {} node grid: {}".format(hostname, node_grid))
                for i in range(len(self.assignments[hostname])):
                    # Loop through all the recieving worker and send them their respective partitions
                    worker_grid = worker_partition[i]
                    # print("Hostname: {} Worker: {} worker grid: {}".format(hostname, i, worker_grid))
                    # Get the portion of data from the actual tiff file . Remember lazy loading until .values is called
                    self.global_timers["{}-{}-unpackData".format(self.node_id,self.id)].append(clock.time())
                    lat_coord = da[:, 0, worker_grid[0]:worker_grid[1], node_grid[0]:node_grid[1]].x[0].values
                    long_coord = da[:, 0, worker_grid[0]:worker_grid[1], node_grid[0]:node_grid[1]].y[0].values
                    data = da[:, 0, worker_grid[0]:worker_grid[1], node_grid[0]:node_grid[1]].values
                    self.total_uncompressed_size = self.total_uncompressed_size + data.nbytes
                    self.global_timers["{}-{}-unpackData".format(self.node_id,self.id)].append(clock.time())
                    # print("Hostname: {} Worker: {} Data Shape: {}".format(hostname, i, data.shape))
                    top_left = [worker_grid[0], node_grid[0]]
                    # Send the portion of data retrieved from file
                    self.global_timers["{}-{}-sendDataToWorkers".format(self.node_id,self.id)].append(clock.time())
                    res.append(send_data_to_worker(data, times, top_left, self.assignments[hostname][i], lat_coord, long_coord))
                    self.global_timers["{}-{}-sendDataToWorkers".format(self.node_id,self.id)].append(clock.time())
            self.global_timers["{}-{}-reshuffle".format(self.node_id,self.id)].append(clock.time())
            return res
        try:
            if len(filenames) > 0:
                node_partition, worker_partition, da = chunk()
                rslt = reshuffle(node_partition, worker_partition, da)
                return rslt
            else:
                return []
        except Exception as e:
            logging.error("Error while redistributing for workerID: {}".format(self.id), exc_info=True)
            return None

    # TODO: Trace this function
    def apply_model(self, model: Model,ctx_dic=None) -> bool:
        ctx = get_parent_context(ctx_dic['traceId'],ctx_dic['spanId'])
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("apply_model_worker", context=ctx):
            print("Hostname: {} top left: {}".format(self.hostname, self.top_left))

            return model.apply_model(self.data, self.times, self.top_left)

    # # Store xarray partition on this worker
    # # Will be called by other workers when passing partitions to this worker
    # # Concatenate new partition to stored data
    # TODO: Trace this function
    def store_partition(self, partition, times, top_left, top_left_lat, top_left_long, compression=False):
        if compression:
            partition = io.BytesIO(zlib.decompress(partition))
            partition = np.load(partition)
            print(type(partition))
        if self.data is None:
            self.data = partition
            self.times = times
            self.top_left = top_left
            self.top_left_lat = top_left_lat
            self.top_left_long = top_left_long
        else:
            self.data = np.concatenate([self.data, partition])
            self.times = np.concatenate([self.times, times])
        self.partition_id += 1
        # store data to file after receiving all partitions
        if self.partition_id == self.total_nodes:  # Partition id 10 ->1
            # span.set_tag("host",self.hostname)
            if self.bucket is None:
                filename = str(self.id) + '_data.npy'
                os.chdir('/home/ray/raw/')
                np.save(filename, self.data)
                # print(self.hostname, self.id, "saved data")
            else:
                filename = '{}-{}-{}-{}-{}-{}-{}-{}-data.npy'.format(self.path,self.row,self.node_id, self.id,
                                                               self.top_left[0], self.top_left[1],
                                                               self.top_left_lat, self.top_left_long)
                bytes_ = io.BytesIO()
                np.save(bytes_, self.data, allow_pickle=True)
                bytes_.seek(0)
                self.aws_client.save_data(self.bucket, filename, bytes_)
                # print(self.hostname, self.id, "saved data")
                time_filename = '{}-{}-{}-{}-{}-{}-{}-{}-times.npy'.format(self.path,self.row, self.node_id, self.id,
                                                                     self.top_left[0], self.top_left[1],
                                                                     self.top_left_lat, self.top_left_long)
                timebytes_ = io.BytesIO()
                np.save(timebytes_, self.times, allow_pickle=True)
                timebytes_.seek(0)
                self.aws_client.save_data(self.bucket+"-times", time_filename, timebytes_)
                # print(self.hostname, self.id, "saved time data")
        return

    # TODO: Trace this function
    def store_partition_idx(self, partition_idx):
        # This only needs to be called once for each of the 80 workers
        # But this was a quick fix based on the current design of the code
        # Under current design of the code, this will be called 8*80 times.
        if self.partition_idx is None:
            self.partition_idx = partition_idx
        return

    # Partition the data to stripes for each node
    # TODO: Trace this function
    def partition_nodes(self, x):
        nodes = list(self.assignments.keys())
        root = x // len(nodes)
        decimal = float(x) / len(nodes) - root
        remainder = int(decimal * len(nodes))
        partition = {}
        last = 0

        for i in range(len(nodes)):
            if i < len(nodes) - remainder:
                partition[nodes[i]] = [last, last + root]
                last += root
            else:
                partition[nodes[i]] = [last, last + (root + 1)]
                last += (root + 1)
        return partition

    # Partition the stripes to grids for each worker
    # TODO: Trace this function
    def partition_workers(self, y):
        workers_per_node = len(list(self.assignments.values())[0])
        root = y // workers_per_node
        decimal = float(y) / workers_per_node - root
        remainder = int(decimal * workers_per_node)
        partition = []
        last = 0

        for i in range(workers_per_node):
            if i < workers_per_node - remainder:
                partition.append([last, last + root])
                last += root
            else:
                partition.append([last, last + root + 1])
                last += (root + 1)

        return partition

    # TODO: Trace this function
    def read_stored_data(self):
        os.chdir('/home/ray/raw/')
        self.data = np.load(str(self.id) + '_data.npy')

    # TODO: Trace this function
    def get_output_files(self,convertor:Convertor = None):
        files_names = list(glob.glob('/home/ray/raw/results_*.npy'))
        output_map = {}
        if convertor is not None:
            convertor.set_top_left(self.top_left_lat, self.top_left_long)
        for file in files_names:
            if convertor is None:
                output_map[os.path.basename(file)] = np.load(file)
            else:
                converted_data = convertor.convert(np.load(file), self.id)
                for key in converted_data:
                    if key in output_map:
                        output_map[key].extend(converted_data[key])
                    else:
                        output_map[key] = converted_data[key]
        print("All files have been obtained... sending the results... Keys: ", len(output_map.keys()))
        return output_map

    # TODO: Trace this function
    def populate_data(self, file_name, time_file_name):
        if self.aws_client is not None:
            file_name_data = file_name.split("-")
            data_byte = io.BytesIO(self.aws_client.get_data(self.bucket, file_name))
            data_byte.seek(0)
            self.data = np.load(data_byte)
            time_data_byte = io.BytesIO(self.aws_client.get_data(self.bucket+"-times", time_file_name))
            time_data_byte.seek(0)
            self.times = np.load(time_data_byte)
            self.top_left = [file_name_data[2], file_name_data[3]]
            self.top_left_lat = float(file_name_data[4])
            self.top_left_long = float(file_name_data[5])
        return True

    # TODO: Trace this function
    def delete_raw_folder(self):
        try:
            shutil.rmtree("/home/ray/raw")
        except:
            print("Directory doesnt exists....")
        os.makedirs('/home/ray/raw', exist_ok=True)
        print("Raw folder recreated: {}".format(os.path.isdir('/home/ray/raw')))
        return True

    # TODO: Trace this function
    def reset_worker(self):
        self.data = None
        # store times of observations
        self.times = None
        self.downloaded_files = []
        self.global_timers = defaultdict(list)
        # id of partition received
        self.partition_id = 0
        self.top_left = None
        self.top_left_lat = None
        self.top_left_long = None
        # print("Worker variables reset")
        return True



