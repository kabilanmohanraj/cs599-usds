FROM rayproject/ray:latest-py39-cpu
#Install RSR dependencies \
RUN sudo apt-get --assume-yes update
RUN sudo apt-get --assume-yes install libspatialindex-dev libgdal-dev libgl1-mesa-glx
RUN conda install -y cython scipy numpy libspatialindex zarr gdal wget conda-forge::python-wget
RUN conda install -y rasterio
RUN /home/ray/anaconda3/bin/pip install git+https://github.com/jgrss/geowombat
RUN /home/ray/anaconda3/bin/pip install opencv-python gsutil h5netcdf xarray rioxarray pyproj scipy boto3
#For telemetry
RUN /home/ray/anaconda3/bin/pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-jaeger
RUN mkdir RSR
ADD ./* RSR/
ENTRYPOINT ["bash", "RSR/start-ray-node.sh"]
CMD ["head"]

