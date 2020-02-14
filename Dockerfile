FROM tensorflow/tensorflow:devel-gpu
RUN apt-get update
Run apt-get install -y git
RUN pip3 install \
	tensorflow-gpu==2.0 \
	lxml \
	tqdm \
	opencv-python \
	tensorflow_hub

WORKDIR root/tensorflow
COPY Tensorflow2.0-Workbench root/tensorflow