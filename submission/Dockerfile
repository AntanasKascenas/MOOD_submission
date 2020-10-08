FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

# copy files
ADD files /workspace/
RUN chmod +x /workspace/*.sh
RUN mkdir /mnt/data
RUN mkdir /mnt/pred
RUN pip install nibabel
RUN pip install scipy
