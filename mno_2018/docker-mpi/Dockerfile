FROM ubuntu:xenial
RUN apt-get update -y && apt-get install -y build-essential \
	nano \
	man \
	sudo \
	openssh-server
RUN groupadd mpi_user
RUN useradd mpi_user -g mpi_user -m -s /bin/bash
ARG file_ompi=$file_ompi
ARG dir_ompi=$dir_ompi
RUN mkdir -p /home/mpi_user/openmpi/installation
RUN if [ ! -d "/home/mpi_user/openmpi/installation/$dir_ompi" ]; then mkdir -p /home/mpi_user/openmpi/installation/; wget https://www.open-mpi.org/software/ompi/v3.0/downloads/$file_ompi -P /home/mpi_user/openmpi/installation; cd /home/mpi_user/openmpi/installation && tar -xzvf $file_ompi;fi
ARG inst_ompi=$inst_ompi
RUN cd /home/mpi_user/openmpi/installation && chown -hR mpi_user:mpi_user $dir_ompi
RUN mkdir -p /var/run/sshd
RUN echo "mpi_user ALL=(ALL:ALL) NOPASSWD:ALL" | (EDITOR="tee -a" visudo)
RUN echo "mpi_user:mpi" | chpasswd
USER mpi_user
RUN cd $inst_ompi/ && ./configure --prefix=$inst_ompi -with-sge && make all install
ENV PATH=$inst_ompi/bin:$PATH
ENV LD_LIBRARY_PATH=$inst_ompi/lib:$LD_LIBRARY_PATH
