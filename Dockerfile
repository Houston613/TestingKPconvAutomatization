FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN pip install numpy scikit-learn PyYAML matplotlib mayavi PyQt5
COPY . .
WORKDIR /workspace/cpp_wrappers
RUN sh compile_wrappers.sh 

WORKDIR /workspace

CMD ["python3", "test_models.py"]