# docker build -t whisper-persian --build-arg WHISPER_IMPLEMENTATION=transformers .
# docker run -d --gpus all -p 7860:7860 -v whisper-cache:/root/.cache whisper-persian

FROM huggingface/transformers-pytorch-gpu
EXPOSE 7860

ARG WHISPER_IMPLEMENTATION=transformers
ENV WHISPER_IMPLEMENTATION=${WHISPER_IMPLEMENTATION}

ADD . /opt/whisper-persian/

# Latest version of transformers-pytorch-gpu seems to lack tk.
# Further, pip install fails, so we must upgrade pip first.
RUN apt-get -y install python3-tk ffmpeg
RUN python3 -m pip install --upgrade pip

RUN if [ "${WHISPER_IMPLEMENTATION}" = "whisper" ]; then \
    python3 -m pip install -r /opt/whisper-persian/requirements-whisper.txt; \
  elif [ "${WHISPER_IMPLEMENTATION}" = "faster-whisper" ]; then \
    python3 -m pip install -r /opt/whisper-persian/requirements-fasterWhisper.txt; \
  else \
    python3 -m pip install -r /opt/whisper-persian/requirements.txt; \
  fi

# Note: Models are downloaded on demand into /root/.cache. Bind that directory to the host to
# avoid re-downloading the Persian models on every container start.

# To be able to see logs in real time
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/whisper-persian/
ENTRYPOINT ["python3"]
CMD ["app.py", "--input_audio_max_duration", "-1", "--server_name", "0.0.0.0", "--auto_parallel", "True"]
