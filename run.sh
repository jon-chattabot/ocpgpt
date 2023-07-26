#!/bin/bash

# CHAINLIT_HOST and CHAINLIT_PORT
# --host and --port when running chainlit run ....
# nohup ./run.sh &> ocpgpt.log &
# nohup chainlit run app.py &> ocpgpt.log &
chainlit run app.py
