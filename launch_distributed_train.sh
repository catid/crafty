#!/bin/bash

deepspeed -H hostfile train_recon.py --deepspeed $@
