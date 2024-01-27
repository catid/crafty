#!/bin/bash

deepspeed train_recon.py --deepspeed --deepspeed_config deepspeed_config.json $@
