#!/bin/bash
wget https://github.com/JokerYan/NeRF-DS/releases/download/v0.1-pre-release/NeRF-DS.dataset.zip
python -m zipfile -e NeRF-DS.dataset.zip .
rm NeRF-DS.dataset.zip