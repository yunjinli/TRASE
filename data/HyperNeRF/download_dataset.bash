#!/bin/bash
wget https://github.com/google/hypernerf/releases/download/v0.1/interp_chickchicken.zip
wget https://github.com/google/hypernerf/releases/download/v0.1/interp_cut-lemon.zip
wget https://github.com/google/hypernerf/releases/download/v0.1/interp_hand.zip
wget https://github.com/google/hypernerf/releases/download/v0.1/interp_slice-banana.zip
wget https://github.com/google/hypernerf/releases/download/v0.1/interp_torchocolate.zip
wget https://github.com/google/hypernerf/releases/download/v0.1/misc_americano.zip
wget https://github.com/google/hypernerf/releases/download/v0.1/misc_espresso.zip
wget https://github.com/google/hypernerf/releases/download/v0.1/misc_keyboard.zip
wget https://github.com/google/hypernerf/releases/download/v0.1/misc_oven-mitts.zip
wget https://github.com/google/hypernerf/releases/download/v0.1/misc_split-cookie.zip

python -m zipfile -e interp_chickchicken.zip .
python -m zipfile -e interp_cut-lemon.zip .
python -m zipfile -e interp_hand.zip .
python -m zipfile -e interp_slice-banana.zip .
python -m zipfile -e interp_torchocolate.zip .
python -m zipfile -e misc_americano.zip .
python -m zipfile -e misc_espresso.zip .
python -m zipfile -e misc_keyboard.zip .
python -m zipfile -e misc_oven-mitts.zip .
python -m zipfile -e misc_split-cookie.zip .

rm interp_chickchicken.zip 
rm interp_cut-lemon.zip 
rm interp_hand.zip 
rm interp_slice-banana.zip 
rm interp_torchocolate.zip 
rm misc_americano.zip 
rm misc_espresso.zip 
rm misc_keyboard.zip 
rm misc_oven-mitts.zip 
rm misc_split-cookie.zip 

mkdir interp
mkdir misc

mv chickchicken ./interp
mv cut-lemon1 ./interp
mv hand1-dense-v2 ./interp
mv slice-banana ./interp
mv torchocolate ./interp

mv americano ./misc
mv espresso ./misc
mv keyboard ./misc
mv oven-mitts ./misc
mv split-cookie ./misc
