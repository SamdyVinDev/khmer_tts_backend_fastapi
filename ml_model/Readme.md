# Khmer TTS Python Project
==============

## Before running: 
- Create a python virtual environment
- Install all dependencies `pip install -r requirements.txt`
- please create folder model
- download waveglow pretain model and khmerTTS model
      from : https://drive.google.com/drive/folders/17ne_fgdQk9AYdW67qt9mWphfsdRhvTAU?usp=sharing
- put khmerTTS model and waveglow model into model folder
- create folder result/plot and result/sound in example folder


+ After download:
    - go to synthesis.py file
    - set path for
        - waveglow_model
        - tacotron2_model

+ How to run :
    - write: python synthesis.py

+ Check result :
    - after synthesis sound file will store in example/result/sound
    - after synthesis plot graph will store in example/result/plot
    - you can put the sound file in other directory , but make sure to change to location in synthesis.py file
