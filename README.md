# Project Cena
## Setup
- Install OpenCV
    - [Full instructions here](http://www.pyimagesearch.com/2016/12/19/install-opencv-3-on-macos-with-homebrew-the-easy-way/)
    - Basically:
        - `brew tap homebrew/science`
        - `brew install opencv3 --with-contrib --with-python3 --HEAD`
- `pip install -r requirements.txt`
- need to install dlib
- need to install openface
    - `git clone https://github.com/cmusatyalab/openface.git`
    - `cd openface`
    - `python setup.py install` 
- `./create_symlinks.sh` to symlink data from dropbox to local
- install torch http://torch.ch/docs/getting-started.html
    - add torch to your path
    - `luarocks install csvigo`
    - `luarocks install dpnn`
- download openface model nn4.small2.v1 here: http://cmusatyalab.github.io/openface/models-and-accuracies/
- brew install findutils
- brew install coreutils
- generate data
    - https://gist.github.com/ageitgey/ddbae3b209b6344a458fa41a3cf75719
    - cd into openface then `./batch-represent/main.lua -outDir ../../datascope/cena/data/generated-embeddings/ -data ../../datascope/cena/data/training-images/`
    
- install dlib on the rpi 
    - http://www.pyimagesearch.com/2017/05/01/install-dlib-raspberry-pi/