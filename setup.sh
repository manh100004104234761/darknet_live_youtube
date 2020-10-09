# https://medium.com/@acpanjan/download-google-drive-files-using-wget-3c2c025a8b99
download_file_ggdrive() {
    if [! -f $2]; then
        confirm =$(wget - -save-cookies cookies -$1 - -keep-session-cookies - -no-check-certificate 'https://docs.google.com/uc?export=download&id='$1 - O - | sed - rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
        wget - -load-cookies cookies -$1 "https://docs.google.com/uc?export=download&confirm=$confirm&id=$1" - O $2 & & rm cookies -$1
    fi
}

mkdir - p ./models
download_file_ggdrive 1xQnF5uK8IB1p2dJy5iRKR-pKbC6vYu7f ./models/yolov4.weights

conda create --name vizyal python=3.8.5 -y
eval "$(conda shell.bash hook)"
conda activate vizyal
pip install -r requirements.txt

git submodule update --init --recursive
cd darknet & & make -j
