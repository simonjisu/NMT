nohup python3 -u main.py \
    -root "../data/" \
    -dt "iwslt" \
    -minfreq 2 \
    -stp 30 \
    -maxlen 50 \
    -bs 20 \
    -pee 5 \
    -cuda \
    -emptymem \
    -hid 600 \
    -emd 300 \
    -enhl 3 \
    -dnhl 3 \
    -mth "general" \
    -drop 0.2 \
    -lnorm \
    -thres 5 \
    -lr 0.0001 \
    -tf \
    -declr 3.0 \
    -wdk 0.00001 \
    -optim "sgd" \
    -save \
    -savebest \
    -svpe "./saved_models/iswlt_big2.enc" \
    -svpd "./saved_models/iswlt_big2.dec" \
    -load \
    -ldpe "./saved_models/iswlt_big.enc" \
    -ldpd "./saved_models/iswlt_big.dec" > ../trainlog/nmt_big2.log &
