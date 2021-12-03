
pushd datasets

# whole dataset:
curl https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz --output mvtec.tar.xz

# metal nut only:
#curl https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937637-1629952063/metal_nut.tar.xz --output mvtec.tar.xz

tar -xvf mvtec.tar.xz

popd

