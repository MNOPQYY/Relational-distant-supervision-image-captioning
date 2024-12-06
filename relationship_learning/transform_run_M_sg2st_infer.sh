python run_inference.py --batchsize 100 --out_name 'coco_inference_result.h5' --bestmodel 'bestmodel.pth'

cd ../relationship_to_sentence  

python run_inference.py --batchsize 200 --input_name '../relationship_learning/models/coco_inference_result.h5' --out_name './models/coco_gcc_pseudo_captions.json'

cd ../image_captioning
python preprocess_distdata.py --path '../relationship_to_sentence/models/coco_gcc_pseudo_captions.json'
python train.py