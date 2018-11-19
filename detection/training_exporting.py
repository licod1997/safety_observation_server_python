import subprocess

TRAINING_PATH = 'C:/Users/Notebook/Desktop/train_model/models/research/object_detection'


def train_export_model():
    try:
        subprocess.call( 'cd ' + TRAINING_PATH +
                         '&'
                         'python model_main.py --pipeline_config_path=training/faster_rcnn_resnet101_coco.config --model_dir=training/ --num_train_steps=70000 --logtostderr --sample_1_of_n_eval_examples=1'
                         '&'
                         'rmdir /s /q /dangerous_thing_graph'
                         '&'
                         'python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_resnet101_coco.config  --trained_checkpoint_prefix training/model.ckpt-50000 --output_directory dangerous_thing_graph',
                         shell = True )
    except OSError as e:
        print( e )

