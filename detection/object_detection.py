import os
import tensorflow as tf
import numpy as np
import PIL.Image as Image

from detection.models import DetectionResult
from utils import ops as utils_ops
from utils import label_map_util

# Minimum ratio of detecting objects
THRESHOLD = 0.5

# Maximum number of objects detect in image
MAXIMUM_DETECTING_OBJECTS = 20

# What model to download.
MODEL_NAME = 'detection/ssd_mobilenet_v1_coco_2017_11_17'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join( MODEL_NAME, 'mscoco_label_map.pbtxt' )

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile( os.path.abspath( PATH_TO_FROZEN_GRAPH ), 'rb' ) as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString( serialized_graph )
        tf.import_graph_def( od_graph_def, name = '' )

category_index = label_map_util.create_category_index_from_labelmap( os.path.abspath( PATH_TO_LABELS ),
                                                                     use_display_name = True )


def load_image_into_numpy_array( image ):
    (im_width, im_height) = image.size
    return np.array( image.getdata() ).reshape(
        (im_height, im_width, 3) ).astype( np.uint8 )


def run_inference_for_single_image( image, graph ):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = { output.name for op in ops for output in op.outputs }
            tensor_dict = { }
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[ key ] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name )
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze( tensor_dict[ 'detection_boxes' ], [ 0 ] )
                detection_masks = tf.squeeze( tensor_dict[ 'detection_masks' ], [ 0 ] )
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast( tensor_dict[ 'num_detections' ][ 0 ], tf.int32 )
                detection_boxes = tf.slice( detection_boxes, [ 0, 0 ], [ real_num_detection, -1 ] )
                detection_masks = tf.slice( detection_masks, [ 0, 0, 0 ], [ real_num_detection, -1, -1 ] )
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[ 0 ], image.shape[ 1 ] )
                detection_masks_reframed = tf.cast(
                    tf.greater( detection_masks_reframed, 0.5 ), tf.uint8 )
                # Follow the convention by adding back the batch dimension
                tensor_dict[ 'detection_masks' ] = tf.expand_dims(
                    detection_masks_reframed, 0 )
            image_tensor = tf.get_default_graph().get_tensor_by_name( 'image_tensor:0' )

            # Run inference
            output_dict = sess.run( tensor_dict,
                                    feed_dict = { image_tensor: np.expand_dims( image, 0 ) } )

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict[ 'num_detections' ] = int( output_dict[ 'num_detections' ][ 0 ] )
            output_dict[ 'detection_classes' ] = output_dict[
                'detection_classes' ][ 0 ].astype( np.uint8 )
            output_dict[ 'detection_boxes' ] = output_dict[ 'detection_boxes' ][ 0 ]
            output_dict[ 'detection_scores' ] = output_dict[ 'detection_scores' ][ 0 ]
            if 'detection_masks' in output_dict:
                output_dict[ 'detection_masks' ] = output_dict[ 'detection_masks' ][ 0 ]
    return output_dict


# In[ ]:

def detect_object_in_image( image_path ):

    image = Image.open( image_path )
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array( image )
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    # image_np_expanded = np.expand_dims( image_np, axis = 0 )
    # Actual detection.
    output_dict = run_inference_for_single_image( image_np, detection_graph )

    results = [ ]

    for i in range( 0, MAXIMUM_DETECTING_OBJECTS ):
        tag = ''
        score = 0
        if output_dict[ 'detection_scores' ][ i ] > THRESHOLD:
            if output_dict[ 'detection_classes' ][ i ] in category_index.keys():
                tag = category_index[ output_dict[ 'detection_classes' ][ i ] ][ 'name' ]
            score = output_dict[ 'detection_scores' ][ i ]
            result = DetectionResult( tag = tag, score = score ).__dict__
            results.append( result )

    return results
    # In[ ]:
