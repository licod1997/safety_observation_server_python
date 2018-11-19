import os
import tensorflow as tf
import numpy as np
import PIL
import time
import cv2

from io import BytesIO
from detection.models import DetectionResult
from utils import ops as utils_ops
from utils import label_map_util
from matplotlib import pyplot as plt
from utils import visualization_utils as vis_util
from tensorflow.python.client import device_lib

# Minimum ratio of detecting objects
MIN_SCORE_THRESH = 0.1

# Maximum number of objects detect in image
MAX_BOXES_TO_DRAW = 20

# What model to download.
MODEL_NAME = 'detection/ssd_mobilenet_v1_coco_2017_11_17'
# MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'

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

        # force CPU device placement for NMS ops (performance increase by xs)
        for node in od_graph_def.node:
            if 'BatchMultiClassNonMaxSuppression' in node.name:
                node.device = '/device:CPU:0'
            else:
                node.device = '/device:GPU:0'
        tf.import_graph_def( od_graph_def, name = '' )

sess = tf.Session( graph = detection_graph, config = tf.ConfigProto( allow_soft_placement = True ) )

category_index = label_map_util.create_category_index_from_labelmap( os.path.abspath( PATH_TO_LABELS ),
                                                                     use_display_name = True )

# use this when use ssd pretrained_model, which requires the shape of (?, ?, ?, 3)
# it takes quite long time to process (~0.4s per image)
# def load_image_into_numpy_array( image ):
#     (im_width, im_height) = image.size
#     return np.array( image.getdata() ).reshape(
#         (im_height, im_width, 3) ).astype( np.uint8 )

# In[ ]:

# def run_inference_for_single_image( image, graph ):
#     with graph.as_default():
#         # Get handles to input and output tensors
#         ops = tf.get_default_graph().get_operations()
#         all_tensor_names = { output.name for op in ops for output in op.outputs }
#         tensor_dict = { }
#         for key in [
#             'num_detections', 'detection_boxes', 'detection_scores',
#             'detection_classes', 'detection_masks'
#         ]:
#             tensor_name = key + ':0'
#             if tensor_name in all_tensor_names:
#                 tensor_dict[ key ] = tf.get_default_graph().get_tensor_by_name(
#                     tensor_name )
#         if 'detection_masks' in tensor_dict:
#             # The following processing is only for single image
#             detection_boxes = tf.squeeze( tensor_dict[ 'detection_boxes' ], [ 0 ] )
#             detection_masks = tf.squeeze( tensor_dict[ 'detection_masks' ], [ 0 ] )
#             # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
#             real_num_detection = tf.cast( tensor_dict[ 'num_detections' ][ 0 ], tf.int32 )
#             detection_boxes = tf.slice( detection_boxes, [ 0, 0 ], [ real_num_detection, -1 ] )
#             detection_masks = tf.slice( detection_masks, [ 0, 0, 0 ], [ real_num_detection, -1, -1 ] )
#             detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
#                 detection_masks, detection_boxes, image.shape[ 0 ], image.shape[ 1 ] )
#             detection_masks_reframed = tf.cast(
#                 tf.greater( detection_masks_reframed, 0.5 ), tf.uint8 )
#             # Follow the convention by adding back the batch dimension
#             tensor_dict[ 'detection_masks' ] = tf.expand_dims(
#                 detection_masks_reframed, 0 )
#         image_tensor = tf.get_default_graph().get_tensor_by_name( 'image_tensor:0' )
#
#         # Run inference
#         output_dict = sess.run( tensor_dict,
#                                 feed_dict = { image_tensor: np.expand_dims( image, 0 ) } )
#
#         # All outputs are float32 numpy arrays, so convert types as appropriate
#         output_dict[ 'num_detections' ] = int( output_dict[ 'num_detections' ][ 0 ] )
#         output_dict[ 'detection_classes' ] = output_dict[
#             'detection_classes' ][ 0 ].astype( np.uint8 )
#         output_dict[ 'detection_boxes' ] = output_dict[ 'detection_boxes' ][ 0 ]
#         output_dict[ 'detection_scores' ] = output_dict[ 'detection_scores' ][ 0 ]
#         if 'detection_masks' in output_dict:
#             output_dict[ 'detection_masks' ] = output_dict[ 'detection_masks' ][ 0 ]
#     return output_dict


# In[ ]:
print( device_lib.list_local_devices() )

# Definite input and output Tensors for detection_graph
image_tensor = detection_graph.get_tensor_by_name( 'image_tensor:0' )

# Each box represents a part of the image where a particular object was detected.
detection_boxes = detection_graph.get_tensor_by_name( 'detection_boxes:0' )

# Each score represent how level of confidence for each of the objects.
# Score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name( 'detection_scores:0' )
detection_classes = detection_graph.get_tensor_by_name( 'detection_classes:0' )
num_detections = detection_graph.get_tensor_by_name( 'num_detections:0' )


def detect_object_in_image( base64_image ):
    start = time.time()

    # Convert base64 string to numpy array
    image_np = np.asarray( PIL.Image.open( BytesIO( base64_image ) ), dtype = 'uint8' )
    # Make image_np writeble, but no need to use this if using load_image_into_numpy_array func
    image_np.flags.writeable = True

    # output_dict = run_inference_for_single_image( image_np, detection_graph )

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3], no need to
    # use load image into numpy with this
    image_np_expanded = np.expand_dims( image_np, axis = 0 )

    # Actual detection.
    (boxes, scores, classes, num) = sess.run(
        [ detection_boxes, detection_scores, detection_classes, num_detections ],
        feed_dict = { image_tensor: image_np_expanded } )

    squeeze_boxes = np.squeeze( boxes )
    squeeze_classes = np.squeeze( classes ).astype( np.int32 )
    squeeze_score = np.squeeze( scores )

    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        squeeze_boxes,
        squeeze_classes,
        squeeze_score,
        category_index,
        use_normalized_coordinates = True,
        line_thickness = 8,
        min_score_thresh = MIN_SCORE_THRESH,
        max_boxes_to_draw = MAX_BOXES_TO_DRAW
    )

    # Get detection classes and scores
    results = [ ]
    for i in range( 0, MAX_BOXES_TO_DRAW ):
        if squeeze_score[ i ] > MIN_SCORE_THRESH:
            tag = ''
            if squeeze_classes[ i ] in category_index.keys():
                tag = category_index[ squeeze_classes[ i ] ][ 'name' ]
            score = squeeze_score[ i ]
            result = DetectionResult( tag = tag, score = score ).__dict__
            results.append( result )
    end = time.time()

    print( 'TF took: ' + str( end - start ) + ' with the return results: ' + str( results ) )

    # Show image on screen
    converted_image = PIL.Image.fromarray( np.uint8( image_np ) )
    converted_image.show()

    return results

# with detection_graph.as_default():
#     with tf.Session(graph = detection_graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
#         cap = cv2.VideoCapture( 0 )
#         while cap.isOpened():
#             start = time.time()
#             ret, image_np = cap.read()
#             if ret == True:
#                 output_dict = run_inference_for_single_image(image_np, detection_graph)
#                 vis_util.visualize_boxes_and_labels_on_image_array(
#                     image_np,
#                     output_dict[ 'detection_boxes' ],
#                     output_dict[ 'detection_classes' ],
#                     output_dict[ 'detection_scores' ],
#                     category_index,
#                     instance_masks = output_dict.get( 'detection_masks' ),
#                     use_normalized_coordinates = True,
#                     line_thickness = 8,
#                     min_score_thresh = .5)
#                 cv2.imshow( 'Object Detecion', image_np )
#                 if cv2.waitKey( 1 ) & 0xFF == ord( 'q' ):
#                     cap.release()
#                     cv2.destroyAllWindows()
#                     break
#             end = time.time()
#             print('FPS: ' + str(1.0 / (end-start)))

# ~0.02-0.03s better performance
# with detection_graph.as_default():
#     with tf.Session(graph = detection_graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
#         cap = cv2.VideoCapture( 0 )
#
#         # Test optimize
#         # Definite input and output Tensors for detection_graph
#         image_tensor = detection_graph.get_tensor_by_name( 'image_tensor:0' )
#
#         # Each box represents a part of the image where a particular object was detected.
#         detection_boxes = detection_graph.get_tensor_by_name( 'detection_boxes:0' )
#
#         # Each score represent how level of confidence for each of the objects.
#         # Score is shown on the result image, together with the class label.
#         detection_scores = detection_graph.get_tensor_by_name( 'detection_scores:0' )
#         detection_classes = detection_graph.get_tensor_by_name( 'detection_classes:0' )
#         num_detections = detection_graph.get_tensor_by_name( 'num_detections:0' )
#
#         while cap.isOpened():
#             start = time.time()
#             ret, image_np = cap.read()
#
#             # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
#             image_np_expanded = np.expand_dims( image_np, axis = 0 )
#
#             # Actual detection.
#
#             (boxes, scores, classes, num) = sess.run(
#                 [ detection_boxes, detection_scores, detection_classes, num_detections ],
#                 feed_dict = { image_tensor: image_np_expanded } )
#
#             vis_util.visualize_boxes_and_labels_on_image_array(
#                 image_np,
#                 np.squeeze( boxes ),
#                 np.squeeze( classes ).astype( np.int32 ),
#                 np.squeeze( scores ),
#                 category_index,
#                 use_normalized_coordinates = True,
#                 line_thickness = 8,
#                 min_score_thresh = .1)
#             cv2.imshow( 'Object Detecion', image_np )
#             if cv2.waitKey( 1 ) & 0xFF == ord( 'q' ):
#                 cap.release()
#                 cv2.destroyAllWindows()
#                 break
#             end = time.time()
#             print('FPS: ' + str(1.0 / (end-start)))
