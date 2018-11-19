import time
from detection.models import DetectionResult
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image

app = ClarifaiApp( api_key = '00126f3a9bef4aa9a86e748292fc012b' )
model = app.public_models.general_model

# PATH_TO_CONCEPTS = 'detection/ssd_mobilenet_v1_coco_2017_11_17/predefined_concepts.txt'
PATH_TO_CONCEPTS = 'ssd_mobilenet_v1_coco_2017_11_17/predefined_concepts.txt'


def clarifai_detection_object_in_image( base64_image ):
    start = time.time()
    with open( PATH_TO_CONCEPTS ) as file:
        predefined_concepts = list( line for line in (lines.strip() for lines in file) if line )

    response = model.predict( [ Image( base64 = base64_image ) ] )

    results = [ ]
    for clarifai_concept in response[ 'outputs' ][ 0 ][ 'data' ][ 'concepts' ]:
        for predefined_concept in predefined_concepts:
            print( clarifai_concept[ 'name' ] )
            if clarifai_concept[ 'name' ] == predefined_concept and clarifai_concept[ 'value' ] >= 0.3:
                results.append(
                    DetectionResult( tag = clarifai_concept[ 'name' ], score = clarifai_concept[ 'value' ] ).__dict__ )

    end = time.time()
    print( 'Cl took: ' + str( end - start ) + ' with the return results: ' + str( response ) )
    return results


import base64

with open( 'C:/Users/Notebook/Desktop/train_model/models/research/object_detection/images/scissors_387.jpg',
           'rb' ) as image:
    encoded_image = base64.b64encode( image.read() )

clarifai_detection_object_in_image( encoded_image )
