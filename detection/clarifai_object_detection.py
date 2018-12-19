import time

from clarifai.rest import ClarifaiApp, Concept

from detection.models import DetectionResult

app = ClarifaiApp( api_key = '00126f3a9bef4aa9a86e748292fc012b' )
model = app.public_models.general_model

PATH_TO_CONCEPTS = 'detection/ssd_mobilenet_v1_coco_2017_11_17/predefined_concepts.txt'
# PATH_TO_CONCEPTS = 'ssd_mobilenet_v1_coco_2017_11_17/predefined_concepts.txt'

with open( PATH_TO_CONCEPTS ) as file:
    predefined_concepts = list( line for line in (lines.strip() for lines in file) if line )


def clarifai_detection_object_in_image( base64_image ):
    start = time.time()

    select_concept_list = list()
    for predefined_concept in predefined_concepts:
        select_concept_list.append( Concept( concept_name = predefined_concept ) )

    response = model.predict_by_base64( base64_bytes = base64_image,
                                        select_concepts = select_concept_list )

    results = [ ]
    for clarifai_concept in response[ 'outputs' ][ 0 ][ 'data' ][ 'concepts' ]:
        print( str( clarifai_concept[ 'name' ] ) + ': ' + str( clarifai_concept[ 'value' ] ) )
        if clarifai_concept[ 'value' ] >= 0.1:
            results.append(
                DetectionResult( tag = clarifai_concept[ 'name' ], score = clarifai_concept[ 'value' ] ).__dict__ )

    end = time.time()
    print( 'Cl took: ' + str( end - start ) + ' with the return results: ' + str( response ) )
    return results


# import base64
#
# # with open( 'C:/Users/Notebook/Desktop/train_model/models/research/object_detection/images/000000402720.jpg',
# with open( 'C:/Users/Notebook/Desktop/46968800_256596061684774_4723312416018923520_n.jpg',
#            'rb' ) as image:
#     encoded_image = base64.b64encode( image.read() )
#
# clarifai_detection_object_in_image( encoded_image )
