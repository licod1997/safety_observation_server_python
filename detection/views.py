import base64

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

import detection.clarifai_object_detection as Cl
from detection.object_detection import detect_object_in_image
from detection.serializers import DetectionResultSerializer


@api_view( [ 'POST' ] )
def upload_file( request ):
    base64_data = request.data[ 'file' ]
    decode_image = base64.b64decode( base64_data )
    encode_image = base64.b64encode( decode_image )

    results = detect_object_in_image( decode_image )
    if len( results ) < 1:
        results = Cl.clarifai_detection_object_in_image( encode_image )

    results_serializer = DetectionResultSerializer( data = results, many = True )
    if results_serializer.is_valid():
        return Response( results_serializer.data, status = status.HTTP_201_CREATED )
    return Response( results_serializer.errors, status = status.HTTP_400_BAD_REQUEST )



