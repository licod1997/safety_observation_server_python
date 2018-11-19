from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from detection.serializers import FileSerializer, DetectionResultSerializer
from detection.object_detection import detect_object_in_image
from django.core.files.base import ContentFile
from django.views.decorators.csrf import csrf_exempt
from threading import Thread

import base64
import os
import detection.clarifai_object_detection as Cl
import detection.training_exporting as train


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


thread = Thread()

@api_view( [ 'GET' ] )
def training( request ):
    thread = Thread( target = train.train_export_model(), daemon = False )
    thread.start()
    return Response( status = status.HTTP_201_CREATED )


@api_view( [ 'GET' ] )
def check_thread_training( request ):
    if thread.is_alive():
        return Response( status = status.HTTP_201_CREATED )
    return Response( status = status.HTTP_404_NOT_FOUND )

