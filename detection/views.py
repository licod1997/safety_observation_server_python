from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from detection.serializers import FileSerializer, DetectionResultSerializer
from detection.object_detection import detect_object_in_image

import os


@api_view( [ 'POST' ] )
def upload_file( request ):
    data = request.data
    file_serializer = FileSerializer( data = data )
    if file_serializer.is_valid():
        file = file_serializer.save()
        results = detect_object_in_image( file.file.path )
        os.remove( file.file.path )
        results_serializer = DetectionResultSerializer( data = results, many = True )
        if results_serializer.is_valid():
            return Response( results_serializer.data, status = status.HTTP_201_CREATED )
    return Response( file_serializer.errors, status = status.HTTP_400_BAD_REQUEST )
