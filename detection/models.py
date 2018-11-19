from django.db import models

class File( models.Model ):
    file = models.FileField( blank = False, null = False )


class DetectionResult( object ):

    def __init__( self, tag, score ):
        self.tag = tag
        self.score = score
