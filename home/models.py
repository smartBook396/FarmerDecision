from django.db import models

#Define our Seeds class 
class Seeds(models.Model): 
    gain = models.DecimalField()
    irrigation = 