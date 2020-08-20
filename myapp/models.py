from django.db import models
from django.contrib.auth.models import User
from phone_field import PhoneField
from datetime import datetime

    





class scanupload(models.Model):
    CHOICES = (
        ('allow', 'Allow'),
        ('pending', 'Pending'),
    )
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=50, blank=True, null=True)
    phone = models.CharField(max_length=10, blank=True, null=True)
    upload_Main_Img = models.ImageField( blank=True, null=True)
    visiname = models.CharField(max_length=50, blank=True, null=True)
    date = models.DateTimeField(default=datetime.now, blank=True)
    status =  models.CharField(max_length=50, choices= CHOICES, blank=True, default = "pending")



class expectedvisitor(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=50, blank=True, null=True)
    phone = models.CharField(max_length=10, blank=True, null=True)
    no_of_visitors = models.CharField(max_length=10, blank=True, null=True)
    expected_datetime_of_arrival = models.DateTimeField( blank=True)
    user = models.ForeignKey(User, to_field="username", on_delete=models.PROTECT, default="vijay")
    date = models.DateTimeField(default=datetime.now, blank=True)


class friendvisitor(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=50, blank=True, null=True)
    phone = models.CharField(max_length=10, blank=True, null=True)
    image = models.ImageField( blank=True, null=True)
    role = models.CharField(max_length=50, blank=True, null=True, default="friend")

    user = models.ForeignKey(User, to_field="username", on_delete=models.PROTECT, default="vijay")
    date = models.DateTimeField(default=datetime.now, blank=True)


class otp(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=50, blank=True, null=True)
    phone = models.CharField(max_length=10, blank=True, null=True)
    otp = models.CharField(max_length=5, blank=True, null=True)
    visiname = models.CharField( max_length=5, blank=True, null=True)
