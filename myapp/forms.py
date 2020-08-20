from django import forms
from django.core import validators
from myapp.models import User
from .models import *

class Authentic(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput())

    class Meta:
        model = User
        fields =("username","password","first_name","last_name")




class UploadForm(forms.ModelForm):

    class Meta:
        model = scanupload
        fields = ['name', 'phone','upload_Main_Img','visiname']



class expectedvis(forms.ModelForm):

    class Meta:
        model = expectedvisitor
        fields = ['name', 'phone','no_of_visitors','expected_datetime_of_arrival']



class friendvis(forms.ModelForm):

    class Meta:
        model = friendvisitor
        fields = ['name', 'phone','image','role']


class otp(forms.ModelForm):

    class Meta:
        model = otp
        fields = ['name','phone', 'otp','visiname']



# class emg(forms.ModelForm):

#     class Meta:
#         model = emg
#         fields = ['image']        
      
